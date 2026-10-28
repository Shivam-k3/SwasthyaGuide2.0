"""Unit tests for the RSNA CNN baseline (scripts/rsna/cnn_*).

All fixtures are synthetic mini dataset trees built with the same helpers as
tests/test_preprocess.py; no RSNA data, no pretrained-weight downloads, and
no DenseNet training happen here (the training loop is exercised with an
injected tiny linear model; DenseNet is only built with pretrained=False for
a single forward-shape check).
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts' / 'rsna'
sys.path.insert(0, str(SCRIPTS))

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - environment guard
    pytest.skip('torch not installed', allow_module_level=True)

import cnn_data
import cnn_metrics
import cnn_train  # noqa: F401  (import surface check)
from test_preprocess import _write_csv, make_dicom


# ------------------------------------------------------------ fixtures
def build_cnn_base(tmp_path):
    """13 synthetic images: per split 2 pos + 2 neg (labelled) plus 1
    unlabelled in train; processed into a npy cache via the real pipeline."""
    base = tmp_path / 'rsna'
    imgroot = base / 'raw' / 'images' / 'study'
    imgroot.mkdir(parents=True)
    rng = np.random.default_rng(7)
    specs = ([('train', '1', 'Lung Opacity')] * 2 + [('train', '0', 'Normal')] * 2
             + [('train', '', '')]
             + [('validation', '1', 'Lung Opacity')] * 2
             + [('validation', '0', 'Normal')] * 2
             + [('test', '1', 'Lung Opacity')] * 2 + [('test', '0', 'Normal')] * 2)
    split_rows, meta_rows = [], []
    for i, (split, target, label) in enumerate(specs, 1):
        image_id = f'1.2.276.0.7230010.3.1.4.2000.{i}'
        make_dicom(imgroot / f'c{i}.dcm',
                   rng.integers(0, 256, (16, 16), dtype=np.uint8))
        split_rows.append({'image_id': image_id, 'patient_id': f'{i:08d}',
                           'dicom_patient_id': f'uuid-{i}', 'study_id': f'study-{i}',
                           'target': target, 'label_class': label,
                           'bbox_count': '1' if target == '1' else '0',
                           'split': split})
        meta_rows.append({'image_id': image_id,
                           'rel_path': f'images/study/c{i}.dcm', 'file_size': '0'})
    man = base / 'manifests'
    man.mkdir()
    _write_csv(man / 'split_manifest.csv',
               ['image_id', 'patient_id', 'dicom_patient_id', 'study_id',
                'target', 'label_class', 'bbox_count', 'split'], split_rows)
    _write_csv(man / 'rsna_manifest.csv', ['image_id', 'rel_path', 'file_size'],
               meta_rows)
    import hashlib
    from test_preprocess import pp as preprocess_module
    sizes = [(imgroot / f'c{i}.dcm').stat().st_size for i in range(1, len(specs) + 1)]
    (man / 'raw_fingerprint.json').write_text(json.dumps({
        'file_count': len(sizes), 'total_bytes': sum(sizes),
        'sizes_sha256': hashlib.sha256(
            ','.join(map(str, sizes)).encode()).hexdigest()}))
    ids = [r['image_id'] for r in split_rows]
    summary = preprocess_module.run(str(base), ids, 'explicit')
    assert summary['failed'] == 0
    return base, split_rows


def read_proc_manifest(base):
    with open(base / 'proc' / 'proc_manifest.csv', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def rewrite_proc_manifest(base, transform):
    path = base / 'proc' / 'proc_manifest.csv'
    rows = read_proc_manifest(base)
    transform(rows)
    _write_csv(path, ['image_id', 'patient_id', 'study_id', 'split', 'target',
                      'label_class', 'bbox_count', 'out_rel_path', 'shape',
                      'dtype', 'src_sha256', 'config_version', 'origin',
                      'status', 'error'], rows)


# ------------------------------------------------------------- data tests
def test_loader_excludes_unlabelled_and_never_labels_them_negative(tmp_path):
    base, split_rows = build_cnn_base(tmp_path)
    rows = cnn_data.load_split_rows(str(base))
    assert {s: len(v) for s, v in rows.items()} == {
        'train': 4, 'validation': 4, 'test': 4}
    unlabelled_id = next(r['image_id'] for r in split_rows if r['target'] == '')
    all_loaded = [r for v in rows.values() for r in v]
    assert unlabelled_id not in {r['image_id'] for r in all_loaded}
    assert all(r['target'] in ('0', '1') for r in all_loaded)


def test_loader_rejects_invalid_target_values(tmp_path):
    base, _ = build_cnn_base(tmp_path)

    def corrupt(rows):
        rows[0]['target'] = '2'
    rewrite_proc_manifest(base, corrupt)
    with pytest.raises(ValueError, match='invalid target'):
        cnn_data.load_split_rows(str(base))


def test_loader_fails_clearly_on_missing_cache_files(tmp_path):
    base, _ = build_cnn_base(tmp_path)
    rows = cnn_data.load_split_rows(str(base))
    victim = (base / 'proc' / rows['train'][0]['out_rel_path']).resolve()
    victim.unlink()
    with pytest.raises(ValueError, match='missing'):
        cnn_data.RSNADataset(str(base / 'proc'), rows['train'])


def test_loader_rejects_patient_leakage_across_splits(tmp_path):
    base, _ = build_cnn_base(tmp_path)

    def leak(rows):
        val = next(r for r in rows if r['split'] == 'validation')
        tr = next(r for r in rows if r['split'] == 'train')
        val['patient_id'] = tr['patient_id']
    rewrite_proc_manifest(base, leak)
    with pytest.raises(RuntimeError, match='leakage'):
        cnn_data.load_split_rows(str(base))


def test_dataset_item_contract(tmp_path):
    base, _ = build_cnn_base(tmp_path)
    rows = cnn_data.load_split_rows(str(base))
    ds = cnn_data.RSNADataset(str(base / 'proc'), rows['train'])
    x, y = ds[0]
    assert x.shape == (1, 224, 224) and x.dtype == torch.float32
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0
    assert y.dtype == torch.float32 and float(y) in (0.0, 1.0)
    assert len(ds) == 4


def test_pos_weight_matches_label_counts(tmp_path):
    base, _ = build_cnn_base(tmp_path)
    rows = cnn_data.load_split_rows(str(base))
    n_pos, n_neg = cnn_data.label_counts(rows['train'])
    assert (n_pos, n_neg) == (2, 2)
    assert cnn_data.pos_weight_value(rows['train']) == 1.0


def test_stratified_limit_keeps_both_classes(tmp_path):
    base, _ = build_cnn_base(tmp_path)
    rows = cnn_data.load_split_rows(str(base))
    limited = cnn_data.stratified_limit(rows['train'], 1)
    assert len(limited) == 2
    assert {r['target'] for r in limited} == {'0', '1'}


# ---------------------------------------------------------- metrics tests
def test_metrics_known_values():
    m = cnn_metrics.compute_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 0.5)
    assert m['auroc'] == pytest.approx(0.75)
    assert m['sensitivity'] == pytest.approx(0.5)
    assert m['specificity'] == pytest.approx(1.0)
    assert m['precision'] == pytest.approx(1.0)
    assert m['recall'] == pytest.approx(0.5)
    assert m['f1'] == pytest.approx(2 / 3)
    assert m['confusion_matrix']['matrix'] == [[2, 0], [1, 1]]


def test_metrics_reject_single_class_and_bad_probabilities():
    with pytest.raises(ValueError, match='both classes'):
        cnn_metrics.compute_metrics([0, 0], [0.2, 0.3], 0.5)
    with pytest.raises(ValueError, match='finite'):
        cnn_metrics.compute_metrics([0, 1], [0.2, float('nan')], 0.5)
    with pytest.raises(ValueError, match='within'):
        cnn_metrics.compute_metrics([0, 1], [0.2, 1.4], 0.5)


def test_threshold_selection_youden():
    t = cnn_metrics.select_threshold_youden([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert t == pytest.approx(0.8)      # perfect separation; lowest such cut
    with pytest.raises(ValueError, match='both classes'):
        cnn_metrics.select_threshold_youden([1, 1], [0.2, 0.4])


# ----------------------------------------------------------- model tests
def test_build_model_forward_shape_random_init():
    import cnn_model
    m = cnn_model.build_model(pretrained=False)   # no download, random init
    x = torch.randn(2, 1, 224, 224)
    m.eval()
    with torch.no_grad():
        out = m(x)
    assert out.shape == (2, 1)
    assert torch.isfinite(out).all()


def test_pneumonia_net_rejects_multichannel_input():
    import cnn_model
    net = cnn_model.PneumoniaNet(nn.Linear(3, 1))
    with pytest.raises(ValueError, match='grayscale'):
        net(torch.randn(2, 3, 4, 4))


# ------------------------------------------------- training-loop tests
class TinyNet(nn.Module):
    """Stand-in model: accepts (B, 1, 224, 224), one logit out."""

    def __init__(self):
        super().__init__()
        self.flat = nn.Flatten()
        self.head = nn.Linear(224 * 224, 1)

    def forward(self, x):
        return self.head(self.flat(x))


def test_train_model_end_to_end_with_tiny_model(tmp_path):
    base, _ = build_cnn_base(tmp_path)
    torch.manual_seed(0)
    rows = cnn_data.load_split_rows(str(base))
    cfg = cnn_train.TrainConfig(rsna_base=str(base), out_dir=str(tmp_path / 'run'),
                                epochs=2, batch_size=4, num_workers=0,
                                device='cpu', pretrained=False)
    loaders = {s: cnn_train.make_loader(
        cnn_data.RSNADataset(str(base / 'proc'), v), cfg, shuffle=(s == 'train'))
        for s, v in rows.items()}
    history = cnn_train.train_model(TinyNet(), loaders, cfg, cfg.out_dir)

    assert len(history) == 2
    assert all(h['train_loss'] == h['train_loss'] for h in history)  # finite
    for name in ('config.json', 'history.json', 'best.pt', 'last.pt',
                 'threshold.json', 'val_metrics.json', 'test_metrics.json'):
        assert (tmp_path / 'run' / name).is_file(), name
    threshold = json.loads((tmp_path / 'run' / 'threshold.json').read_text())
    assert threshold['selected_on'] == 'validation'
    test_m = json.loads((tmp_path / 'run' / 'test_metrics.json').read_text())
    assert test_m['n'] == 4
    assert sum(sum(row) for row in test_m['confusion_matrix']['matrix']) == 4
    for key in ('auroc', 'sensitivity', 'specificity', 'precision', 'recall'):
        assert key in test_m


def test_run_epoch_raises_on_non_finite_loss(tmp_path, monkeypatch):
    base, _ = build_cnn_base(tmp_path)
    rows = cnn_data.load_split_rows(str(base))
    cfg = cnn_train.TrainConfig(rsna_base=str(base), out_dir=str(tmp_path),
                                batch_size=4, num_workers=0, device='cpu')

    class NaNNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))   # optimizer needs a param

        def forward(self, x):
            return torch.full((x.size(0), 1), float('nan'))

    nan_net = NaNNet()
    loader = cnn_train.make_loader(
        cnn_data.RSNADataset(str(base / 'proc'), rows['train']), cfg, shuffle=False)
    with pytest.raises(RuntimeError, match='non-finite'):
        cnn_train.run_epoch(nan_net, loader,
                            torch.optim.SGD(nan_net.parameters(), lr=0.0),
                            nn.BCEWithLogitsLoss(), torch.device('cpu'))
