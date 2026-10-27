"""Tests for the RSNA preprocessing pipeline.

The pipeline lives in ``scripts/rsna/`` in this repository (the RSNA dataset
itself stays outside the repo and is located via ``RSNA_BASE``; the tests
never need it). Everything runs against small synthetic DICOMs built in
``tmp_path``.

Nothing here touches the cardiovascular module.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts' / 'rsna'

if not (SCRIPTS / 'preprocess.py').is_file():
    pytest.fail(f'repository RSNA scripts missing at {SCRIPTS}',
                allow_module_level=True)


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pp = _load_module('rsna_preprocess', 'preprocess.py')
vp = _load_module('rsna_verify_proc', 'verify_proc.py')


# --------------------------------------------------------------- fixtures
def make_dicom(path, arr, *, photo='MONOCHROME2', ts='1.2.840.10008.1.2.1',
               signed=False, spp=1, pixel_bytes=None, rescale=None):
    """Write a minimal but structurally valid single-frame DICOM.

    Compressed transfer syntaxes get encapsulated framing (a single dummy
    fragment wrapping the payload), as DICOM requires for compressed streams.
    """
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.encaps import encapsulate

    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7'
    ds.file_meta.MediaStorageSOPInstanceUID = '1.2.276.0.7230010.3.1.4.9.1'
    ds.file_meta.TransferSyntaxUID = ts
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = '1.2.276.0.7230010.3.1.4.9.1'
    ds.PatientID = 'SYNTHETIC'
    ds.StudyInstanceUID = '1.2.276.0.7230010.3.1.2.9.1'
    ds.SeriesInstanceUID = '1.2.276.0.7230010.3.1.3.9.1'
    ds.Modality = 'CR'
    ds.Rows, ds.Columns = arr.shape
    ds.SamplesPerPixel = spp
    ds.PhotometricInterpretation = photo
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 1 if signed else 0
    payload = pixel_bytes if pixel_bytes is not None else arr.tobytes()
    ds.PixelData = (encapsulate([payload]) if ts.startswith('1.2.840.10008.1.2.4.')
                    else payload)
    if rescale is not None:
        ds.RescaleSlope, ds.RescaleIntercept = rescale
    ds.save_as(str(path), enforce_file_format=True)
    return str(path)


def gradient(size=32):
    """A non-constant uint8 test image with spatial structure."""
    return np.tile(np.linspace(10, 240, size, dtype=np.uint8), (size, 1))


def build_fake_base(tmp_path, *, include_missing_source=False):
    """A miniature dataset tree: 5 images across splits, classes, and one
    unlabelled record; optionally one inventory row whose file never exists."""
    base = tmp_path / 'rsna'
    imgroot = base / 'raw' / 'images' / 'study'
    imgroot.mkdir(parents=True)
    specs = [
        ('train', '1', 'Lung Opacity'),
        ('train', '0', 'Normal'),
        ('train', '', ''),                       # unlabelled
        ('validation', '0', 'No Lung Opacity / Not Normal'),
        ('test', '1', 'Lung Opacity'),
    ]
    if include_missing_source:
        specs.append(('test', '0', 'Normal'))    # row exists, file never written

    rng = np.random.default_rng(0)
    split_rows, meta_rows = [], []
    for i, (split, target, label) in enumerate(specs, 1):
        image_id = f'1.2.276.0.7230010.3.1.4.1000.{i}'
        rel = f'images/study/img{i}.dcm'
        if not (include_missing_source and i == len(specs)):
            make_dicom(imgroot / f'img{i}.dcm',
                       rng.integers(0, 256, (16, 16), dtype=np.uint8))
        split_rows.append({'image_id': image_id, 'patient_id': f'{i:08d}',
                           'dicom_patient_id': f'uuid-{i}', 'study_id': f'study-{i}',
                           'target': target, 'label_class': label,
                           'bbox_count': '2' if target == '1' else '0', 'split': split})
        meta_rows.append({'image_id': image_id, 'rel_path': rel,
                          'file_size': '0'})

    man = base / 'manifests'
    man.mkdir()
    _write_csv(man / 'split_manifest.csv',
               ['image_id', 'patient_id', 'dicom_patient_id', 'study_id',
                'target', 'label_class', 'bbox_count', 'split'], split_rows)
    _write_csv(man / 'rsna_manifest.csv', ['image_id', 'rel_path', 'file_size'],
               meta_rows)
    # fingerprint over files that actually exist (the missing-source fixture
    # intentionally has one inventory row with no file on disk)
    existing = [r for r in split_rows
                if (base / 'raw' / 'images' / 'study' /
                    f"img{split_rows.index(r) + 1}.dcm").is_file()]
    sizes = [os.path.getsize(base / 'raw' / 'images' / 'study' /
                             f"img{split_rows.index(r) + 1}.dcm")
             for r in existing]
    import hashlib
    fp = {'file_count': len(sizes), 'total_bytes': sum(sizes),
          'sizes_sha256': hashlib.sha256(
              ','.join(map(str, sizes)).encode()).hexdigest()}
    (man / 'raw_fingerprint.json').write_text(json.dumps(fp))
    return base, [r['image_id'] for r in split_rows]


def _write_csv(path, cols, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def read_proc_manifest(base):
    with open(base / 'proc' / 'proc_manifest.csv', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def run_ids(base, ids, **kw):
    return pp.run(str(base), list(ids), 'explicit', **kw)


# ------------------------------------------------------ decode-level tests
def test_valid_monochrome2_dicom_produces_expected_output(tmp_path):
    src = make_dicom(tmp_path / 'ok.dcm', gradient())
    out = pp.decode_and_preprocess(src)
    assert out.shape == (224, 224)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert np.unique(out).size >= 2          # nonempty, not a blank substitute


def test_output_is_resized_not_cropped_or_padded(tmp_path):
    # left half dark, right half bright; after resize the same halves hold
    arr = np.zeros((32, 32), dtype=np.uint8)
    arr[:, 16:] = 220
    out = pp.decode_and_preprocess(make_dicom(tmp_path / 'h.dcm', arr))
    assert out[:, :100].mean() < 0.2
    assert out[:, 130:].mean() > 0.7


def test_monochrome1_is_inverted_to_monochrome2_polarity(tmp_path):
    arr = gradient()
    out2 = pp.decode_and_preprocess(make_dicom(tmp_path / 'm2.dcm', arr,
                                               photo='MONOCHROME2'))
    out1 = pp.decode_and_preprocess(make_dicom(tmp_path / 'm1.dcm', arr,
                                               photo='MONOCHROME1'))
    assert np.allclose(out1 + out2, 1.0, atol=3 / 255)   # inverted polarity


def test_preprocessing_is_deterministic(tmp_path):
    src = make_dicom(tmp_path / 'd.dcm', gradient())
    a = pp.decode_and_preprocess(src)
    b = pp.decode_and_preprocess(src)
    assert np.array_equal(a, b)


def test_missing_source_reports_file_missing(tmp_path):
    with pytest.raises(pp.PreprocessError) as ei:
        pp.decode_and_preprocess(str(tmp_path / 'absent.dcm'))
    assert ei.value.category == 'FILE_MISSING'


def test_corrupt_dicom_reports_decode_error(tmp_path):
    p = tmp_path / 'corrupt.dcm'
    p.write_bytes(b'this is not a dicom file at all')
    with pytest.raises(pp.PreprocessError) as ei:
        pp.decode_and_preprocess(str(p))
    assert ei.value.category == 'DECODE_ERROR'


def test_unsupported_transfer_syntax_is_rejected(tmp_path):
    # JPEG 2000: supported DICOM syntax, but not in this pipeline's policy set
    src = make_dicom(tmp_path / 'j2k.dcm', gradient(),
                     ts='1.2.840.10008.1.2.4.91')
    with pytest.raises(pp.PreprocessError) as ei:
        pp.decode_and_preprocess(src)
    assert ei.value.category == 'UNSUPPORTED_TRANSFER_SYNTAX'


def test_supported_syntax_with_undecodable_data_reports_decode_error(tmp_path):
    # JPEG Baseline is in policy, but the payload is not a JPEG stream
    arr = gradient(8)
    src = make_dicom(tmp_path / 'fakejpg.dcm', arr,
                     ts='1.2.840.10008.1.2.4.50',
                     pixel_bytes=bytes(range(64)))
    with pytest.raises(pp.PreprocessError) as ei:
        pp.decode_and_preprocess(src)
    assert ei.value.category == 'DECODE_ERROR'


def test_truncated_pixel_data_reports_metadata_error(tmp_path):
    arr = gradient(8)                       # 64 bytes expected
    src = make_dicom(tmp_path / 'short.dcm', arr, pixel_bytes=arr.tobytes()[:37])
    with pytest.raises(pp.PreprocessError) as ei:
        pp.decode_and_preprocess(src)
    assert ei.value.category == 'PIXEL_METADATA_ERROR'


def test_rgb_photometric_is_rejected(tmp_path):
    arr = gradient()
    src = make_dicom(tmp_path / 'rgb.dcm', arr, photo='RGB', spp=3)
    with pytest.raises(pp.PreprocessError) as ei:
        pp.decode_and_preprocess(src)
    assert ei.value.category == 'PIXEL_METADATA_ERROR'


def test_signed_pixel_representation_is_rejected(tmp_path):
    src = make_dicom(tmp_path / 'signed.dcm', gradient(), signed=True)
    with pytest.raises(pp.PreprocessError) as ei:
        pp.decode_and_preprocess(src)
    assert ei.value.category == 'PIXEL_METADATA_ERROR'


def test_non_identity_rescale_is_rejected(tmp_path):
    src = make_dicom(tmp_path / 'rescale.dcm', gradient(), rescale=(2.0, -1024.0))
    with pytest.raises(pp.PreprocessError) as ei:
        pp.decode_and_preprocess(src)
    assert ei.value.category == 'PIXEL_METADATA_ERROR'


# ------------------------------------------------- run/Cache-level tests
def test_run_creates_cache_and_verifier_passes(tmp_path):
    base, ids = build_fake_base(tmp_path)
    summary = run_ids(base, ids)
    assert summary['failed'] == 0 and summary['processed'] == len(ids)

    rows = read_proc_manifest(base)
    assert [r['image_id'] for r in rows] == ids
    for r, i in zip(rows, ids):
        assert r['status'] == 'ok' and r['error'] == ''
        assert (base / 'proc' / r['out_rel_path']).is_file()

    unlabelled = next(r for r in rows if r['target'] == '')
    assert unlabelled['label_class'] == '' and unlabelled['status'] == 'ok'
    assert vp.main([str(base)]) == 0


def test_unlabelled_targets_survive_the_pipeline(tmp_path):
    base, ids = build_fake_base(tmp_path)
    run_ids(base, ids)
    frozen = {r['image_id']: r['target'] for r in
              csv.DictReader(open(base / 'manifests' / 'split_manifest.csv',
                                  encoding='utf-8'))}
    for r in read_proc_manifest(base):
        assert r['target'] == frozen[r['image_id']]   # '' stays ''


def test_cache_reuse_avoids_rewriting_outputs(tmp_path):
    base, ids = build_fake_base(tmp_path)
    run_ids(base, ids)
    outputs = {r['image_id']: (base / 'proc' / r['out_rel_path'])
               for r in read_proc_manifest(base)}
    mtimes = {i: p.stat().st_mtime_ns for i, p in outputs.items()}

    summary = run_ids(base, ids)
    assert summary['reused'] == len(ids) and summary['processed'] == 0
    assert {i: p.stat().st_mtime_ns for i, p in outputs.items()} == mtimes


def test_changed_source_invalidates_the_cache_entry(tmp_path):
    base, ids = build_fake_base(tmp_path)
    run_ids(base, ids)

    victim = base / 'raw' / 'images' / 'study' / 'img1.dcm'
    make_dicom(victim, gradient(64))          # same path, different content
    summary = run_ids(base, ids)
    assert summary['processed'] == 1 and summary['reused'] == len(ids) - 1


def test_config_version_change_invalidates_the_whole_cache(tmp_path, monkeypatch):
    base, ids = build_fake_base(tmp_path)
    run_ids(base, ids)
    monkeypatch.setattr(pp, 'CONFIG_VERSION', pp.CONFIG_VERSION + 1)
    summary = run_ids(base, ids)
    assert summary['processed'] == len(ids) and summary['reused'] == 0


def test_truncated_output_is_reprocessed(tmp_path):
    base, ids = build_fake_base(tmp_path)
    run_ids(base, ids)
    victim = base / 'proc' / 'images' / 'train' / f'{ids[0]}.npy'
    with open(victim, 'r+b') as f:
        f.truncate(64)                        # simulate an interrupted write

    summary = run_ids(base, ids)
    assert summary['processed'] == 1 and summary['reused'] == len(ids) - 1
    assert vp.main([str(base)]) == 0          # cache is whole again


def test_missing_source_is_a_recorded_failure_never_a_blank(tmp_path):
    base, ids = build_fake_base(tmp_path, include_missing_source=True)
    summary = run_ids(base, ids)
    assert summary['failed'] == 1
    assert summary['failures'][0]['category'] == 'FILE_MISSING'

    rows = {r['image_id']: r for r in read_proc_manifest(base)}
    bad = rows[ids[-1]]
    assert bad['status'] == 'failed'
    assert bad['error'].startswith('FILE_MISSING:')
    assert not list((base / 'proc' / 'images').rglob(f'{ids[-1]}.npy'))
    # the verifier refuses to bless an incomplete cache and names the failure
    assert vp.main([str(base)]) == 1


def test_verifier_flags_partial_tmp_outputs(tmp_path):
    base, ids = build_fake_base(tmp_path)
    run_ids(base, ids)
    stray = base / 'proc' / 'images' / 'train' / 'interrupted.npy.tmp'
    stray.write_bytes(b'\x00' * 16)
    assert vp.main([str(base)]) == 1


def test_verifier_flags_duplicate_manifest_ids(tmp_path):
    base, ids = build_fake_base(tmp_path)
    run_ids(base, ids)
    mpath = base / 'proc' / 'proc_manifest.csv'
    rows = read_proc_manifest(base)
    _write_csv(mpath, pp.PROC_MANIFEST_COLS, rows + [dict(rows[0])])
    assert vp.main([str(base)]) == 1


def test_verifier_flags_split_mismatch(tmp_path):
    base, ids = build_fake_base(tmp_path)
    run_ids(base, ids)
    mpath = base / 'proc' / 'proc_manifest.csv'
    rows = read_proc_manifest(base)
    rows[0]['split'] = 'test'                 # tamper: no longer the frozen split
    _write_csv(mpath, pp.PROC_MANIFEST_COLS, rows)
    assert vp.main([str(base)]) == 1


def test_verifier_flags_policy_drift_in_config(tmp_path):
    base, ids = build_fake_base(tmp_path)
    run_ids(base, ids)
    cpath = base / 'proc' / 'config.json'
    config = json.loads(cpath.read_text())
    config['policy']['target_size'] = [512, 512]
    cpath.write_text(json.dumps(config))
    assert vp.main([str(base)]) == 1


def test_sample_selection_spans_every_split_and_class_deterministically(tmp_path):
    base, _ = build_fake_base(tmp_path)
    first = pp.select_sample_ids(str(base), 1)
    again = pp.select_sample_ids(str(base), 1)
    assert first == again
    frozen = list(csv.DictReader(open(base / 'manifests' / 'split_manifest.csv',
                                      encoding='utf-8')))
    groups = {(r['split'], r['label_class']) for r in frozen}
    picked = {(r['split'], r['label_class']) for r in frozen
              if r['image_id'] in first}
    assert picked == groups                   # one per group, incl. unlabelled


def test_run_is_byte_deterministic_across_forced_runs(tmp_path):
    base, ids = build_fake_base(tmp_path)
    run_ids(base, ids, force=True)
    before = {r['image_id']: pp.file_sha256(str(base / 'proc' / r['out_rel_path']))
              for r in read_proc_manifest(base)}
    run_ids(base, ids, force=True)
    after = {r['image_id']: pp.file_sha256(str(base / 'proc' / r['out_rel_path']))
             for r in read_proc_manifest(base)}
    assert before == after
