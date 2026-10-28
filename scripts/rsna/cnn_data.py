"""Dataset + manifest loading for the RSNA CNN baseline.

Reads the verified preprocessing cache (proc/proc_manifest.csv + .npy files,
see preprocess.py / verify_proc.py). Unlabelled images (target '') are
excluded explicitly and never coerced to the negative class. Patient-level
train/validation/test separation is asserted at load time.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset

VALID_TARGETS = {'', '0', '1'}


class RSNADataset(Dataset):
    """Labelled processed images for one split.

    Each item is (float32 tensor of shape (1, 224, 224) in [0, 1],
    float32 scalar label in {0.0, 1.0}).
    """

    def __init__(self, proc_dir: str, rows: list[dict]):
        self.proc_dir = proc_dir
        self.rows = rows
        missing = [r['image_id'] for r in rows if not os.path.isfile(self._path(r))]
        if missing:
            raise ValueError(
                f'{len(missing)} processed image file(s) missing for this split '
                f'(first: {missing[:5]}). Run scripts/rsna/preprocess.py --full '
                f'and scripts/rsna/verify_proc.py before training.')

    def _path(self, r: dict) -> str:
        return os.path.join(self.proc_dir, r['out_rel_path'].replace('/', os.sep))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        arr = np.load(self._path(r))          # float32 (224, 224) in [0, 1]
        if arr.shape != (224, 224) or arr.dtype != np.float32:
            raise ValueError(f'{r["image_id"]}: cache contract violated '
                             f'({arr.dtype} {arr.shape}); rerun verify_proc.py')
        x = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)
        y = torch.tensor(1.0 if r['target'] == '1' else 0.0)
        return x, y


def load_split_rows(base: str,
                    splits: tuple[str, ...] = ('train', 'validation', 'test')
                    ) -> dict[str, list[dict]]:
    """Load proc_manifest rows per split, excluding unlabelled records.

    Raises with a clear message on invalid targets, failed preprocess rows,
    empty splits, or patient-level leakage between splits.
    """
    proc = os.path.join(base, 'proc')
    path = os.path.join(proc, 'proc_manifest.csv')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'no preprocessing manifest at {path}; run '
            f'scripts/rsna/preprocess.py --full first')

    with open(path, encoding='utf-8') as f:
        all_rows = list(csv.DictReader(f))
    bad = [r['image_id'] for r in all_rows if r['target'] not in VALID_TARGETS]
    if bad:
        raise ValueError(f'invalid target values in proc_manifest (first: '
                         f'{bad[:5]}); refusing to guess labels')
    ok = [r for r in all_rows if r['status'] == 'ok']
    if len(ok) != len(all_rows):
        raise ValueError(f'{len(all_rows) - len(ok)} images have status != ok in '
                         f'proc_manifest; resolve preprocessing failures first '
                         f'(scripts/rsna/verify_proc.py)')

    rows = {s: [r for r in ok if r['split'] == s and r['target'] != '']
            for s in splits}
    excluded = sum(1 for r in ok if r['target'] == '')
    print(f'loaded {sum(len(v) for v in rows.values()):,} labelled images '
          f'({", ".join(f"{s}={len(v)}" for s, v in rows.items())}); excluded '
          f'{excluded} unlabelled images (never treated as negative)')

    for s in splits:
        if not rows[s]:
            raise ValueError(f'split {s!r} has no labelled images')
    pat = {s: {r['patient_id'] for r in v} for s, v in rows.items()}
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            overlap = pat[a] & pat[b]
            if overlap:
                raise RuntimeError(f'patient-level leakage: {len(overlap)} patients '
                                   f'appear in both {a!r} and {b!r} '
                                   f'(first: {sorted(overlap)[:3]})')
    return rows


def make_datasets(base: str,
                  splits: tuple[str, ...] = ('train', 'validation', 'test')
                  ) -> tuple[dict[str, RSNADataset], dict[str, list[dict]]]:
    rows = load_split_rows(base, splits)
    proc = os.path.join(base, 'proc')
    return {s: RSNADataset(proc, v) for s, v in rows.items()}, rows


def label_counts(rows: list[dict]) -> tuple[int, int]:
    """(n_positive, n_negative) over labelled rows."""
    n_pos = sum(1 for r in rows if r['target'] == '1')
    return n_pos, len(rows) - n_pos


def pos_weight_value(rows: list[dict]) -> float:
    """negatives / positives for BCEWithLogitsLoss(pos_weight=...)."""
    n_pos, n_neg = label_counts(rows)
    if n_pos == 0:
        raise ValueError('no positive examples; class weighting undefined')
    return n_neg / n_pos


def stratified_limit(rows: list[dict], n_per_class: int) -> list[dict]:
    """Deterministic subset: first n rows (by image_id) per target class."""
    out = []
    for target in ('0', '1'):
        out.extend(sorted((r for r in rows if r['target'] == target),
                          key=lambda r: r['image_id'])[:n_per_class])
    return sorted(out, key=lambda r: r['image_id'])
