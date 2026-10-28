"""Regression tests for the patient-level split report statistics.

Covers the 2026-10-27 fix: `images_per_patient` in split_report.json is now
PATIENT-weighted (one entry per patient). The generator
(``split_patients.py``) lives with the dataset under ``$RSNA_BASE/scripts``
(default ``D:\\Datasets\\RSNA_Pneumonia``); the synthetic test runs anywhere
the module is present, and the live-report test additionally needs the
manifests. Both skip cleanly when the dataset drive is absent.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path

import pytest

BASE = Path(os.environ.get('RSNA_BASE', r'D:\Datasets\RSNA_Pneumonia'))
SPLITTER = BASE / 'scripts' / 'split_patients.py'

if not SPLITTER.is_file():
    pytest.skip(f'split_patients.py not found under {BASE}', allow_module_level=True)

spec = importlib.util.spec_from_file_location('rsna_split_patients', SPLITTER)
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)


def _row(pid, image, target):
    return {'image_id': image, 'patient_id': pid, 'target': target}


def test_split_stats_is_patient_weighted_not_image_weighted():
    # patients: p1 has 3 images, p2 and p3 have 1 each -> sizes [1, 1, 3]
    sel = ([_row('p1', f'i{ k }', '0') for k in range(3)]
           + [_row('p2', 'i3', '0'), _row('p3', 'i4', '1')])
    patients = {'p1': sel[:3], 'p2': [sel[3]], 'p3': [sel[4]]}
    st = sp.split_stats(sel, patients)
    ipp = st['images_per_patient']
    assert ipp['min'] == 1 and ipp['max'] == 3
    assert ipp['median'] == 1                      # patient-weighted median
    assert ipp['mean'] == round(5 / 3, 3)          # 5 images / 3 patients
    # the pre-fix image-weighted bug would report median 3, mean 2.2
    assert st['n_patients'] == 3 and st['n_images'] == 5
    assert st['n_positive'] == 1 and st['n_negative'] == 4
    assert st['n_unlabelled'] == 0


def test_split_stats_excludes_unlabelled_from_pos_neg_but_counts_them():
    sel = [_row('p1', 'a', '1'), _row('p1', 'b', ''), _row('p2', 'c', '')]
    patients = {'p1': sel[:2], 'p2': [sel[2]]}
    st = sp.split_stats(sel, patients)
    assert st['n_labelled'] == 1 and st['n_positive'] == 1
    assert st['n_negative'] == 0 and st['n_unlabelled'] == 2
    # sizes [1, 2]; the report uses the upper median (sizes[len//2]), like the
    # original implementation and the live split_report.json
    assert st['images_per_patient'] == {'min': 1, 'max': 2, 'median': 2,
                                        'mean': 1.5}


def _patient_weighted(rows, split):
    from collections import defaultdict
    pat = defaultdict(list)
    for r in rows:
        pat[r['patient_id']].append(r)
    sel = [r for r in rows if r['split'] == split]
    sizes = sorted(len(pat[pid]) for pid in {r['patient_id'] for r in sel})
    return {'min': sizes[0], 'max': sizes[-1],
            'median': sizes[len(sizes) // 2],
            'mean': round(sum(sizes) / len(sizes), 3)}


def test_live_split_report_matches_patient_weighted_recompute():
    man = BASE / 'manifests'
    report_path = man / 'split_report.json'
    split_path = man / 'split_manifest.csv'
    if not (report_path.is_file() and split_path.is_file()):
        pytest.skip('dataset manifests not present')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    with open(split_path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for split in ('train', 'validation', 'test'):
        expected = _patient_weighted(rows, split)
        actual = report['partitions'][split]['images_per_patient']
        assert actual == expected, f'{split}: report {actual} != recompute {expected}'