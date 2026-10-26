"""Standalone verification of the processed-image cache (read-only).

Run:  python verify_proc.py [base_dir]      (default D:\\Datasets\\RSNA_Pneumonia)

Checks (any FAIL -> exit 1; nothing is repaired or regenerated):
  1. every required image id (per config scope) appears exactly once
  2. each output's split matches the frozen split manifest
  3. no unexpected ids; cache files on disk == manifest ok-rows exactly
  4. output files exist and load
  5. every image has shape (224, 224)
  6. dtype float32 and pixel range within [0, 1]
  7. no NaN / Inf; image not constant
  8. no duplicate image ids
  9. raw tree unchanged (fingerprint) and frozen manifests unchanged (SHA-256)
 10. all failures carry an explicit error category
 11. unlabelled images remain unlabelled (target/label match frozen)
 12. per-row config matches config.json and the code's policy constants
 13. no partial outputs (*.tmp) under proc\
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocess as pp  # noqa: E402  (shares the frozen policy constants)

DEFAULT_BASE = r'D:\Datasets\RSNA_Pneumonia'


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(base: str) -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = '') -> None:
        print(('PASS  ' if ok else 'FAIL  ') + name + (f'  {detail}' if detail else ''))
        if not ok:
            failures.append(name)

    proc = os.path.join(base, pp.PROC_SUBDIR)
    man = os.path.join(base, pp.MAN_SUBDIR)

    config_path = os.path.join(proc, 'config.json')
    manifest_path = os.path.join(proc, 'proc_manifest.csv')
    check('files: config.json and proc_manifest.csv exist',
          os.path.isfile(config_path) and os.path.isfile(manifest_path))
    if failures:
        print('\nFAILED CHECKS:', failures)
        return 1
    config = json.load(open(config_path, encoding='utf-8'))
    with open(manifest_path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    inv_rows, rel = pp.load_inventory(base)
    inv = {r['image_id']: r for r in inv_rows}

    # 12. configuration consistency (before using config to define the scope)
    policy = config.get('policy', {})
    expect = pp._policy_dict()
    check('12. config.json policy matches code constants', policy == expect)
    check('12b. row config_version uniform and == config.json',
          {r['config_version'] for r in rows} == {str(pp.CONFIG_VERSION)}
          and str(config['policy']['config_version']) == str(pp.CONFIG_VERSION))

    # scope: full -> entire frozen inventory, else the recorded id list
    scope = config.get('scope', {})
    if scope.get('mode') == 'full':
        required = set(inv)
    else:
        required = set(scope.get('image_ids') or [])
        check('12c. recorded scope ids all exist in frozen inventory',
              required <= set(inv), f'({len(required - set(inv))} unknown)')
    check('12d. required_count matches actual required set',
          scope.get('required_count') == len(required),
          f"recorded={scope.get('required_count')} actual={len(required)}")

    ids = [r['image_id'] for r in rows]
    ok_rows = [r for r in rows if r['status'] == 'ok']
    failed_rows = [r for r in rows if r['status'] == 'failed']

    # 8. duplicates
    check('8. no duplicate image ids in proc manifest', len(ids) == len(set(ids)))

    # 1. required coverage, exactly once
    from collections import Counter
    cnt = Counter(ids)
    multi = [i for i, c in cnt.items() if c > 1]
    check('1. every required id appears exactly once',
          not multi and required <= set(ids) and set(ids) <= (set(inv)),
          f'(missing={len(required - set(ids))}, duplicated={len(multi)})')

    # 3. no unexpected ids
    check('3. no unexpected ids vs frozen inventory', set(ids) <= set(inv))

    # 2. split matches frozen; 11. target/label preserved verbatim
    bad_split = [r['image_id'] for r in rows if r['split'] != inv[r['image_id']]['split']]
    check('2. every output split matches the frozen manifest', not bad_split,
          f'({len(bad_split)} mismatched)')
    bad_label = [r['image_id'] for r in rows
                 if r['target'] != inv[r['image_id']]['target']
                 or r['label_class'] != inv[r['image_id']]['label_class']]
    check('11. targets and label classes preserved (unlabelled stay unlabelled)',
          not bad_label, f'({len(bad_label)} altered)')

    # 13. no partial outputs
    tmps = glob.glob(os.path.join(proc, 'images', '**', '*.tmp'), recursive=True)
    check('13. no partial (*.tmp) outputs under proc', not tmps, f'({len(tmps)} found)')

    # 3b. cache files on disk == ok rows exactly
    disk = set()
    for p in glob.glob(os.path.join(proc, 'images', '**', '*.npy'), recursive=True):
        disk.add(os.path.relpath(p, proc).replace(os.sep, '/'))
    listed = {r['out_rel_path'] for r in ok_rows}
    check('3b. cache files on disk exactly match successful manifest rows',
          disk == listed, f'(extra={len(disk - listed)}, missing={len(listed - disk)})')

    # 4-7 + per-file src hash + path layout
    bad_exist: list[str] = []
    bad_shape: list[str] = []
    bad_dtype: list[str] = []
    bad_range: list[str] = []
    bad_nan: list[str] = []
    bad_const: list[str] = []
    bad_sha: list[str] = []
    bad_path: list[str] = []
    for r in ok_rows:
        out_path = os.path.join(proc, r['out_rel_path'].replace('/', os.sep))
        if not os.path.isfile(out_path):
            bad_exist.append(r['image_id'])
            continue
        if (os.path.basename(out_path) != r['image_id'] + '.npy'
                or r['out_rel_path'] != f"images/{r['split']}/{r['image_id']}.npy"):
            bad_path.append(r['image_id'])
        try:
            a = np.load(out_path, allow_pickle=False)
        except Exception:
            bad_exist.append(r['image_id'])
            continue
        if a.shape != pp.TARGET_SIZE:
            bad_shape.append(r['image_id'])
        if a.dtype != np.dtype(pp.OUTPUT_DTYPE):
            bad_dtype.append(r['image_id'])
        if a.size and (float(a.min()) < 0.0 or float(a.max()) > 1.0):
            bad_range.append(r['image_id'])
        if not np.isfinite(a).all():
            bad_nan.append(r['image_id'])
        elif np.unique(a).size < 2:
            bad_const.append(r['image_id'])
        src = pp.raw_path(base, rel[r['image_id']])
        if r['src_sha256'] != file_sha256(src):
            bad_sha.append(r['image_id'])
        del a

    check('4. every ok-row output exists and loads', not bad_exist, f'({len(bad_exist)} bad)')
    check('4b. output paths follow images/<split>/<image_id>.npy', not bad_path,
          f'({len(bad_path)} bad)')
    check('5. every image has shape (224, 224)', not bad_shape, f'({len(bad_shape)} bad)')
    check('6. dtype is float32', not bad_dtype, f'({len(bad_dtype)} bad)')
    check('6b. pixel values within [0, 1]', not bad_range, f'({len(bad_range)} bad)')
    check('7. no NaN or infinite pixel values', not bad_nan, f'({len(bad_nan)} bad)')
    check('7b. no constant images', not bad_const, f'({len(bad_const)} bad)')
    check('12e. recorded source SHA-256 matches current raw file', not bad_sha,
          f'({len(bad_sha)} stale)')

    # 10. failures accounted: failed row -> non-empty 'CATEGORY: ...' error
    unaccounted = [r['image_id'] for r in failed_rows
                   if not r['error'] or r['error'].split(':', 1)[0] not in pp.CATEGORIES]
    check('10. every failure carries a valid error category', not unaccounted,
          f'({len(unaccounted)} unaccounted)')
    check('10b. zero failed images (required for a complete cache)',
          not failed_rows, f'({len(failed_rows)} failed)')
    for r in failed_rows:
        print(f"       failed: {r['image_id']}  {r['error'][:120]}")

    # 9. raw + frozen manifests unchanged
    fp_path = os.path.join(man, 'raw_fingerprint.json')
    fp_ok = False
    detail = ''
    if os.path.isfile(fp_path):
        recorded = json.load(open(fp_path, encoding='utf-8'))
        try:
            current = pp.compute_raw_fingerprint(base)
            # recorded file also carries anchor_time_utc; compare content keys
            fp_ok = all(recorded.get(k) == v for k, v in current.items())
            detail = f'({current["file_count"]} files / {current["total_bytes"]:,} bytes)'
        except OSError as exc:
            detail = f'raw tree unreadable: {exc}'
    check('9. raw DICOM tree unchanged (fingerprint)', fp_ok, detail)
    check('9b. frozen split manifest unchanged since the run',
          config.get('sha256_split_manifest') ==
          file_sha256(os.path.join(man, 'split_manifest.csv')))
    check('9c. frozen rsna manifest unchanged since the run',
          config.get('sha256_rsna_manifest') ==
          file_sha256(os.path.join(man, 'rsna_manifest.csv')))

    ok_count = len(ok_rows)
    print(f'\nproc manifest: {len(rows)} rows ({ok_count} ok, {len(failed_rows)} failed); '
          f'scope required: {len(required)}')
    if failures:
        print('FAILED CHECKS:', failures)
        return 1
    print('ALL CHECKS PASSED')
    return 0


def main(argv: list[str] | None = None) -> int:
    base = argv[0] if argv else DEFAULT_BASE
    if not os.path.isdir(os.path.join(base, pp.MAN_SUBDIR)):
        print(f'not a dataset base dir: {base}')
        return 2
    return verify(base)


if __name__ == '__main__':
    sys.exit(main())
