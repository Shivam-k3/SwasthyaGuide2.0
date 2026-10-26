"""RSNA preprocessing: decode DICOM -> deterministic 224x224 float32 [0,1] cache.

Read-only over the raw tree. All outputs go under proc/. The frozen manifests
are inputs only and are never modified here.

Intensity policy (CONFIG_VERSION 1)
-----------------------------------
Source contract (verified on all 30,000 files at design time):
  - transfer syntax 1.2.840.10008.1.2.4.50 (JPEG Baseline, lossy, 8-bit)
  - PhotometricInterpretation MONOCHROME2, SamplesPerPixel 1, unsigned
  - 1024x1024, no RescaleSlope/RescaleIntercept

Transformation, in order:
  1. Validate metadata against the contract above (reject anything else with
     a named error category - no silent fallbacks).
  2. MONOCHROME1 images are inverted (255 - v) so every output shares
     MONOCHROME2 polarity (high value = high attenuation).
  3. Resize to exactly 224x224 with PIL LANCZOS on the uint8 image.
  4. Cast to float32 and divide by 255 -> closed range [0.0, 1.0].

No per-image min-max or percentile normalisation (source is already an
8-bit windowed display image; a per-image affine would destroy cross-image
intensity comparability without justification). No augmentation. The same
deterministic path is used for train, validation and test.

Cache layout
------------
  proc/config.json          run configuration + scope (rewritten per run)
  proc/proc_manifest.csv    one row per image in the run scope
  proc/images/<split>/<image_id>.npy   float32 (224, 224)

Writes are atomic (tmp file + os.replace). Reuse of an existing output
requires: same config_version, same split, same source SHA-256, and a
structurally valid .npy on disk. Anything else is reprocessed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import pydicom
import PIL
from PIL import Image

BASE = r'D:\Datasets\RSNA_Pneumonia'
RAW_SUBDIR, MAN_SUBDIR, PROC_SUBDIR = 'raw', 'manifests', 'proc'

# --- frozen policy constants (also asserted by verify_proc.py) ---
CONFIG_VERSION = 1
TARGET_SIZE = (224, 224)
OUTPUT_DTYPE = 'float32'
SOURCE_BITS = 8
RESIZE_FILTER = 'LANCZOS'
SUPPORTED_TRANSFER_SYNTAXES = {
    '1.2.840.10008.1.2',        # Implicit VR Little Endian (uncompressed)
    '1.2.840.10008.1.2.1',      # Explicit VR Little Endian (uncompressed)
    '1.2.840.10008.1.2.4.50',   # JPEG Baseline (Pillow decoder, installed)
}
UNCOMPRESSED_TRANSFER_SYNTAXES = {
    '1.2.840.10008.1.2',        # only these have a deterministic raw length
    '1.2.840.10008.1.2.1',      # Rows*Cols*(BitsAllocated/8); compressed streams
}                               # are length-validated by the decoder itself
ALLOWED_PHOTOMETRIC = {'MONOCHROME1', 'MONOCHROME2'}

REQUIRED_SPLIT_COLS = {'image_id', 'patient_id', 'study_id', 'target',
                       'label_class', 'bbox_count', 'split'}
REQUIRED_MANIFEST_COLS = {'image_id', 'rel_path'}

# error categories (status='failed' rows always carry one)
CATEGORIES = {
    'FILE_MISSING', 'UNSUPPORTED_TRANSFER_SYNTAX', 'PIXEL_METADATA_ERROR',
    'DECODE_ERROR', 'WRITE_ERROR', 'PREPROCESS_INTERNAL',
}

PROC_MANIFEST_COLS = ['image_id', 'patient_id', 'study_id', 'split', 'target',
                      'label_class', 'bbox_count', 'out_rel_path', 'shape',
                      'dtype', 'src_sha256', 'config_version', 'origin',
                      'status', 'error']


class PreprocessError(Exception):
    """Raised with a stable category so failures can be accounted for."""

    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category
        self.message = message


# ---------------------------------------------------------------- helpers
def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def raw_path(base: str, rel_path: str) -> str:
    return os.path.join(base, RAW_SUBDIR, rel_path.replace('/', os.sep))


def load_inventory(base: str) -> tuple[list[dict], dict[str, str]]:
    """Read the frozen split manifest (+ rel_path lookup) unmodified."""
    man = os.path.join(base, MAN_SUBDIR)
    with open(os.path.join(man, 'split_manifest.csv'), encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    with open(os.path.join(man, 'rsna_manifest.csv'), encoding='utf-8') as f:
        meta_rows = list(csv.DictReader(f))
    missing = REQUIRED_SPLIT_COLS - set(rows[0]) if rows else REQUIRED_SPLIT_COLS
    if missing:
        raise SystemExit(f'split_manifest.csv missing columns: {sorted(missing)}')
    missing = REQUIRED_MANIFEST_COLS - set(meta_rows[0]) if meta_rows else REQUIRED_MANIFEST_COLS
    if missing:
        raise SystemExit(f'rsna_manifest.csv missing columns: {sorted(missing)}')
    rel = {r['image_id']: r['rel_path'] for r in meta_rows}
    unknown = [r['image_id'] for r in rows if r['image_id'] not in rel]
    if unknown:
        raise SystemExit(f'{len(unknown)} split rows lack a rel_path '
                         f'(first: {unknown[0]})')
    return rows, rel


def compute_raw_fingerprint(base: str) -> dict:
    """Cheap raw immutability anchor, same algorithm AND row order as
    raw_fingerprint.json / verify_split.py (manifest row order, not sorted)."""
    rows, rel = load_inventory(base)
    sizes = [os.path.getsize(raw_path(base, rel[r['image_id']])) for r in rows]
    return {'file_count': len(sizes),
            'total_bytes': sum(sizes),
            'sizes_sha256': hashlib.sha256(
                ','.join(map(str, sizes)).encode()).hexdigest()}


def select_sample_ids(base: str, n_per_group: int) -> list[str]:
    """First n image IDs (sorted) per (split, label_class) group."""
    rows, _ = load_inventory(base)
    groups: dict[tuple[str, str], list[str]] = {}
    for r in rows:
        groups.setdefault((r['split'], r['label_class']), []).append(r['image_id'])
    ids: list[str] = []
    for key in sorted(groups):
        ids.extend(sorted(groups[key])[:n_per_group])
    return ids


# ------------------------------------------------------------- the policy
def decode_and_preprocess(dicom_path: str) -> np.ndarray:
    """Decode one DICOM and return a float32 (224, 224) array in [0, 1].

    Raises PreprocessError with a stable category on any violation.
    """
    if not os.path.isfile(dicom_path):
        raise PreprocessError('FILE_MISSING', f'no source file: {dicom_path}')
    try:
        ds = pydicom.dcmread(dicom_path)
    except PreprocessError:
        raise
    except Exception as exc:  # pydicom raises many types on malformed files
        raise PreprocessError('DECODE_ERROR', f'dcmread failed: {type(exc).__name__}: {exc}') from exc

    try:
        ts = str(ds.file_meta.TransferSyntaxUID)
    except Exception:
        raise PreprocessError('PIXEL_METADATA_ERROR', 'file has no TransferSyntaxUID')
    if ts not in SUPPORTED_TRANSFER_SYNTAXES:
        raise PreprocessError(
            'UNSUPPORTED_TRANSFER_SYNTAX',
            f'transfer syntax {ts} not in supported set '
            f'{sorted(SUPPORTED_TRANSFER_SYNTAXES)}')

    photo = str(ds.get('PhotometricInterpretation', ''))
    if photo not in ALLOWED_PHOTOMETRIC:
        raise PreprocessError('PIXEL_METADATA_ERROR',
                              f'photometric {photo!r} not in {sorted(ALLOWED_PHOTOMETRIC)}')
    if int(ds.get('SamplesPerPixel', 1)) != 1:
        raise PreprocessError('PIXEL_METADATA_ERROR',
                              f'SamplesPerPixel={ds.SamplesPerPixel}, expected 1')
    if int(ds.get('PixelRepresentation', 0)) != 0:
        raise PreprocessError('PIXEL_METADATA_ERROR', 'signed pixels (PixelRepresentation=1) are out of policy')
    if int(ds.get('BitsAllocated', 0)) != SOURCE_BITS:
        raise PreprocessError('PIXEL_METADATA_ERROR',
                              f'BitsAllocated={ds.BitsAllocated}, policy is {SOURCE_BITS}-bit sources')
    slope, intercept = ds.get('RescaleSlope', None), ds.get('RescaleIntercept', None)
    if (slope is not None and float(slope) != 1.0) or (intercept is not None and float(intercept) != 0.0):
        raise PreprocessError('PIXEL_METADATA_ERROR',
                              f'non-identity modality rescale (slope={slope}, intercept={intercept}) '
                              'would apply a per-image transform; out of policy')

    rows_n, cols_n = int(ds.Rows), int(ds.Columns)
    if rows_n < 1 or cols_n < 1:
        raise PreprocessError('PIXEL_METADATA_ERROR', f'bad dimensions {rows_n}x{cols_n}')
    expected = rows_n * cols_n * (SOURCE_BITS // 8)
    data = ds.get('PixelData', None)
    if data is None:
        raise PreprocessError('PIXEL_METADATA_ERROR', 'no PixelData element')
    if ts in UNCOMPRESSED_TRANSFER_SYNTAXES and len(data) != expected:
        raise PreprocessError('PIXEL_METADATA_ERROR',
                              f'PixelData length {len(data)} != Rows*Cols ({expected})')

    try:
        arr = ds.pixel_array
    except Exception as exc:
        raise PreprocessError('DECODE_ERROR', f'pixel decoding failed: {type(exc).__name__}: {exc}') from exc
    if arr.dtype != np.uint8 or arr.shape != (rows_n, cols_n):
        raise PreprocessError('PIXEL_METADATA_ERROR',
                              f'decoded array {arr.dtype} {arr.shape} contradicts metadata '
                              f'uint8 ({rows_n}, {cols_n})')

    if photo == 'MONOCHROME1':
        arr = 255 - arr  # normalise polarity to MONOCHROME2

    img = Image.fromarray(arr, mode='L').resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    out = np.asarray(img, dtype=np.float32) / np.float32(255.0)

    if out.shape != TARGET_SIZE or out.dtype != np.float32:
        raise PreprocessError('PREPROCESS_INTERNAL', f'bad output {out.dtype} {out.shape}')
    if not np.isfinite(out).all() or out.min() < 0.0 or out.max() > 1.0:
        raise PreprocessError('PREPROCESS_INTERNAL',
                              f'output range [{out.min()}, {out.max()}] violates [0, 1]')
    return out


def npy_quick_ok(path: str) -> bool:
    """Cheap structural check (header only): right shape and dtype."""
    try:
        a = np.load(path, mmap_mode='r')
        ok = a.shape == TARGET_SIZE and a.dtype == np.dtype(OUTPUT_DTYPE)
        del a
        return ok
    except Exception:
        return False


# ------------------------------------------------------------- orchestration
def _atomic_write_bytes(path: str, payload: bytes) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_npy(path: str, arr: np.ndarray) -> None:
    import io
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    _atomic_write_bytes(path, buf.getvalue())


def _policy_dict() -> dict:
    return {
        'config_version': CONFIG_VERSION,
        'target_size': list(TARGET_SIZE),
        'output_dtype': OUTPUT_DTYPE,
        'source_bits': SOURCE_BITS,
        'resize_filter': RESIZE_FILTER,
        'photometric_polarity': 'MONOCHROME1 inverted to MONOCHROME2',
        'intensity': 'linear v/255 to [0,1]; no per-image normalisation; no augmentation',
        'supported_transfer_syntaxes': sorted(SUPPORTED_TRANSFER_SYNTAXES),
        'allowed_photometric': sorted(ALLOWED_PHOTOMETRIC),
    }


def run(base: str, scope_ids: list[str], scope_mode: str, *,
        force: bool = False, sample_n: int | None = None,
        limit: int | None = None, prune_unlisted: bool = False) -> dict:
    """Process exactly `scope_ids` into <base>/proc. Returns a summary dict."""
    assert scope_mode in ('sample', 'explicit', 'full')
    proc = os.path.join(base, PROC_SUBDIR)
    img_root = os.path.join(proc, 'images')
    os.makedirs(img_root, exist_ok=True)

    rows, rel = load_inventory(base)
    by_id = {r['image_id']: r for r in rows}
    unknown = [i for i in scope_ids if i not in by_id]
    if unknown:
        raise SystemExit(f'scope contains unknown image ids (first: {unknown[0]})')
    if limit is not None:
        scope_ids = sorted(scope_ids)[:limit]

    prev: dict[str, dict] = {}
    prev_path = os.path.join(proc, 'proc_manifest.csv')
    if os.path.exists(prev_path):
        with open(prev_path, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r.get('status') == 'ok':
                    prev[r['image_id']] = r

    out_rows: list[dict] = []
    processed = reused = failed = 0
    failures: list[dict] = []
    t0 = time.time()
    for n, image_id in enumerate(scope_ids, 1):
        src = raw_path(base, rel[image_id])
        inv = by_id[image_id]
        split, target = inv['split'], inv['target']
        rel_out = f'images/{split}/{image_id}.npy'
        out_path = os.path.join(proc, rel_out.replace('/', os.sep))
        row = {'image_id': image_id, 'patient_id': inv['patient_id'],
               'study_id': inv['study_id'], 'split': split, 'target': target,
               'label_class': inv['label_class'], 'bbox_count': inv['bbox_count'],
               'out_rel_path': '', 'shape': f'{TARGET_SIZE[0]}x{TARGET_SIZE[1]}',
               'dtype': OUTPUT_DTYPE, 'src_sha256': '', 'config_version': CONFIG_VERSION,
               'origin': '', 'status': '', 'error': ''}

        sha = file_sha256(src) if os.path.isfile(src) else ''
        prev_row = prev.get(image_id)
        if (not force and prev_row is not None
                and prev_row.get('config_version') == str(CONFIG_VERSION)
                and prev_row.get('split') == split
                and prev_row.get('src_sha256') == sha
                and prev_row.get('out_rel_path') == rel_out
                and npy_quick_ok(out_path)):
            row.update(out_rel_path=rel_out, src_sha256=sha, origin='reused', status='ok')
            reused += 1
        else:
            try:
                if not sha:
                    raise PreprocessError('FILE_MISSING', f'no source file: {src}')
                arr = decode_and_preprocess(src)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                _atomic_write_npy(out_path, arr)
                if not npy_quick_ok(out_path):
                    raise PreprocessError('WRITE_ERROR', 'output failed structural check after write')
                row.update(out_rel_path=rel_out, src_sha256=sha, origin='processed', status='ok')
                processed += 1
            except PreprocessError as exc:
                row.update(status='failed', error=f'{exc.category}: {exc.message}'[:300])
                failed += 1
                failures.append({'image_id': image_id, 'category': exc.category,
                                 'message': exc.message})
            except Exception as exc:  # never leave a row unaccounted
                row.update(status='failed',
                           error=f'PREPROCESS_INTERNAL: {type(exc).__name__}: {exc}'[:300])
                failed += 1
                failures.append({'image_id': image_id, 'category': 'PREPROCESS_INTERNAL',
                                 'message': str(exc)})
        out_rows.append(row)
        if n % 1000 == 0:
            print(f'  {n}/{len(scope_ids)} ({time.time()-t0:.0f}s) '
                  f'processed={processed} reused={reused} failed={failed}', flush=True)

    # manifest + config, both atomic
    tmp = prev_path + '.tmp'
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=PROC_MANIFEST_COLS)
        w.writeheader()
        w.writerows(out_rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, prev_path)

    config = {
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'policy': _policy_dict(),
        'software': {'python': sys.version.split()[0],
                     'pydicom': pydicom.__version__,
                     'numpy': np.__version__,
                     'PIL': PIL.__version__},
        'sha256_split_manifest': file_sha256(os.path.join(base, MAN_SUBDIR, 'split_manifest.csv')),
        'sha256_rsna_manifest': file_sha256(os.path.join(base, MAN_SUBDIR, 'rsna_manifest.csv')),
        'scope': {'mode': scope_mode,
                  'sample_n_per_group': sample_n,
                  'image_ids': None if scope_mode == 'full' else sorted(scope_ids),
                  'required_count': len(scope_ids)},
    }
    _atomic_write_bytes(os.path.join(proc, 'config.json'),
                        json.dumps(config, indent=2).encode('utf-8'))

    if prune_unlisted:
        listed = {r['out_rel_path'] for r in out_rows if r['status'] == 'ok'}
        removed = 0
        for dirpath, _, filenames in os.walk(img_root):
            for fn in filenames:
                if not fn.endswith('.npy'):
                    continue
                p = os.path.join(dirpath, fn)
                if os.path.relpath(p, proc).replace(os.sep, '/') not in listed:
                    os.remove(p)
                    removed += 1
        if removed:
            print(f'pruned {removed} cache files not in the current scope')

    per_split = {}
    for r in out_rows:
        d = per_split.setdefault(r['split'], {'ok': 0, 'failed': 0})
        d[r['status']] += 1
    print(f"scope={len(scope_ids)} processed={processed} reused={reused} "
          f"failed={failed} ({time.time()-t0:.1f}s) per_split={per_split}")
    for f_ in failures:
        print(f"FAILED {f_['image_id']}: {f_['category']}: {f_['message'][:160]}")
    return {'processed': processed, 'reused': reused, 'failed': failed,
            'failures': failures, 'per_split': per_split}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--sample', type=int, metavar='N',
                   help='first N image ids per (split, label_class) group')
    g.add_argument('--image-ids', metavar='ID,ID,...', help='explicit image id list')
    g.add_argument('--full', action='store_true',
                   help='entire frozen inventory (explicit opt-in)')
    p.add_argument('--limit', type=int, help='with --full: only first K ids (staged rollout)')
    p.add_argument('--force', action='store_true', help='reprocess even if cache is reusable')
    p.add_argument('--prune-unlisted', action='store_true',
                   help='after the run, delete proc cache files not in the new manifest')
    args = p.parse_args(argv)

    if args.sample is not None:
        if args.sample < 1:
            p.error('--sample must be >= 1')
        ids = select_sample_ids(BASE, args.sample)
        mode, sample_n = 'sample', args.sample
    elif args.image_ids:
        ids = [s.strip() for s in args.image_ids.split(',') if s.strip()]
        mode, sample_n = 'explicit', None
    else:
        rows, _ = load_inventory(BASE)
        ids = [r['image_id'] for r in rows]
        mode, sample_n = 'full', None

    summary = run(BASE, ids, mode, force=args.force, sample_n=sample_n,
                  limit=args.limit, prune_unlisted=args.prune_unlisted)
    return 1 if summary['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
