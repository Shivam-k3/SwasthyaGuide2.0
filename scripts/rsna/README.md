# RSNA Pneumonia — preprocessing pipeline

Repository copies of the dataset-side preprocessing code for the SWASTHYAGUIDE 2.0
pneumonia module (methodology and dataset provenance:
`docs/PNEUMONIA_DATASET_PLAN.md`). The RSNA dataset itself is **not** in the
repository: it lives outside the repo, by default at `D:\Datasets\RSNA_Pneumonia`,
overridable with the `RSNA_BASE` environment variable. The raw DICOM tree and the
frozen manifests are read-only inputs; all outputs go under `<RSNA_BASE>/proc/`.

| File | Purpose |
|---|---|
| `preprocess.py` | Decode DICOM → deterministic 224×224 float32 [0,1] `.npy` cache with atomic writes, safe resumption, per-run `proc_manifest.csv` and `config.json` |
| `verify_proc.py` | Standalone read-only verifier for the cache (24 checks); exits 1 on any failure, repairs nothing |

## Requirements

The root `requirements.txt` is intentionally untouched (its pinning is asserted by
`src/verify_stage3.py`). The preprocessing scripts additionally need:

```bash
pip install pydicom==3.0.2 pillow
```

(`numpy` is already pinned in the root `requirements.txt`.)

## Usage

```bash
python scripts/rsna/preprocess.py --sample 3           # N ids per (split, label_class)
python scripts/rsna/preprocess.py --image-ids ID,ID    # explicit ids
python scripts/rsna/preprocess.py --full               # entire frozen inventory (explicit opt-in)
python scripts/rsna/preprocess.py --full --limit 1000  # staged rollout
python scripts/rsna/verify_proc.py                     # verify the cache
```

## Intensity policy (config version 1)

- Source contract: JPEG Baseline (`1.2.840.10008.1.2.4.50`), 8-bit unsigned,
  MONOCHROME2, 1024×1024, identity rescale. Any violation fails with a named
  error category — no silent fallbacks, no blank substitutes.
- MONOCHROME1 inputs are inverted so every output shares MONOCHROME2 polarity.
- LANCZOS resize to exactly 224×224 on the uint8 image, then float32 / 255 →
  closed range [0, 1].
- No per-image min-max normalisation (the source is already an 8-bit windowed
  display image; a per-image affine would destroy cross-image intensity
  comparability). No augmentation. Train/validation/test share one
  deterministic path.

Cache reuse requires a matching config version, split, source SHA-256 and a
structurally valid `.npy`; anything else is reprocessed. Unlabelled records
keep their empty target and are never treated as negatives.

Tests (synthetic fixtures only; the dataset is not required):
`python -m pytest tests/test_preprocess.py -q`
