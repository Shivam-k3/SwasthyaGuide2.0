# PNEUMONIA_DATASET_PLAN.md

## Phase
Dataset acquisition + read-only validation + **data engineering/manifest + patient-level
split (all completed 2026-10-06)**. No training, no integration, no preprocessing, no
model artifacts. Next phase (not started): CNN development.

## Dataset source (official)
- RSNA Pneumonia Detection Challenge (2018) official page:
  https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018
- Kaggle API is **not** used (returns HTTP 403). No mirrors, no unofficial copies,
  no unrelated NIH data.

## Resources downloaded (official RSNA S3)
| Resource | Location on D: | Size (bytes) | SHA-256 |
|---|---|---|---|
| Image archive `pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip` | `raw\pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip` | 3,978,753,654 (3.71 GiB) | `96b97d81eb042c6513196e01079a060a980d5e62502c3266527dd9c0b2a63b50` |
| Annotations `pneumonia-challenge-annotations-adjudicated-kaggle_2018.json` | `raw\annotations\` | 52,003,071 (49.6 MiB) | `f8f5c54bae158fc0e2f019af979471803151a5922463ff3a378d7734e01b18fb` |
| Mapping `pneumonia-challenge-dataset-mappings_2018.json` | `raw\annotations\` | 17,587,143 (16.8 MiB) | `803ce79e3bc9c66d3631738e91e62e1175730e98ad1415e8dc4d6292ba10bf27` |

URLs:
- https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip
- https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-annotations-adjudicated-kaggle_2018.json
- https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-mappings_2018.json

The mapping file **was** required: the DICOM `PatientID` is a per-image pseudonymous
UUID, not a patient key. Patient-level identity is only recoverable through the
mapping (`img_id` prefix => NIH patient).

Verified: archive SHA-256 matches download; `zipfile.testzip()` = all CRCs valid,
30,000 entries; both JSON files parse as JSON.

## Layout (immutable raw source-of-truth)
- `raw\pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip` (kept)
- `raw\images\mdai_rsna_project_x9N20BZa_images_2018-07-20-153330\**\*.dcm` (30,000, extracted)
- `raw\annotations\` (2 JSON files)
- Filenames are `{StudyUID}/{SeriesUID}/{SOPUID}.dcm`. Do NOT rename, convert,
  resize, strip metadata, or copy. Everything below this line of the filesystem is
  derived output and must be built elsewhere (e.g. `D:\Datasets\RSNA_Pneumonia\proc\`).

## Image counts
- Total DICOMs on disk: 30,000 (unique SOPInstanceUID = 30,000)
- Unique studies (StudyInstanceUID): 30,000 (one image per study)
- Unique DICOM PatientID (pseudonym UUID): 30,000 (one per image — Kaggle semantics)
- Unique real patients (NIH `img_id` numeric prefix via mapping): **12,274**

## Annotations (authoritative = `Calculated` label group)
The annotation file is an MD.ai export. Label groups: Default (empty), QA (3
studies), Team 1 (30,000), Team 2 (5,205), Team 2a (4,577), Adjudication (1,380),
**Calculated (29,684 studies — final labels)**. Raw annotator boxes live in
Team 1/2/2a with High/Med/Low probability labels; `Calculated` is the resolved gold
standard.

- Positive class **"Lung Opacity"** (script name `Lung Opacity`): bbox annotation,
  present on **7,106 images** (11,390 bbox records)
- Negative classes (global, no box): **"No Lung Opacity / Not Normal"** on 12,788
  images; **"Normal"** on 9,790 images -> 22,578 non-positive images
- Classes are mutually exclusive per image (0 images carry both a box and a global
  class; verified).
- **316 studies have no `Calculated` label**: 106 are explicitly flagged
  `Exclude` by annotation teams; 210 unannotated/ambiguous. These 316 must be
  treated as unavailable (do not autolabel).
- Total annotation records in `Calculated`: 33,968.

### Label semantics (do NOT assume 0=normal/1=pneumonia)
- Positive (Kaggle `Target = 1`): image has >=1 `Lung Opacity` bounding-box in the
  `Calculated` group. Bounding-box margin is the region of lung opacity — not the
  whole lung, not a segmentation.
- Negative (`Target = 0`): `Normal` or `No Lung Opacity / Not Normal`. "No Lung
  Opacity / Not Normal" is **not** normal — it means present but no opacity
  (e.g. cardiomegaly, other pathology without opacity). Both are non-pneumonia.
- Multiple boxes per image: allowed. Distribution across the 7,106 positive images:
  {1: 3030, 2: 3892, 3: 160, 4: 24}; per-positive-image min/median/mean/max =
  1 / 2 / 1.603 / 4. Total boxes = 11,390.
- Images without pneumonia are represented by a global class label (no box).
- `stage_2_train_labels.csv` / `stage_2_detailed_class_info.csv` /
  `stage_2_sample_submission.csv` are Kaggle-side artifacts and are **not** on the
  official page. They are not downloaded. Target/class/bbox are derivable from the
  `Calculated` group; do that in the D4 dataset-engineering step, not now.

## Patient-level leakage findings (read-only)
- Real patient key = NIH `img_id` prefix, e.g. `img_id` `00012773_000.png` -> patient
  `00012773`. Patients: 12,274.
- Images per patient: median 1, mean 2.444, max 75. Patients with >1 image: 4,810
  (39.2%). Single-image patients: 7,464.
- Images/patient distribution head: {1:7464, 2:1970, 3:987, 4:524, 5:299, 6:198,
  7:171, 8:107, 9:87, 10:76, ... 75:2}.
- Positive cases belong to 2,631 distinct patients.
- Same patient recurs across multiple images/studies -> all future splits MUST be
  patient-level (disjoint by NIH image-patient prefix), never image-level.
- Do NOT key splits on DICOM PatientID (per-image UUID) or StudyInstanceUID.

## DICOM metadata findings (read-only, pydicom 3.0.2)
- Transfer syntax: 100% `1.2.840.10008.1.2.4.50` (JPEG Baseline, lossy)
- Lossy Image Compression Method: ISO_10918_1
- Modality: CR (100%); Body Part: CHEST; Photometric: MONOCHROME2, 8-bit (Bits Allocated 8)
- Dimensions: 1024 x 1024 for all 30,000; Pixel Spacing 0.143 x 0.143 mm (sample)
- Projection: PA 16,248 / AP 13,752 (SeriesDescription `view: PA`/`view: AP`)
- Sex: M 17,078 / F 12,922. PatientAge present as string, 98 distinct values
  (e.g. "58", "56"); no birth date.
- StudyDate `19010101`, StudyTime `000000.00` (scrubbed); AccessionNumber,
  ReferringPhysicianName, StudyID, PatientBirthDate all empty; Manufacturer /
  InstitutionName absent.
- PatientName == PatientID == pseudonymous UUID per image (e.g.
  `14615182-db79-42e7-9b05-3d65cea33d18`).
- No burned-in PHI appears to remain: no values in identity/history fields, PHI
  scrubbed to dummy dates; official description confirms burned-in PHI removed.

## Storage
| Item | Value |
|---|---|
| D: total capacity | 160.58 GB |
| D: free before download | 159.57 GB |
| Downloaded bytes (3 files) | 4,048,343,868 (3.77 GiB) |
| Extracted images | 3,992,178,864 (3.72 GiB), 30,000 files |
| D: free after download + extraction | ~151.93 GB |
| Safety floor (>60 GB) | satisfied (no STOP condition triggered) |

## Licensing / attribution
- RSNA open data, non-commercial academic/research/education use; attribution
  required (see RSNA page terms PDFs). Citation: Shih G, et al. *Augmenting the
  National Institutes of Health Chest Radiograph Dataset with Expert Annotations of
  Possible Pneumonia*. Radiology: Artificial Intelligence 2019.
  https://pubs.rsna.org/doi/10.1148/ryai.2019180041
- RSNA image/annotation source page (above) must be cited in any publication.

## Known limitations
- 316/30,000 images lack gold labels (106 excluded, 210 unannotated).
- One image per StudyInstanceUID; DICOM PatientID is per-image, not per-person.
- JPEG-lossy 1024x1024 8-bit source; no kinfree PNG originals.
- Target semantics differ from a benign/malignant or normal/abnormal task: "No Lung
  Opacity / Not Normal" encodes non-normal non-opacity conditions.
- Kaggle Stage-2 CSV artifacts are not reproduced byte-for-byte; labels derived from
  the `Calculated` group may differ trivially from Kaggle's published CSVs.

## Safety
- Future image-upload security requirements:
  `FUTURE_MEDICAL_IMAGE_SECURITY_REQUIREMENTS.md` (repo).
- Image data never enters the Git repository. Pipeline design only (future):
  DICOM -> safe decode -> patient-level split -> CNN (DenseNet121/EfficientNet-B0/
  ResNet50, 224x224/320x320) -> Grad-CAM -> external eval on Kermany (domain shift).
- Kermany (pediatric) vs RSNA — treat as domain shift; research prototype only.

---

## D4 — Data engineering: manifest + patient-level split (2026-10-06)

All derived artifacts live **outside** the repo on D:, under `D:\Datasets\RSNA_Pneumonia\`.
The `raw\` tree is untouched (read-only). Scripts:
`D:\Datasets\RSNA_Pneumonia\scripts\{build_manifest.py, split_patients.py, verify_split.py}`.

### Manifest design
- `manifests\rsna_manifest.csv` — 30,000 rows, one per image. Columns: `image_id`
  (SOPInstanceUID), `series_id`, `study_id`, `dicom_patient_id` (per-image pseudo-UUID),
  `patient_id` (real patient = NIH `img_id` numeric prefix — **split key**),
  `nih_img_id`, `subset_img_id`, `rel_path` (relative to `raw\`, no absolute user paths),
  `file_size`, `rows`, `columns`, `sex`, `age`, `view_position`, `series_desc`, `modality`,
  `transfer_syntax`, `label_class`, `target`, `annotated`, `bbox_count`.
  No image pixel data, no PHI (only pseudonymous UUIDs + anonymized NIH ids).
- `manifests\bboxes.csv` — 11,390 bounding boxes (authoritative `Calculated` group):
  `image_id, study_id, index, x, y, width, height`.
- `manifests\provenance.json` — sources, URLs, acquisition date, archive/annotation/mapping
  SHA-256, label semantics, split seed, split fractions, software versions.
- `manifests\raw_fingerprint.json` — raw immutability anchor (file count 30,000,
  total bytes 3,992,178,864, SHA-256 over per-file sizes in manifest path order).
  Intentionally does **not** hash all 30k DICOMs (unnecessary cost at this stage).
- `manifests\split_manifest.csv` — `image_id, patient_id, dicom_patient_id, study_id,
  target, label_class, bbox_count, split`. Deterministic (see below).

### Label semantics (as implemented)
- `target = 1` (positive) ⇔ official `Calculated` group label **"Lung Opacity"** and the
  image carries ≥1 bounding box. This means "**evidence of pneumonia present**" — not
  "normal vs abnormal".
- `target = 0` (negative/non-positive) ⇔ **"Normal"** OR **"No Lung Opacity / Not Normal"**
  (the latter is abnormal but without lung opacity — both are non-pneumonia).
- Unannotated images (`annotated = 0`, 316 total) have empty `target`; never autolabelled.
- Classes are mutually exclusive per image (verified: 0 violations).

### Patient-level split methodology
- Unit = real patient (`patient_id`, NIH `img_id` prefix), NOT image or DICOM PatientID.
- Stratification at patient level by `(patient_class, images-per-patient bucket)`, where
  `patient_class ∈ {pos, neg, none}` (any positive image vs labelled-but-no-positive vs
  no label). Buckets: {1, 2, 3-5, 6+}.
- Deterministic: patients sorted, shuffled with `random.Random(42)` (fixed seed `42`),
  70/15/15 → train/validation/test with floors; remainder to train.
- Every image of a patient is assigned that patient's split (patient-disjoint by construction).
- Reproducibility proven: re-running the splitter produces a byte-identical
  `split_manifest.csv` (SHA-256 `5431ac74aea139183fc305eae63c86fd82b3bb3de0c02396a433438ecbf11032`).

### Split sizes & class distribution (patient-weighted split)
| Split | Patients | Images | Labelled | Positive | Negative | Pos % (labelled) | Unlabelled |
|---|---|---|---|---|---|---|---|
| train | 8,596 (70.0%) | 21,147 | 20,918 | 5,112 | 15,806 | 24.44% | 229 |
| validation | 1,839 (15.0%) | 4,332 | 4,297 | 974 | 3,323 | 22.67% | 35 |
| test | 1,839 (15.0%) | 4,521 | 4,469 | 1,020 | 3,449 | 22.82% | 52 |
| total | 12,274 | 30,000 | 29,684 | 7,106 | 22,578 | 23.94% | 316 |

Image-weighted splits are 70.5 / 14.4 / 15.1% (patient-weighted exact 70/15/15).
Images/patient: train 1–75 (median 1, mean 2.46), validation 1–45 (median 1, mean 2.36),
test 1–37 (median 1, mean 2.46). Not rebalanced/oversampled (per instructions).

### Leakage audit (verified)
`train ∩ validation = ∅`, `train ∩ test = ∅`, `validation ∩ test = ∅` for patient IDs AND
image IDs. 0 duplicate image ids; every image in exactly one split; every image has one
patient id. No path-based leakage (secure: splits defined by patients, not paths).

### Verification (standalone `verify_split.py`, exit 0)
1. every manifest image exists on disk ✓ (0 missing)
2. every image has exactly one split ✓
3. every image has exactly one valid patient id ✓
4. no patient crosses splits ✓ (0)
5. no image id crosses splits ✓
6. target values valid and consistent with `label_class` ✓
7. split counts match `split_report.json` ✓
8. raw tree unchanged (count / total bytes / sizes-hash) ✓
   (`bbox_count` ↔ `bboxes.csv` consistency also ✓)

### Provenance / environment
Seed `42`; fractions 70/15/15; Python 3.13.2, pandas 2.3.2, numpy 2.3.1, pydicom 3.0.2
(no new packages installed for engineering). Source: official RSNA page + RSNA S3 URLs;
acquisition 2026-10-06; archive SHA-256 `96b97d81…a63b50`; annotations `f8f5c54b…18fb`;
mapping `803ce79e…bf27` (full values in `provenance.json` on D:).

### Limitations (engineering-specific)
- Patient key (`patient_id`) relies entirely on the official mapping's `img_id`; if the
  mapping is ever regenerated by RSNA, identity joins must be re-verified.
- 316/30,000 images carry no authoritative label and are carried through splits as
  unlabelled (their patients still count toward split integrity).
- Stratified on coarse patient class/bucket only; fine-grained rebalancing is explicitly
  deferred to the modeling phase.