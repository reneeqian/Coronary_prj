# Data Contract — COCA Coronary Calcium CT Dataset

## 1. Purpose
This document defines the explicit data assumptions, constraints, and guarantees
for using the COCA dataset in a coronary artery calcium (CAC) detection pipeline.

These constraints inform:
- Data requirements (DR-*)
- Validation requirements (VR-*)
- Dataset validators implemented in code

---

## 2. Data Overview
- Image type: 3D CT volumes
- Image format: NIfTI (.nii, .nii.gz)
- Label format: NIfTI (.nii, .nii.gz)
- One label volume per image volume
- Imaging is non-contrast CT

---

## 3. Spatial Assumptions

- Volumes represent 3D anatomical imaging of the thorax/heart
- Input orientation: arbitrary
- Canonical orientation enforced: RAS
- Target voxel spacing: 1.0 × 1.0 × 1.0 mm
- Dimensionality: `(Z, Y, X)` after loading

---

## 4. Intensity Assumptions

- Intensity units: Hounsfield Units (HU)
- Expected HU range (raw): approximately `[-1000, 3000]`
- Preprocessing operations may include:
  - Clipping to `[-1000, 1000]`
  - Per-volume normalization

---

## 5. Label Semantics

- Binary segmentation masks
- Label encoding:
  - `0`: background
  - `1`: calcified coronary plaque (CAC)
- Labels may contain noise and partial volume effects

---

## 6. Known Dataset Limitations

- Variable voxel spacing across scans
- Inconsistent scan coverage of the heart
- Potential label noise and inter-annotator variability
- Mixed gated and non-gated acquisitions

These limitations are explicitly tolerated and handled through validation logic.

---

## 7. Validation Rules (Conceptual)

The data pipeline shall:
- Reject non-3D volumes
- Reject image/label shape mismatches
- Reject non-finite image values
- Reject labels containing invalid class values
- Resample volumes with non-conforming voxel spacing
- Enforce canonical orientation

Concrete enforcement is implemented in dataset validators.

---

## 8. Traceability

This data contract informs:
- Data Requirements (DR-*)
- Validation Requirements (VR-*)
- Dataset validators in `validators.py`

Each validator is expected to trace back to at least one data requirement
derived from this contract.

```
CanonicalCACSlice
├── image              # 2D numpy array (HU)
├── spacing_mm         # (sx, sy)
├── slice_index        # int
├── has_cac            # bool
├── rois               # optional list of polygons
├── source_id          # patient / study id
```