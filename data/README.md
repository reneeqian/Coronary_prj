# Coronary CT Data Contract

## 1. Data Overview
This project uses cardiac CT angiography (CTA) volumes for coronary artery segmentation.

- Image format: NIfTI (.nii.gz)
- Label format: NIfTI (.nii.gz)
- One label per image volume

## 2. Directory Structure

raw/
  images/    # Raw CT volumes
  labels/    # Raw segmentation masks

processed/
  images/    # Preprocessed volumes
  labels/    # Preprocessed masks

## 3. Spatial Assumptions

- Input orientation: arbitrary
- Pipeline enforces: RAS orientation
- Target voxel spacing: 1.0 x 1.0 x 1.0 mm
- Dimensionality: 3D volumes (Z, Y, X)

## 4. Intensity Handling

- Intensity units: Hounsfield Units (HU)
- Clip range: [-1000, 1000]
- Normalization: z-score per volume

## 5. Label Semantics

- 0: background
- 1: coronary arteries

## 6. Dataset Split

- Patient-level split
- Train / validation / test: 70 / 15 / 15
- Fixed random seed

## 7. Validation Rules

The data pipeline will:
- Reject samples with missing labels
- Resample volumes with incorrect spacing
- Enforce consistent orientation
