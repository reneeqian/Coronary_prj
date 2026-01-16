# Dataset Overview — COCA (Coronary Calcium CT)

## Dataset Name
COCA — Coronary Calcium CT Dataset

## Source
- Provider: Stanford AIMI
- Dataset page: https://stanfordaimi.azurewebsites.net/datasets/e8ca74dc-8dd4-4340-815a-60b41f6cb2aa
- Access method: Azure Blob Storage (time-limited SAS URL)
- Download tool: AzCopy v10
- Download date: 2026-01-16

## Imaging Modality
- Non-contrast cardiac CT
- Includes gated cardiac CT and non-gated chest CT

## Labels
- Binary coronary artery calcium (CAC) masks
- Label values:
  - `0`: background
  - `1`: calcified coronary plaque

## Local Storage Layout
```
data/
raw/
coca/
images/ # Raw CT volumes (.nii / .nii.gz)
labels/ # Raw CAC masks (.nii / .nii.gz)

processed/
images/ # Preprocessed volumes
labels/ # Preprocessed masks
```

## Intended Use
This dataset is used for **research and educational purposes** to demonstrate:
- Medical imaging data ingestion
- Explicit data contracts
- Validation and traceability in medical AI pipelines

This project is **not intended for clinical use**.

## License
Dataset usage is governed by the Stanford AIMI Research Use Agreement:  
https://stanfordaimi.azurewebsites.net/datasets/e8ca74dc-8dd4-4340-815a-60b41f6cb2aa
