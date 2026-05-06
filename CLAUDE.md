# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Engineering demonstration of a coronary artery calcium (CAC) detection SaMD (Software as a Medical Device) built on two shared libraries: `medical_image_ai_toolkit` and `regulatory_tools`. The project is **not a clinical product** — it demonstrates deterministic ingestion, task encapsulation, traceability, and regulatory-style verification.

## Setup

```bash
pip install -e ../regulatory_tools
pip install -e ../medical_image_ai_toolkit
pip install -e .
```

## Commands

**Run all unit + synthetic tests (no dataset required):**
```bash
python -m pytest
```

**Run a single test:**
```bash
python -m pytest tests/test_coca_ingestor_synthetic.py::TestClassName::test_name
```

**Full suite with traceability matrix and forge health report (requires grade B or above):**
```bash
python runtests.py
```

**Linting:**
```bash
ruff check src/ tests/
```

**Smoke tests** (require completed training artifacts at `artifacts/training_runs/model.pt` + `partitions.json`):
```bash
python scripts/smoketesttraining.py
python scripts/smoketestmodeltesting.py
python scripts/smoketesthyperparametertuning.py
python scripts/smoketestnongatedtraining.py
```

**Status report** (prints latest training/tuning/model-testing metrics):
```bash
python scripts/status_report.py
```

## Architecture

The project consists of three layers:

### 1. Ingestors (`src/Coronary_prj/ingestors/`)

Convert raw DICOM data from the COCA dataset into `PatientSample` objects (defined in `medical_image_ai_toolkit`).

- `BaseIngestor` — abstract interface: `list_patient_ids()`, `load_patient_sample()`, `get_sample()`
- `COCAGatedIngestor` — reads gated CT DICOM + plist XML annotations. Slices sorted by `ImagePositionPatient[2]`. Annotations parsed from `calcium_xml/{patient_id}.xml` (Apple plist format with 1-based `ImageIndex`).
- `COCANongatedIngestor` — reads non-gated CT DICOM + `scores.xlsx` (openpyxl). Score keys use `"{n}A"` format (e.g., `"1A"`).

**Contract:** all public API methods normalize every failure to `DatasetStructureError`. No raw `FileNotFoundError` or `RuntimeError` escapes the boundary (requirement RSK-003).

### 2. Task Definitions (`src/Coronary_prj/task_definitions/`)

Implement `TrainingTaskDefinition` from `medical_image_ai_toolkit`. Each task converts a `PatientSample` into training samples via `generate_training_samples()` (a generator) and defines `compute_loss()`.

- `CoronaryCalciumTask` — segmentation task. Each CT slice → `(1,1,H,W)` image + `(1,1,H,W)` binary mask. CT preprocessed with cardiac HU window: clip `[-160, 240]`, normalize `(hu - 40) / 200`. Loss = 0.5 × BCE + 0.5 × Dice.
- `NongatedCalciumScoreTask` — regression task. Each slice gets the same patient-level target: `[log1p(lca), log1p(lad), log1p(lcx), log1p(rca)]` shape `(4,)`. Same HU preprocessing. Loss = MSE. At inference, apply `torch.expm1()` to recover Agatston units.

Both tasks use the same cardiac HU window (WL=40, WW=400) and broadcast the patient-level label to every slice (design decision: model learns from any representative slice, not a privileged one — TSK-006).

### 3. Models (`src/Coronary_prj/models/`)

- `UNet2D` — 2D U-Net for segmentation. Input `(B,1,H,W)`, output raw logits `(B,1,H,W)`. `base_channels=32`, `depth=4` (~1.9M params). Bilinear upsampling avoids checkerboard artifacts.
- `CalciumScoreRegressor` — lightweight CNN regression. Input `(B,1,H,W)`, output `(B,4)` log1p-scaled Agatston scores. `base_channels=16` (~47K params), fast enough for CPU smoke tests.

### Shared Dependencies (external packages)

- `medical_image_ai_toolkit` — provides `PatientSample`, `AnnotationBundle`, `VectorROI`, `TrainingTaskDefinition`, `MedicalImageDataSource`, `TrainingPipeline`, `TrainingConfig`, `DeterministicHoldoutSplit`
- `regulatory_tools` — provides `EvidenceReport`, `YamlRequirementProvider`, `run_tests_and_trace`

### Performance Thresholds (`src/Coronary_prj/thresholds.py`)

- `SEGMENTATION_MIN_DICE = 0.50` (MOD-005)
- `REGRESSION_MAX_MAE_AU = 100.0` Agatston units (MOD-006)

## Dataset

**Gated CT (segmentation):** `data/raw/coca/cocacoronarycalciumandchestcts-2/Gated_release_final/`
- Structure: `patient/{patient_id}/{series_dir}/*.dcm` + `calcium_xml/{patient_id}.xml`

**Non-gated CT (regression):** separate dataset root with `{patient_id}/*.dcm` + `scores.xlsx`

Tests requiring the real dataset use the `requires_dataset` pytest marker and are auto-skipped if the path is absent. All other tests use synthetic data.

## Requirements & Traceability

Requirements are declared in `docs/requirements.yaml` using prefixes: `SYS-`, `DAT-`, `TSK-`, `TRN-`, `MOD-`, `RSK-`, `INF-`, `REP-`, `VER-`, `DOC-`. Tests are linked to requirements via the `@pytest.mark.requirement("REQ-ID")` marker. `runtests.py` generates the traceability matrix at `docs/traceability_matrix.md` and must maintain ≥ grade B in the forge health report.

## Artifacts

Generated under `artifacts/` (gitignored):
- `artifacts/training_runs/` — `training_report.json`, `model.pt`, `partitions.json`
- `artifacts/tuning_runs/` — `tuning_report.json` per sweep
- `artifacts/model_testing_runs/` — `model_testing_report.json`
- `artifacts/evidence_runs/` — evidence output per test session
