# Coronary CT AI – System Requirements

## 1. Scope
This system processes coronary CT images to support AI-based analysis of coronary anatomy and disease-related features.

## 2. Functional Requirements

### FR-001 Dataset Ingestion
The system shall ingest coronary CT imaging data from a predefined raw data directory.

**Rationale:** Enables standardized dataset construction.
**Verification:** Unit test validates raw directory scanning.

---

### FR-002 Dataset Validation
The system shall validate image dimensionality, spacing, and intensity ranges prior to training.

**Verification:** validators.py tests fail on invalid samples.

---

### FR-003 Dataset Access API
The system shall expose a dataset interface compatible with PyTorch DataLoader.

**Verification:** test_dataset.py successfully iterates dataset.
