# System Requirements — Coronary Artery Calcium (CAC) Detection Pipeline

## 1. Purpose

This document defines the functional, data, validation, and non-functional requirements for a software pipeline that ingests cardiac CT data and supports coronary artery calcium (CAC) detection for research and educational purposes.

The primary objective is to demonstrate robust **medical AI software engineering practices**, including data contracts, validation, traceability, and reproducibility, rather than clinical deployment or model optimization.

---

## 2. Scope

The system shall:
- Ingest publicly available non-contrast cardiac CT datasets
- Enforce explicit data contracts on input structure and semantics
- Provide a reproducible dataset abstraction suitable for downstream machine learning workflows
- Support validation and testing of data integrity

The system shall **not**:
- Provide diagnostic output
- Make clinical claims
- Be used for patient care

---

## 3. Definitions and Abbreviations

| Term | Definition |
|----|----|
| CAC | Coronary Artery Calcium |
| CT | Computed Tomography |
| HU | Hounsfield Unit |
| Dataset | Structured collection of CT volumes and associated metadata |

---

## Requirements Legend

The following table defines the requirement prefixes used throughout this project.

| Prefix | Requirement Type | Description | Example |
|------:|------------------|-------------|---------|
| **FR** | Functional Requirement | Defines system behavior, capabilities, or workflows. | `FR-01: The system shall load coronary CT volumes from disk.` |
| **DR** | Data Requirement | Defines constraints, assumptions, and guarantees on input data and labels. | `DR-02: Input CT volumes shall be 3D arrays.` |
| **VR** | Validation Requirement | Defines runtime or test-time checks that enforce requirements. | `VR-03: The system shall reject labels containing invalid values.` |
| **MR** | Model Requirement | Defines constraints on model inputs, outputs, and training assumptions. | `MR-01: The model shall accept single-channel CT volumes.` |
| **NFR** | Non-Functional Requirement | Defines performance, reliability, maintainability, or reproducibility constraints. | `NFR-01: Dataset loading shall complete within 2 seconds per case.` |

### Identification Format

Each requirement is uniquely identified using the format:

`<PREFIX>-<NN>`

Where:
- `<PREFIX>` is one of the requirement types listed above
- `<NN>` is a zero-padded numeric identifier (e.g., `DR-01`, `FR-02`)

### Traceability

Requirements may trace to:
- Clinical assumptions (`CA-*`)
- Dataset validators
- Unit or integration tests
- Documentation artifacts

All requirements are expected to be traceable to at least one implementation or verification mechanism.

---

## 4. Functional Requirements

### FR-01: Dataset Ingestion

**FR-01.1**  
The system shall ingest cardiac CT image volumes from a specified root directory.

**FR-01.2**  
The system shall support volumetric image formats commonly used in public CT datasets.

**FR-01.3**  
The system shall fail gracefully with informative errors when required directories or files are missing.

---

### FR-02: Dataset Structure Validation

**FR-02.1**  
The system shall enforce a predefined directory structure for processed datasets.

**FR-02.2**  
The system shall validate the presence and non-emptiness of required image and label subdirectories.

---

### FR-03: Metadata and Image Integrity Validation

**FR-03.1**  
The system shall verify that image volumes are readable and non-corrupt.

**FR-03.2**  
The system shall validate that image volumes contain valid numeric data.

**FR-03.3**  
The system shall verify that image intensity values fall within expected CT Hounsfield Unit ranges.

---

### FR-04: Dataset Abstraction

**FR-04.1**  
The system shall provide a dataset abstraction that exposes individual samples via a consistent interface.

**FR-04.2**  
The dataset abstraction shall support iteration over samples.

**FR-04.3**  
The dataset abstraction shall decouple data loading from downstream modeling code.

---

### FR-05: Reproducibility

**FR-05.1**  
The system shall define all dependencies via a version-controlled environment specification.

**FR-05.2**  
The system shall produce deterministic dataset ordering when configured to do so.

---

## 5. Data Requirements (DR)

### DR-01: Volumetric CT Representation  
CT images shall be represented as three-dimensional volumetric arrays corresponding to anatomical cardiac imaging.

**Derived From:** CA-2  
**Rationale:** Coronary anatomy is spatial and volumetric.

---

### DR-02: Image–Label Spatial Consistency  
Each image volume shall have a corresponding label volume with identical spatial dimensions.

**Derived From:** CA-3  
**Rationale:** Enables voxel-wise learning and validation.

---

### DR-03: Numerical Integrity of Image Data  
Image volumes shall contain only finite numerical values.

**Derived From:** CA-4  
**Rationale:** Prevents numerical instability in preprocessing and modeling.

---

### DR-04: Valid Label Semantics  
Label volumes shall contain only predefined discrete class values representing coronary calcium annotations.

**Derived From:** CA-5  
**Rationale:** Prevents annotation corruption and class ambiguity.

---

### DR-05: Dataset Readiness Guarantee  
All dataset samples shall satisfy defined postconditions prior to downstream model consumption.

**Derived From:** System Safety and Reproducibility Goals

---

## 6. Validation Requirements (VR)

### VR-01: Enforce 3D Volume Dimensionality  
The system shall reject image or label inputs that are not three-dimensional.

**Satisfies:** DR-01  
**Implemented By:** 

---

### VR-02: Enforce Image–Label Shape Matching  
The system shall reject samples where image and label volumes differ in shape.

**Satisfies:** DR-02  
**Implemented By:** `validate_shape_match()`

---

### VR-03: Enforce Finite Image Values  
The system shall reject image volumes containing NaN or infinite values.

**Satisfies:** DR-03  
**Implemented By:** `validate_finite()`

---

### VR-04: Enforce Allowed Label Values  
The system shall reject label volumes containing values outside the defined class set.

**Satisfies:** DR-04  
**Implemented By:** `validate_label_values()`

---

### VR-05: Enforce Dataset Postconditions  
The system shall assert dataset postconditions prior to model input.

**Satisfies:** DR-05  
**Implemented By:** `assert_postconditions()`

---

### VR-06: Enforce Unified Data Contract Boundary  
The system shall enforce all validation requirements through a single dataset boundary.

**Satisfies:** DR-01 through DR-05  
**Implemented By:** `enforce_data_contract()`

---

## 7. Non-Functional Requirements

### NFR-01: Maintainability

**NFR-01.1**  
The codebase shall follow modular design principles.

**NFR-01.2**  
Dataset validation logic shall be isolated from dataset iteration logic.

---

### NFR-02: Traceability

**NFR-02.1**  
Each requirement shall be traceable to at least one implementation artifact.

**NFR-02.2**  
Each requirement shall be verifiable via tests or validation checks.

---

### NFR-03: Transparency

**NFR-03.1**  
Assumptions about dataset structure and content shall be documented.

**NFR-03.2**  
Limitations of the pipeline shall be explicitly stated in documentation.

---

## 8. Assumptions and Constraints

- Input datasets are de-identified and publicly available.
- CT volumes represent non-contrast cardiac imaging.
- Ground truth labels (if present) are assumed to be externally generated.

---

## 9. Verification Strategy

Verification of requirements shall be performed through:
- Dataset validators
- Unit tests
- Script-based sanity checks

Verification artifacts are documented in `docs/traceability.md`.

---

## 10. Disclaimer

This system is intended for research and educational purposes only and is not a medical device.
