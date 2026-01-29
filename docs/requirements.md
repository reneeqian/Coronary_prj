# System Requirements — Coronary Artery Calcium (CAC) Detection Pipeline

## 1. Purpose

This document defines the functional, data, and non-functional requirements for a software pipeline that ingests cardiac CT data and supports coronary artery calcium (CAC) detection for research and educational purposes.

The primary objective is to demonstrate robust **medical AI software engineering practices**, including data contracts, enforcement, traceability, and reproducibility, rather than clinical deployment or model optimization. 

Traceability ends at the tensor adapter boundary.

---

## 2. Scope

The system shall:
- Ingest publicly available non-contrast cardiac CT datasets
- Enforce explicit data contracts on input structure and semantics
- Provide a reproducible dataset abstraction suitable for downstream machine learning workflows
- Support enforcing and testing of data integrity

The system shall **not**:
- Provide diagnostic output
- Make clinical claims
- Be used for patient care

### 2.1 Traceability Scope

The traceability matrix covers requirements for the coronary_prj system from data ingestion through enforcing PatientSample data contract and record evidence.

Conversion of PatientSample objects into framework-specific tensor representations, along with training and optimization logic, is handled by the medical_image_ai_toolkit module and is out of scope for this document.


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
| **VRF** | Verification Requirement | Defines test-time checks that verify requirements. | `VRF-03: The system shall reject labels containing invalid values.` |
| **VAL** | Validation Requirement | Reserved for V&V validation against intended use. |
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
- Dataset contracts and enforcers
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

### FR-02: Dataset Structure Enforcement

**FR-02.1**  
The system shall enforce a predefined directory structure for processed datasets.

**FR-02.2**  
The system shall enforce the presence and non-emptiness of required image and label subdirectories.

---

### FR-03: Metadata and Image Integrity Enforcement

**FR-03.1**  
The system shall verify that image volumes are readable and non-corrupt.

**FR-03.2**  
The system shall enforce that image volumes contain valid numeric data.

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

The following data requirements define the PatientSample data contract enforced by the coronary_prj system.
All downstream components may assume these requirements hold once a PatientSample has passed enforcement.

### DR-01: Volumetric CT Representation 

**DR-01.1**  
A PatientSample shall contain an image_volume represented as a NumPy array.

**DR-01.2**  
The image_volume shall be three-dimensional with shape (z, y, x).

**Derived From:** CA-2  
**Rationale:** Cardiac CT data is inherently volumetric.

---

### DR-02: Valid Image Spatial Dimensions

**DR-02.1**. 
The spatial dimensions (y, x) of the image volume shall be greater than zero.

**Rationale:** Prevents degenerate or malformed image data.

---

### DR-03: Image Spacing Definition

**DR-03.1**  
A PatientSample shall define voxel spacing as a tuple (z, y, x).

**DR-03.2**  
All spacing values shall be strictly greater than zero.

**Rationale:** Spatial reasoning and annotation alignment depend on valid spacing.

---

### DR-04: Patient Identifier Integrity

**DR-04.1**  
Each PatientSample shall define a non-empty patient_id.

**Rationale:** Required for traceability, logging, and dataset bookkeeping

---

### DR-05: Annotation Presence Policy

**DR-05.1**  
A PatientSample may optionally include annotations.

**DR-05.2**  
If annotations are required by the consuming workflow, the system shall reject samples without annotations.

**Rationale:** Supports both labeled and unlabeled datasets while preserving explicit intent.

---

### DR-06: Annotation Slice Validity

**DR-06.1**  
All annotation slice indices shall be integers.

**DR-06.2**  
All annotation slice indices shall lie within the bounds of the image volume depth.

**Rationale:** Prevents spatial misalignment between annotations and image data.

---

### DR-07: Annotation Geometry Validity

**DR-07.1**  
All vector ROI contours shall be defined as (N, 2) arrays of pixel coordinates.

**DR-07.2**  
All ROI coordinates shall lie within the spatial bounds of the image volume.

**Rationale:** Ensures annotations are geometrically valid and renderable.

---

### DR-08: Unified Data Contract Boundary

**DR-08.1**  
All data requirements (DR-01 through DR-07) shall be enforced at the PatientSample contract boundary.

**DR-08.2**  
Downstream systems may assume all enforced PatientSample objects satisfy these requirements.

**Rationale:** Establishes a single, authoritative data contract.
---

## 7. Non-Functional Requirements

### NFR-01: Maintainability

**NFR-01.1**  
The codebase shall follow modular design principles.

**NFR-01.2**  
Dataset contract enforcement logic shall be isolated from dataset iteration logic.

---

### NFR-02: Traceability

**NFR-02.1**  
Each requirement shall be traceable to at least one implementation artifact.

**NFR-02.2**  
Each requirement shall be verifiable via tests or contract enforcement checks.

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
- Dataset contract enforcement
- Unit tests
- Script-based sanity checks

Verification artifacts are documented in `docs/traceability.md`.

---

## 10. Disclaimer

This system is intended for research and educational purposes only and is not a medical device.
