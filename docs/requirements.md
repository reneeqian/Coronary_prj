# System Requirements — Coronary Artery Calcium (CAC) Detection Pipeline

## 1. Requirements Overview

This project defines requirements across multiple domains to reflect
best practices for regulated medical imaging software and SaMD tooling.

Requirements are grouped by intent:

- Dataset and data integrity requirements
- Medical image data contract requirements
- Training pipeline functional requirements
- Model artifact requirements
- System-level auditability and reproducibility requirements

Not all requirements are enforced in the current implementation.
Some are documented to establish design intent and future validation scope.


## 2. Purpose

This document defines the functional, data, and non-functional requirements for a software pipeline that ingests cardiac CT data and supports coronary artery calcium (CAC) detection for research and educational purposes.

The primary objective is to demonstrate robust **medical AI software engineering practices**, including data contracts, enforcement, traceability, and reproducibility, rather than clinical deployment or model optimization. 

Traceability ends at the tensor adapter boundary.

---

## 3. Scope

The system shall:
- Ingest publicly available non-contrast cardiac CT datasets
- Enforce explicit data contracts on input structure and semantics
- Provide a reproducible dataset abstraction suitable for downstream machine learning workflows
- Support enforcing and testing of data integrity

The system shall **not**:
- Provide diagnostic output
- Make clinical claims
- Be used for patient care

### 3.1 Traceability Scope

The traceability matrix covers requirements for the coronary_prj system from data ingestion through enforcing PatientSample data contract and record evidence.

Conversion of PatientSample objects into framework-specific tensor representations, along with training and optimization logic, is handled by the medical_image_ai_toolkit module and is out of scope for this document.

### 3.2 Assumptions and Constraints

- Input datasets are de-identified and publicly available.
- CT volumes represent non-contrast cardiac imaging.
- Ground truth labels (if present) are assumed to be externally generated.

---

## 4. Definitions and Abbreviations

| Term | Definition |
|----|----|
| CAC | Coronary Artery Calcium |
| CT | Computed Tomography |
| HU | Hounsfield Unit |
| Dataset | Structured collection of CT volumes and associated metadata |

---

## 5. Requirements Legend

The following table defines the requirement prefixes used throughout this project.

| Prefix | Requirement Type | Description | Example |
|------:|------------------|-------------|---------|
| **FR** | Functional Requirement | Defines system behavior, capabilities, or workflows. | `CAC-FR-01: The system shall load coronary CT volumes from disk.` |
| **DR** | Data Requirement | Defines constraints, assumptions, and guarantees on input data and labels. | `MIT-DR-02: Input CT volumes shall be 3D arrays.` |
| **MR** | Model Requirement | Defines constraints on model inputs, outputs, and training assumptions. | `MIT-MR-01: The model shall accept single-channel CT volumes.` |
| **TR** | Training Requirement | Defines requirements for training configuration, determinism, and execution behavior. | `MIT-TR-01: Training shall be deterministic given fixed seeds.` |
| **MAR** | Model Artifact Requirement | Defines expectations for artifacts generated during training and validation. | `MIT-MAR-02: Model artifacts shall include output shape metadata.` |
| **SYS** | System Auditability & Reproducibility | Defines requirements for evidence generation and regulatory traceability. | `MIT-SYS-01: Training runs shall generate immutable evidence artifacts.` |
| **VRF** | Verification Requirement | Defines test-time checks that verify requirements are met. | `MIT-VRF-03: The system shall reject invalid voxel spacing.` |
| **VAL** | Validation Requirement | Reserved for validation against intended clinical use. | *(Reserved – not used in this project)* |
| **NFR** | Non-Functional Requirement | Defines performance, reliability, maintainability, or reproducibility constraints. | `CAC-NFR-01: Dataset loading shall complete within 2 seconds per case.` |

### 5.1 Identification Format

Each requirement is uniquely identified using the format:

`<PROJ>-<PREFIX>-<NN>`

Where:
- `<PROJ>` is either **CAC** (Coronary Artery Calcium Project) or **MIT** (Medical Image AI Toolkit)
- `<PREFIX>` is one of the requirement types listed above
- `<NN>` is a zero-padded numeric identifier (e.g., `01`, `02`)


### 5.2 Traceability

Requirements may trace to:
- Clinical assumptions (`CA-*`)
- Dataset contracts and enforcers
- Unit or integration tests
- Documentation artifacts

All requirements are expected to be traceable to at least one implementation or verification mechanism.

### 5.3 Verification Strategy

Verification of requirements shall be performed through:
- Dataset contract enforcement
- Unit tests
- Script-based sanity checks

Verification artifacts are documented in `docs/traceability.md`.

---

## 6. Test & Evidence Naming Conventions

This section defines mandatory naming conventions for test files, test functions, and evidence artifacts to ensure deterministic traceability between requirements, verification activities, and generated evidence in accordance with FDA SaMD expectations (IEC 62304, FDA AI/ML Good Machine Learning Practice).

### 6.1 Test Function Naming

Each test function shall reference the requirement it verifies.

**Format**

`def test_<REQ_ID>_<behavior_under_test>():`


**Rules**
* The requirement ID must appear verbatim
* The function name must clearly state the expected behavior or constraint
* Tests verifying negative behavior (rejection, failure) should explicitly state so

**Examples**

```python
def test_CAC_DR_01_valid_ct_volume_is_accepted():
    ...

def test_CAC_DR_01_non_3d_volume_is_rejected():
    ...

def test_MIT_SYS_01_evidence_written_once_and_immutable():
    ...
```

### 6.2 Requirement Annotation (Optional but Recommended)

Tests may explicitly annotate the requirement they verify to support automated traceability extraction.

Example

```python
@pytest.mark.requirement("CAC-DR-01")
def test_CAC_DR_01_valid_ct_volume_is_accepted():
    ...
```

### 6.3 Evidence Artifact Naming

Each verification test shall generate a corresponding immutable evidence artifact.

**Format**

`<REQ_ID>_<test_name>_<YYYYMMDD_HHMMSS>.json`


**Rules**
* `<REQ_ID>` must match the requirement under verification
* `<test_name>` must match the pytest function name
* Timestamps must be UTC

Evidence artifacts are write-once and never overwritten

**Examples**

```text
CAC-DR-01_test_CAC_DR_01_valid_ct_volume_is_accepted_20260210_153422.json
MIT-SYS-01_test_MIT_SYS_01_evidence_written_once_and_immutable_20260210_153501.json
```

### 6.4 Evidence JSON Required Fields

All evidence artifacts shall include the following minimum fields to support regulatory auditability and traceability:

```json
{
  "project_id": "CAC",
  "requirement_id": "CAC-DR-01",
  "requirement_type": "Data Requirement",
  "test_id": "test_CAC_DR_01_valid_ct_volume_is_accepted",
  "result": "PASS",
  "timestamp_utc": "2026-02-10T15:34:22Z",
  "inputs": { },
  "outputs": { },
  "pass_fail_criteria": "Input CT volume is a 3D array",
  "tool_name": "medical-image-ai-toolkit",
  "tool_version": "0.1.0",
  "code_revision": "<git_commit_hash>"
}
```

Additional fields may be included as needed (e.g., dataset identifiers, model hashes, configuration snapshots).

### 6.5 Traceability Guarantee

Adherence to the above conventions guarantees a deterministic and auditable trace path:

```text
Requirement → Test File → Test Function → Evidence Artifact → Result
```

This structure enables automated generation of requirement traceability matrices for regulatory review and supports reuse of the Medical Image AI Toolkit across multiple SaMD projects.

---

## 4. CAC – Dataset & Collection Requirements

| Req ID | Description | Project | Status | Rationale |
| --------- | -------------------------- | --- | ------- | ------------------------------------- |
| CAC-FR-01 | The system shall load coronary CT volumes from disk using explicit file paths. | CAC | Enforced | Ensures controlled and reproducible dataset ingestion. |
| CAC-FR-02 | The system shall associate each CT volume with a unique case identifier. | CAC | Enforced | Prevents case ambiguity and supports traceability. |
| CAC-DR-01 | Input CT volumes shall be 3D arrays with consistent axial orientation. | CAC | Enforced | Ensures compatibility with downstream processing and training. |
| CAC-DR-02 | CT voxel spacing shall be explicitly provided and validated. | CAC | Enforced | Prevents invalid physical interpretation of imaging data. |
| CAC-DR-03 | Labels shall be numeric and bounded within a predefined domain. | CAC | Enforced | Prevents undefined or unsafe training behavior. |
| CAC-NFR-01 | Dataset loading shall complete within a reasonable time per case on commodity hardware. | CAC | Documented | Establishes performance expectations without strict guarantees. |


---

## MIT – Medical Image Data Contract Requirements

| Req ID | Description | Project | Status | Rationale |
| --------- | -------------------------- | --- | ------- | ------------------------------------- |
| MIT-DR-01 | Medical image samples shall include image data, spacing metadata, and identifiers. | MIT | Enforced | Establishes a strict data contract for all toolkit consumers. |
| MIT-DR-02 | Image spacing values shall be positive and non-zero. | MIT | Enforced | Prevents invalid geometric assumptions during training. |
| MIT-DR-03 | Image tensor shapes shall be validated prior to training. | MIT | Enforced | Prevents silent shape mismatches in models. |
| MIT-MR-01 | Models shall explicitly define expected input tensor shape. | MIT | Documented | Prevents ambiguous model usage across projects. |
| MIT-MR-02 | Models shall define the semantic meaning of each output channel. | MIT | Documented | Supports correct downstream interpretation of results. |
| MIT-TR-01 | Training shall be deterministic when provided fixed random seeds and identical inputs. | MIT | Planned | Enables reproducibility and auditability of results. |
| MIT-TR-02 | Training shall require an explicit configuration object. | MIT | Enforced | Prevents undocumented or implicit training behavior. |
| MIT-TR-03 | Training shall fail fast upon detection of invalid data or configuration. | MIT | Enforced | Prevents silent propagation of invalid states. |
| MIT-TR-04 | Training shall not silently coerce data types or shapes. | MIT | Documented | Prevents unintentional changes to training behavior. |
| MIT-MAR-01 | Training runs shall produce versioned model artifacts. | MIT | Planned | Enables model comparison and archival. |
| MIT-MAR-02 | Model artifacts shall include training configuration metadata. | MIT | Planned | Supports traceability and reproducibility. |
| MIT-MAR-03 | Model artifacts shall include dataset identifiers or hashes. | MIT | Planned | Enables linkage between models and training data. |
| MIT-SYS-01 | Each training run shall generate immutable evidence artifacts. | MIT | Enforced | Supports regulated development workflows and audits. |
| MIT-SYS-02 | Training evidence shall include timestamps and execution context. | MIT | Enforced | Enables reconstruction of training events. |
| MIT-SYS-03 | The system shall record code version identifiers for training runs. | MIT | Documented | Supports root-cause analysis and reproducibility. |
| MIT-VRF-01 | The system shall reject medical image samples violating data contracts. | MIT | Enforced | Verifies enforcement of MIT-DR requirements. |
| MIT-VRF-02 | The system shall surface descriptive errors for invalid training inputs. | MIT | Enforced | Ensures debuggability and safe failure modes. |

---

## 10. Disclaimer

This system is intended for research and educational purposes only and is not a medical device.
