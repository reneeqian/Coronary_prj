# Software Requirements Specification (SRS)

## 1. Scope

This document defines structured software requirements for Coronary_prj.

These requirements are written in the style of a regulated medical device
software specification, but the project is demonstrative only.

This software is NOT intended for clinical use.

---

# 2. Functional Requirements

## CAC-FR-01: Dataset Ingestion
The system shall ingest gated CT DICOM datasets from a specified dataset root directory.

## CAC-FR-02: Deterministic Ordering
The system shall deterministically list and process patient datasets in sorted order.

## CAC-FR-03: Volume Construction
The system shall construct a 3D image volume from DICOM slices using metadata-based ordering.

## CAC-FR-04: Standardized Data Representation
The system shall represent each dataset as a standardized PatientSample object.

## CAC-FR-05: Model Inference
The system shall apply a CAC detection model to a valid CT volume.

## CAC-FR-06: Calcium Scoring
The system shall compute a calcium score using volumetric and intensity-based logic.

## CAC-FR-07: Failure Handling
The system shall raise an exception if the dataset root does not exist.

---

# 3. Data Requirements

## CAC-DR-01: DICOM Validation
Input images shall be valid DICOM files.

## CAC-DR-02: Spacing Preservation
Voxel spacing shall be extracted and preserved.

## CAC-DR-03: Non-Empty Volume
Constructed volumes shall be non-empty.

---

# 4. Determinism & Safety-Oriented Requirements

## CAC-SR-01: Deterministic Behavior
Given identical input and seed configuration, outputs shall be reproducible.

## CAC-SR-02: Explicit Failure
Invalid datasets shall not produce silent or undefined behavior.

---

# 5. Verification Requirement

Each requirement shall be traceable to at least one automated test.

---

## 6. Regulatory Positioning

This document models what an SRS would look like for an FDA-regulated
Software as a Medical Device (SaMD), but the project remains non-clinical.
