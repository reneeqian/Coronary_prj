# Software Architecture and Design (SDS)

## 1. Purpose

This document defines the architectural design of Coronary_prj.

The design reflects IEC 62304-style modular decomposition but is demonstrative only.

---

# 2. High-Level Architecture

Modules:

- Ingestion Layer
- Data Representation Layer
- Model Layer
- Scoring Algorithm Layer
- Verification & Traceability Layer

---

# 3. Module Responsibilities

## 3.1 Ingestion Module

Class: COCAGatedIngestor

Responsibilities:
- Locate dataset root
- Deterministically list patients
- Load DICOM slices
- Construct ordered volume
- Extract spacing

---

## 3.2 Data Representation

Class: PatientSample

Encapsulates:
- patient_id
- image_volume
- spacing
- annotations (optional)

This ensures contract enforcement and validation boundaries.

---

## 3.3 Model Layer

Responsibilities:
- Load trained model
- Perform inference
- Produce segmentation or detection output

---

## 3.4 Scoring Layer

Responsibilities:
- Apply intensity thresholds
- Compute volumetric score

---

# 4. Determinism Strategy

- Sorted patient listing
- Sorted slice loading
- Controlled random seeds
- Reproducible test execution

---

# 5. Design Philosophy

- Separation of concerns
- Explicit failure states
- No hidden global state
- Traceable requirement markers
- Reproducible execution

This mirrors regulated software design best practices.
