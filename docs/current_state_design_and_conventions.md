# Current State Design & Conventions
**Project:** Coronary CT ML Pipeline  
**Last Updated:** 2026-01-28  
**Status:** Active – authoritative snapshot of current architecture and conventions

---

## 1. Purpose

This document describes the **current, authoritative design state** of the project.

It answers:
- What abstractions exist today
- Where responsibilities live
- What assumptions downstream code may rely on
- What conventions are considered “locked” at this stage

This document is intended to:
- Reduce cognitive load
- Prevent accidental architectural drift
- Serve as a reference during implementation, testing, and training

Historical rationale is captured separately in the Design History (DHF-lite).

---

## 2. High-Level Architecture

### Design Principle
The system is divided into:
- **Reusable medical imaging framework code** (`medical_image_ai_toolkit`)
- **Project- and dataset-specific code** (`coronary_prj`)

Framework code must be importable and usable as if it were an external package.

---

## 3. Core Data Contract

### PatientSample

`PatientSample` is the canonical patient-level data representation.

**Guaranteed fields:**
- `image_volume`: NumPy array, shape `(z, y, x)`
- `spacing`: `(z, y, x)` tuple, all values > 0
- `patient_id`: non-empty string
- `annotations`: optional; supported formats:
  - Vector ROIs (slice-indexed)
  - Dense raster mask (NumPy array)

**Contract enforcement:**
- All invariants are enforced by:
  ```python
  enforce_patient_sample_contract(...)
