# Validation Scope

## 1. Purpose

This document defines the scope and philosophy of validation within the Coronary Artery Calcium (CAC) training pipeline.

Validation exists to ensure **structural correctness, safety, and contract compliance** of data prior to model consumption. It does not attempt to assess clinical correctness or model performance.

---

## 2. Validation Boundary

All validation is enforced at a **single dataset boundary**, prior to conversion into framework-specific tensors.

Once a sample has passed validation:
- Downstream components may assume correctness
- No redundant validation is performed in training or modeling code

This boundary is enforced by the `PatientSampleValidator`.

---

## 3. What Is Validated

Validation checks include:
- Image dimensionality (3D volumes)
- Image–label shape consistency
- Finite numeric image values
- Allowed label values
- Dataset postconditions prior to training

These checks correspond directly to defined **Validation Requirements (VR-01 through VR-06)**.

---

## 4. What Is Not Validated

The system explicitly does **not** validate:
- Clinical accuracy or diagnostic correctness
- Annotation quality or inter-reader agreement
- Biological plausibility of findings
- Model outputs or predictions

These concerns are outside the scope of this project.

---

## 5. Validation Timing

Validation is performed:
- During dataset construction
- Prior to tensor conversion
- Prior to training loop execution

Validation is not performed:
- Per training iteration
- Inside model forward passes

---

## 6. Failure Behavior

Validation failures:
- Fail fast
- Raise explicit, actionable error messages
- Prevent invalid data from reaching training code

---

## 7. Rationale

This approach:
- Centralizes responsibility
- Improves testability and CI compatibility
- Aligns with medical AI software engineering best practices
- Enables clean separation of concerns

