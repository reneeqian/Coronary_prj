# Requirements Key — Coronary_prj

This file defines the abbreviations, domain prefixes, type values, and regulatory mappings used
across all requirements files in this project. It is maintained by hand. The DHF generator does
not read it — it reads the `metadata` block in each YAML file. This document explains *why* the
conventions exist; the metadata enforces them.

For ID format conventions (prefix, numbering, uniqueness rules), see
`COCA-prj-DHF/requirements_convention.md`.

---

## Domain Prefix Registry

| Prefix | Full Name | Source File | Regulatory Role |
|--------|-----------|-------------|-----------------|
| UN | User Need | `user_needs.yaml` | IEC 62366 §4 / ISO 14971 intended use. What the radiologist must be able to do. Maps to `intended_use.md` and the UN rows of the RTM. 510(k) Section 9. |
| SYS | System Requirement | `requirements.yaml` | IEC 62304 §5.2 SRS. Observable behavior from the outside — no implementation detail. Maps to `system_requirements.md`. 510(k) Section 12. |
| DAT | Data Requirement | `requirements.yaml` | IEC 62304 §5.2 SRS. Data ingestion, validation, and access behavior. |
| TSK | Task Requirement | `requirements.yaml` | IEC 62304 §5.2 SRS. Task definition and preprocessing behavior. |
| TRN | Training Requirement | `requirements.yaml` | IEC 62304 §5.2 SRS. Model training process behavior. |
| MOD | Model Requirement | `requirements.yaml` | IEC 62304 §5.2 SRS. Model capability and artifact requirements. |
| INF | Inference Requirement | `requirements.yaml` | IEC 62304 §5.2 SRS. Inference pipeline behavior. |
| REP | Reporting Requirement | `requirements.yaml` | IEC 62304 §5.2 SRS. Report generation and visualization behavior. |
| VER | Verification Requirement | `requirements.yaml` | IEC 62304 §5.7 V&V. Test execution and evidence generation. |
| DOC | Documentation Requirement | `requirements.yaml` | IEC 62304 §4.3 DHF. Machine-readable documentation and traceability. |
| RSK | Risk Control | `risk_controls.yaml` | ISO 14971 §6.7. What the system does to reduce a specific hazard to an acceptable risk level. Maps to `risk_control_measures.md`. |
| HAZ | Hazard | `hazard_analysis.yaml` | ISO 14971 §4–5. Identification of potential harms, their causes, and severity. Maps to `hazard_analysis.md`. Narrative columns are filled by hand. |
| DHF | DHF Generator Requirement | `regulatory_tools/docs/requirements.yaml` | Tooling requirements for the DHF auto-population system. Not part of the device's regulatory submission — governs the tooling. |

> **Note on design requirements:** Several requirements with SYS/DAT/TSK/TRN/MOD prefixes live
> in `design.yaml` rather than `requirements.yaml`. These describe *how* the system is built
> (architecture constraints, algorithm choices, interface contracts) rather than *what* it must do.
> See the SRS vs SDS section below for the distinguishing test.

---

## `type` Field Values

| Value | Plain English | When to Use | Regulatory Framework |
|-------|---------------|-------------|----------------------|
| `user_need` | What the radiologist needs to achieve | Top of the hierarchy. No `derived_from` required. Written in terms of clinical outcome, not software behavior. | IEC 62366 §4, ISO 14971 intended use |
| `system_requirement` | What the software must do (observable behavior) | Could a clinical evaluator write this without knowing the code? If yes, it's a system requirement. | IEC 62304 §5.2 SRS |
| `design_requirement` | How the software does it | Architecture constraint, algorithm choice, named interface contract. Implementation-facing. | IEC 62304 §5.4 SDS |
| `risk_control` | What we do to reduce a specific hazard | Paired with a HAZ-* entry via `hazard_ref`. Describes a specific protective measure. | ISO 14971 §6.7 |

---

## SRS vs SDS: How to Decide

**The clinical evaluator test:** Could someone with deep medical device knowledge but no software
background write this requirement? If yes → `system_requirement` (SRS). If no → `design_requirement` (SDS).

- `system_requirement` example: "Patient ingestion shall produce structured in-memory patient
  data with consistent volume ordering." — Observable, testable from the outside.
- `design_requirement` example: "CT slices shall be ordered by ImagePositionPatient[2] before
  volume construction." — A specific algorithmic decision a clinician would not make.

Both types are tested. The distinction changes *who* the requirement serves and which DHF
document it feeds, not whether it gets a test.

When planning new work: write the system requirement first, then derive the design requirement
from it. A design requirement without a parent system requirement is an orphan — the DHF
validator will reject it (DHF-011).

---

## Standard Abbreviations

| Abbreviation | Meaning |
|---|---|
| SRS | Software Requirements Specification (IEC 62304 §5.2) |
| SDS | Software Design Specification (IEC 62304 §5.4) |
| RTM | Requirements Traceability Matrix |
| DHF | Design History File (IEC 62304 §4.3) |
| SaMD | Software as a Medical Device |
| 510(k) | FDA premarket notification pathway for medical devices |
| IEC 62304 | Medical device software lifecycle processes standard |
| ISO 14971 | Medical device risk management standard |
| IEC 62366 | Medical device usability engineering standard |
| SOUP | Software of Unknown Provenance (third-party dependencies) |
| V&V | Verification and Validation |
| RTM | Requirements Traceability Matrix |

---

## Adding a New Domain Prefix

Adding a new prefix requires updates in three places (intentional friction):

1. **New YAML file** with `metadata.allowed_prefixes`, `metadata.allowed_types`,
   `metadata.file_role`, and `metadata.regulatory_role` fields
2. **`COCA-prj-DHF/dhf_context.yaml`** — add the new file path under `data_sources.requirements`
3. **This file** — add a row to the Domain Prefix Registry above

The validator will reject any requirement whose ID prefix is not listed in its file's
`metadata.allowed_prefixes`. Skipping any of the three updates will cause a DHFValidationError
on the next `generate_dhf.py` run.
