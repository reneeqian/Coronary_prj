# Coronary Artery Calcium (CAC) Detection – Engineering Demonstration Project

This repository demonstrates structured development of a medical imaging AI
pipeline under Software as a Medical Device (SaMD) design control principles.

The goal is **engineering rigor**, not model performance.

This project is for demonstration only and is NOT intended for clinical use.

---

## Purpose

This project demonstrates:

- deterministic DICOM ingestion
- structured dataset construction
- automated verification
- requirements traceability
- evidence artifact generation

The goal is **engineering rigor**, not model performance.

---

## Repository Structure

```
Coronary_prj/
├── docs/
│ └── requirements.yaml
├── src/
│ └── Coronary_prj/
├── tests/
├── artifacts/
└── runtests.py
```


---

## How to Run

### Run Tests + Evidence Capture

```
python runtests.py
```


This will:

1. run all tests
2. compute code coverage
3. validate requirement traceability
4. generate a traceability matrix
5. record uncovered code

Outputs:

```
artifacts/
docs/traceability_matrix.md
```


---

## Dependencies

- `medical_image_ai_toolkit`
- `regulatory_tools`

Requirements follow the convention defined in regulatory_tools/docs/Requirements_Convention.md