# Coronary Artery Calcium (CAC) Detection – Engineering Demonstration Project

## 1. Purpose

This repository demonstrates structured development of a medical imaging AI
pipeline under Software as a Medical Device (SaMD) design control principles.

The goal is **engineering rigor**, not model performance.

This project is for demonstration only and is NOT intended for clinical use.

---

## 2. Scope

The system:

- Ingests gated cardiac CT DICOM studies
- Constructs deterministic 3D volumes
- Applies preprocessing
- Produces coronary artery calcium (CAC) detection outputs
- Captures traceable verification artifacts

---

## 3. Engineering Objectives

- Deterministic execution
- Structured requirements
- Automated verification
- Traceability to tests
- Evidence artifact generation
- Reproducible runs

---

## 4. Repository Structure

```
Coronary_prj/
├── docs/
│ ├── requirements.yaml
│ └── 02_Software_Requirements_Specification.md
├── tests/
├── artifacts/
├── run_tests_and_trace.py
├── environment.yml
└── README.md
```


---

## 5. How to Run

### Create Environment

```bash
conda env create -f environment.yml
conda activate coronary-prj-env
```

### Run Tests + Evidence Capture

```
pytest
```

or 

```
python run_tests_and_trace.py
```

## 6. Regulatory Position

This repository demonstrates engineering patterns aligned with:
* Deterministic processing
* Structured requirement definition
* Traceability
* Verification documentation

It does NOT represent a cleared or approved medical device.


Requirements follow the convention defined in regulatory_tools/docs/Requirements_Convention.md