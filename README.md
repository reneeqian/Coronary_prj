# Coronary Artery Calcium (CAC) Detection from Cardiac CT

## Overview

This repository implements an end-to-end medical AI software pipeline for **coronary artery calcium (CAC) detection** from non-contrast cardiac CT imaging using publicly available datasets.

The primary objective of this project is **not** to optimize model performance, but to demonstrate **software engineering rigor**, **data validation**, and **traceability** practices consistent with real-world medical AI development. Emphasis is placed on reproducibility, explicit data contracts, and clinically grounded problem definition.

---

## Clinical Background

Coronary artery calcium represents calcified atherosclerotic plaque within the coronary arteries and is a well-established marker of **coronary artery disease (CAD)**. Quantification of CAC on non-contrast cardiac CT is routinely used for:

- Cardiovascular risk stratification  
- Guiding preventive therapy (e.g., statins)  
- Long-term outcome prediction  

The most common clinical metric is the **Agatston score**, which combines the area and peak intensity (Hounsfield Units) of calcified regions. In clinical practice, CAC assessment requires careful handling of:

- Hounsfield Unit calibration  
- Slice thickness and spacing  
- Motion and noise artifacts  
- Anatomical localization of calcification to coronary arteries  

These factors make CAC detection an appropriate and challenging target for medical AI systems.

---

## Problem Definition

**Objective:**  
Develop a validated, reproducible pipeline to **detect the presence of coronary artery calcium** in non-contrast cardiac CT scans.

**Initial task formulation:**  
- Binary classification: *CAC present* vs. *CAC absent*

**Future extensibility (out of scope for initial version):**
- Agatston score regression  
- Calcification segmentation  
- Coronary-specific localization  

The project prioritizes **data integrity and correctness** over model complexity.

---

## Scope and Constraints

### In Scope
- Publicly available CT datasets
- 3D volumetric CT data ingestion
- Explicit dataset structure validation
- Reproducible preprocessing
- Clear separation of:
  - Requirements
  - Implementation
  - Verification

### Out of Scope
- Clinical deployment or diagnostic claims
- Regulatory submission
- Model performance benchmarking against proprietary datasets
- Automated clinical decision-making

This project is intended as a **technical and educational exercise** in medical AI software development.

---

## Engineering Principles

This repository is designed around the following principles:

- **Traceability:** Requirements are explicitly mapped to implementation and tests.
- **Data Contracts:** Dataset structure and assumptions are enforced via validators.
- **Reproducibility:** Environment, preprocessing, and execution are deterministic.
- **Transparency:** Limitations and assumptions are documented explicitly.

---

## Datasets

This project uses **publicly available cardiac CT datasets**.  
Specific dataset sources, preprocessing assumptions, and licensing notes are documented in:

data/README.md

Only datasets that permit research use and redistribution of derived metadata are considered.

---

## Disclaimer

This repository is **not a medical device** and is **not intended for clinical use**.  
All outputs are for research and educational purposes only.

---

## Author

**Renee Qian**  
Medical imaging and machine learning engineer

---

