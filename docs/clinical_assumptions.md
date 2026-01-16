# Clinical Assumptions and Context

## 1. Purpose

This document enumerates the clinical assumptions underlying the Coronary Artery Calcium (CAC) detection pipeline.

The intent is to explicitly document the clinical context, constraints, and limitations assumed by the software system. This transparency supports reproducibility, traceability, and responsible interpretation of results.

This system is intended for **research and educational use only** and is not a diagnostic or clinical decision-support tool.

---

## 2. Clinical Background

Coronary artery calcium is a marker of atherosclerotic plaque burden and is commonly assessed using non-contrast cardiac CT imaging. CAC burden has established associations with cardiovascular risk and is frequently quantified using Agatston scoring in clinical practice.

This project focuses on software infrastructure to support CAC-related research workflows, not on replicating or validating clinical scoring methodologies.

---

## 3. Imaging Modality Assumptions

### CA-1: CT Imaging Type

- Input images are assumed to be **non-contrast cardiac CT** volumes.
- Contrast-enhanced CT angiography (CTA) is explicitly out of scope.

### CA-2: Image Orientation and Coverage

- CT volumes are assumed to include sufficient cardiac coverage to contain coronary arteries.
- No assumptions are made about exact slice thickness, in-plane resolution, or scanner manufacturer.

---

## 4. Data and Annotation Assumptions

### CA-3: De-identification

- All input datasets are assumed to be fully de-identified.
- No protected health information (PHI) is expected or supported.

### CA-4: Label Provenance

- Ground truth labels (if present) are assumed to be generated externally.
- The system does not validate clinical correctness of annotations.

### CA-5: Annotation Scope

- Labels may represent calcium presence, location, or segmentation, depending on dataset availability.
- The system does not assume coronary-level or vessel-level labeling.

---

## 5. Physiological and Imaging Assumptions

### CA-6: Calcium Definition

- Coronary calcium is assumed to correspond to **high-attenuation voxels** consistent with calcium on CT.
- A commonly used clinical threshold (e.g., ≥130 HU) is referenced for contextual understanding but is not enforced as a diagnostic criterion.

### CA-7: Non-coronary Calcifications

- The system does not distinguish coronary calcifications from other high-density
