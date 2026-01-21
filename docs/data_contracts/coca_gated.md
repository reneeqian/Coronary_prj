# COCA Gated CT Data Contract

## 1. Purpose

This document defines the **data contract** for ingesting the gated portion of the
COCA (Coronary Calcium CT) dataset.

The purpose of this contract is to:
- Make dataset-specific assumptions explicit
- Define a stable interface between raw data and downstream algorithms
- Enable traceability, validation, and future dataset substitution

This contract governs the **COCAGatedIngestor**.

---

## 2. Dataset Scope

This contract applies **only** to:
- Gated, non-contrast cardiac CT scans
- XML-based coronary calcium annotations
- One 3D volume per patient

Non-gated chest CT scans are explicitly **out of scope** and covered by a separate contract.

---

## 3. Directory Structure Assumptions

The ingestor assumes the following structure:

```
COCA_ROOT/
└── cocacoronarycalciumandchestcts-2/
└── Gated_release_final/
├── patient/
│ └── {patient_id}/
│ └── {series_name}/
│ └── *.dcm
└── calcium_xml/
└── {patient_id}.xml
```

- `{patient_id}` is a numeric identifier
- `{series_name}` is dataset-defined and opaque to downstream code

---

## 4. Imaging Data Contract

### 4.1 Image Volume

Each patient shall produce **one 3D image volume** with the following properties:

| Property | Requirement |
|-------|-------------|
| Data type | Numeric (integer or float) |
| Shape | `(Z, Y, X)` |
| Modality | CT |
| Contrast | Non-contrast |
| Slice order | Sorted by `InstanceNumber` |
| Units | Hounsfield Units (assumed, not enforced) |

### 4.2 Spatial Metadata

The ingestor shall extract and expose:
- Voxel spacing `(dz, dy, dx)`
- Orientation or affine (if available)
- Slice count

---

## 5. Annotation Data Contract

### 5.1 Annotation Semantics

- Annotations represent **coronary artery calcium (CAC) deposits**
- Annotations are **slice-specific**
- Zero or more ROIs may exist per slice
- ROIs are polygonal contours in pixel coordinates

### 5.2 Internal Annotation Representation

Annotations shall be standardized into the following internal structure:

```python
CACAnnotation:
  slice_index: int
  contours_px: List[np.ndarray]  # shape (N, 2)
  artery_name: Optional[str]
```
Annotations are groups by slice index:
```python
Dict[int, List[CACAnnotation]]
```
## 6 Patient Sample Contract
The ingestor shall produce one logical sample per patient:
```python
PatientSample:
  image_volume: np.ndarray
  annotations: Dict[int, List[CACAnnotation]]
  spacing: Tuple[float, float, float]
  metadata: dict
  patient_id: str | int
```
Downstream components must not depend on COCA-specific formats.

## 7 Validation Expectations
The ingestor shall fail loudly if:
* Patient directory is missing
* No DICOM files are found
* XML annotations are missing or unreadable
* Annotation slice indices exceed volume bounds

## 8. Intended Use
This contract supports:
* Research and educational workflows
* Medical AI software engineering demonstrations

This system is not a medical device and shall not be used for clinical decision-making.
