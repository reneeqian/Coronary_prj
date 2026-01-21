# Internal Data Contract — Canonical CAC Volume Representation

## 1. Purpose

This document defines the canonical internal representation emitted by all dataset ingestors after ingestion and canonicalization.

This contract:

* Preserves maximal clinically relevant information
* Decouples downstream algorithms from dataset-specific formats
* Serves as the single source of truth for a patient study

Downstream components MAY derive slice-level or task-specific views from this structure.

---

## 2. Canonical Patient Sample
Each ingestor SHALL emit one **one logical sample per patient/study**.

```
CanonicalCACVolume:
  image_volume: np.ndarray        # shape (Z, Y, X), HU
  spacing_mm: Tuple[float, float, float]  # (dz, dy, dx)
  orientation: str | affine       # e.g., "RAS" or 4x4 affine
  annotations: Dict[int, List[CACAnnotation]]
  patient_id: str | int
  dataset_id: str                # e.g., "COCA"
  acquisition_type: str          # gated / nongated
  metadata: dict                 # free-form, preserved
```

## 3. Imaging Guarantees
After ingestion:

| Property | Guarantee|
|-------|-------------|
| Dimensionality | 3D |
| Shape | `(Z, Y, X)` | 
| Orientation | Canonicalized (RAS or equivalent affine) |
|Spacing | Preserved or resampled (explicitly reported) |
|Units | HU (assumed, documented, not enforced) |

## 4. Annotation Contract (Canonical)
### 4.1 Annotation Semantics
* Annotations represent **coronary artery calcium (CAC)**
* Annotations are **slice-specific**
* Zero or more annotations may exist per slice
* Annotation geometry is preserved **losslessly**

### 4.2 Annotation Structure
```python
CACAnnotation:
  slice_index: int
  contours_px: List[np.ndarray]  # each shape (N, 2)
  artery_name: Optional[str]
```
Grouped by slice index:
```python
Dict[int, List[CACAnnotation]]
```

Notes:
* Pixel coordinates are relative to the canonicalized image
* Rasterization is **optional and configurable**, not required

## 5. Validation Rules (Conceptual)
The pipeline SHALL enforce:
* Image volume is 3D
* Annotation slice indices are valid
* Contour coordinates fall within slice bounds
* No NaN/Inf image values
* Orientation consistency between image and annotations

## 6. Relationahip to Dataset-Specific Contracts
Dataset-specific contracts (e.g., COCA Gated CT Contract) define:
* Raw file formats (DICOM, XML)
* Directory layout
* Annotation encoding

Dataset ingestors transform those raw datasets into the ** Canonical CAC Volume** defined here.

**Downstream code MUST NOT depend on dataset-specific contracts.**
