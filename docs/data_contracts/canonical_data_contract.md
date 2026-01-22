# Canonical Data Contract — Coronary CT with Calcium Annotations

## 1. Purpose

This document defines the **canonical internal data representation** used by the coronary imaging pipeline **after dataset ingestion**.

This contract:
- Decouples modeling and analysis code from dataset-specific formats
- Defines guarantees provided by dataset ingestors
- Preserves raw annotation geometry and semantics
- Enables flexible downstream use (classification, segmentation, scoring, visualization)

All dataset-specific ingestors (e.g., COCA gated, COCA non-gated, future datasets) MUST adapt their raw inputs into this canonical representation.

---

## 2. Canonical Patient Sample

Each patient (or study) is represented as a single logical sample:

```python
CanonicalPatientSample:
  image_volume: np.ndarray
  spacing_mm: Tuple[float, float, float]     # (dz, dy, dx)
  orientation: Optional[str | np.ndarray]    # e.g. RAS or affine
  annotations: AnnotationBundle
  metadata: Dict[str, Any]
  patient_id: str
```

## 3. Image Volume Contract
### 3.1 Image Volume
| Property | Requirement | 
| -------- | ----------- | 
| Data type | Numeric (integer or float) |
| Shape | `(Z, Y, X)` |
| Modality | CT |
| Contrast | Non-constrast |
| Slice order | Anatomically consistent (e.g. sorted by `InstanceNumber`) |
| Units | Hounsfield Units (assumed, not enforced) |

### 3.2 Spatial Metadata
* Voxel spacing is explicitly provided as `(dz, dy, dx)` in millimeters
* Orientation or affine is preserved if available
* Volumes may be resampled by downstream preprocessing, but ingestion must not discard spatial information

## 4. Canonical Annotation Bundle
Annotations are grouped into a unified container that supports **multiple annotation modalities simultaneously**.

```python
AnnotationBundle:
  vector_rois: Optional[VectorROISet]
  segmentation_masks: Optional[SegmentationMaskSet]
  label_map: Optional[Dict[int, str]]
```

Any or all fields may be `None`, depending on dataset availability.

## 5. Vector ROI Annotations
### 5.1 Semantics
* Vector ROIs represent **coronary artery calcium (CAC)** or related structures
* Annotations may be slice-specific or volume-wide
* Geometry is preserved **losslessly**

### 5.2 Representation
```python
VectorROI:
  slice_index: int
  contour_px: np.ndarray        # shape (N, 2)
  label: str                    # e.g. "CAC", "LAD", "RCA"
  metadata: Optional[dict]
```
Grouped as:
```python
VectorROISet = Dict[int, List[VectorROI]]
```

## 6. Segmentation Mask Annotations
### 6.1 Semantics
* Masks represent voxel-wise annotations
* Supports:
  * Binary masks
  * Multi-class masks
* Masks may encode CAC only or multiple anatomical/pathological structures

### 6.2 Representation
```python
SegmentationMask:
  mask: np.ndarray              # shape (Z, Y, X)
  label_map: Dict[int, str]     # e.g. {0: "background", 1: "CAC"}
```
Multiple masks may exist per sample:
```python
SegmentationMaskSet = List[SegmentationMask]
```

## 7. Annotation Flexiility Guarantees
The canonical representation guarantees:
* Vector ROIs are preserved in original pixel coordinates
* Segmentation masks preserve voxel alignment with the image volume
* Multiple annotation types may coexist
* No annotation information is discarded during ingestion

Rasterization, merging, binarization, or label collapsing are **explicit downstream operations**, not ingestion responsibilities.

## 8. Metadata Contract
The `metadata` field may include:
* Study/series identifiers
* Acquisition parameters
* Gating information
* Scanner manufacturer
* Dataset source identifiers
* Ingestor version

Metadata is **opaque** to downstream modeling unless explicity required.

## 9. Validation Rules (Conceptual)
Canonical samples must satisfy:
* `image_volume.ndim == 3`
* `spacing_mm` has length 3 and positive values
* Vector ROIs reference valid slice indices
* Segmantation masks mask image volume shape
* Label maps contain valid, unique mappings

Concrete enforcement is implemented in validators.

## 10. Relationship to Dataset-Specific Contracts
Dataset0speicifc contracts (e.g., COCA Gated CT) define:
* Raw file formats (DICOM, XML, NIfTI)
* Dataset organization
* Annotation encodings

Dataset ingestors transform raw datasets into the canonical representation defined here.

Downstream code MUST depend only on this canonical contract, never on dataset-specific contracts.