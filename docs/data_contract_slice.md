# Derived Data Contract - Canonical CAC Slice

## 1. Purpose
A convenience representation derived from `CanonicalCACVolume` for:
* Model training
* Visualization
* Slice-wise analysis
---
## 2. Structure
```python
CanonicalCACSlice:
  image: np.ndarray              # 2D (Y, X), HU
  spacing_mm: Tuple[float, float]  # (sy, sx)
  slice_index: int
  has_cac: bool
  rois: Optional[List[np.ndarray]]
  source_id: str                 # patient / study id
```
---
## 3. Derivation Rules
* `image` <-- `image_volume[slice_index]`
* `has_cac` <-- `slice_index in annotations`
* `rois` <-- flattened contours for that slice
* No additional preprocessing implied

This structure **MUST NOT** be serialized as ground truth.