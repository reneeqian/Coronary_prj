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
dataset_root/
├── patient/
│   ├── 0/
│   │   └── Pro_Gated_CS_3.0_I30f_3_70%/
│   │       ├── *.dcm
│   ├── 1/
│   │   └── Pro_Gated_CS_3.0_I30f_3_70%/
│   │       ├── *.dcm
├── calcium_xml/
│   ├── 0.xml
│   ├── 1.xml

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

### 5.2 Canonical Annotation Mapping

COCA gated annotations are mapped into the canonical `VectorROI` representation.

For each annotated slice:
- One or more polygonal ROIs are extracted
- Each ROI is represented as a `VectorROI` with:
  - `slice_index` taken from `ImageIndex`
  - `contour_px` parsed from `Point_px`
  - `label` set to `"CAC"`
  - `metadata["artery"]` populated from the XML `Name` field (if present)

Vector ROIs are grouped by slice index and populated into
`AnnotationBundle.vector_rois`.

No segmentation masks are generated during ingestion.


### Annotation Representation

COCA gated annotations are provided as vector-based ROIs in XML format.

The COCA gated ingestor shall:
- Parse XML-based polygon annotations
- Preserve slice index and pixel coordinates
- Populate the `vector_rois` field of the canonical `AnnotationBundle`
- Leave `masks` empty

Rasterized masks are not provided by the dataset and are not generated
during ingestion.

## 6. Patient Sample Contract

The ingestor shall produce one canonical `PatientSample` as defined in
`canonical_data_contract.md`.

Dataset-specific details (directory structure, XML formats, series naming)
must not propagate beyond ingestion.

## 7. Known Dataset Limitations

- Variable voxel spacing across scans
- Inconsistent scan coverage of the heart
- Potential label noise and inter-annotator variability
- Mixed gated and non-gated acquisitions

These limitations are explicitly tolerated and handled through validation logic.

## 8. Validation Expectations
The ingestor shall fail loudly if:
* Patient directory is missing
* No DICOM files are found
* XML annotations are missing or unreadable
* Annotation slice indices exceed volume bounds

## 9. Intended Use
This contract supports:
* Research and educational workflows
* Medical AI software engineering demonstrations

This system is not a medical device and shall not be used for clinical decision-making.
