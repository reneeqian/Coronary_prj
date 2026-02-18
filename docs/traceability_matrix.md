# Requirements Traceability Matrix

| Requirement ID | Description | Linked Tests | Evidence Artifacts | Status |
|----------------|-------------|--------------|--------------------|--------|
| ALG-FR-01 | The algorithm shall accept a 3D CT volume and output CAC predictions. |  |  | UNTESTED |
| ALG-FR-02 | The output prediction shall maintain consistent spatial dimensions relative to input volume. |  |  | UNTESTED |
| ALG-NFR-01 | Model weights shall load deterministically without modification to inference logic. |  |  | UNTESTED |
| DEP-FR-01 | The system shall expose a callable inference interface for integration into external workflows. |  |  | UNTESTED |
| DEP-NFR-01 | The deployment module shall not rely on hardcoded file paths. |  |  | UNTESTED |
| ING-FR-01 | The ingestion module shall validate dataset directory structure before attempting ingestion. |  |  | UNTESTED |
| ING-FR-02 | The ingestion module shall enumerate patient IDs in a deterministic order. |  |  | UNTESTED |
| ING-FR-03 | The ingestion module shall raise a controlled exception if dataset root is missing. |  |  | UNTESTED |
| ING-FR-04 | The system shall sort DICOM slices by the Z component of ImagePositionPatient prior to volume construction to ensure anatomically correct volumetric reconstruction. | tests/test_coca_ingestor_synthetic.py::test_slices_sorted_by_z | tests_test_coca_ingestor_synthetic.py_test_slices_sorted_by_z_20260218_113111_958018.json | PASS |
| ING-FR-05 | The system shall apply RescaleSlope and RescaleIntercept to raw DICOM pixel data to produce clinically accurate Hounsfield Unit (HU) values. | tests/test_coca_ingestor_synthetic.py::test_hounsfield_rescale_applied | tests_test_coca_ingestor_synthetic.py_test_hounsfield_rescale_applied_20260218_113111_959641.json | PASS |
| ING-FR-06 | The system shall convert 1-based external annotation slice indices into 0-based internal indices and associate annotations with the correct reconstructed slice. |  |  | UNTESTED |
| SAF-FR-01 | The system shall validate input tensor dimensions and spacing. |  |  | UNTESTED |
| SAF-FR-02 | Critical failures shall produce structured error messages. |  |  | UNTESTED |
| SAF-FR-03 | The system shall reject annotations that reference slice indices outside the reconstructed volume bounds and raise a DatasetStructureError. |  |  | UNTESTED |
| SAF-FR-04 | The system shall validate presence of ImagePositionPatient and required spacing metadata and raise DatasetStructureError if required metadata is missing or invalid. |  |  | UNTESTED |
| SYS-FR-01 | The system shall accept a CT study directory and produce a deterministic CAC detection output. |  |  | UNTESTED |
| SYS-NFR-01 | Given identical inputs and model weights, the system shall produce identical outputs. |  |  | UNTESTED |


---
Total Requirements: 17

Tested: 2

Failures: 0
