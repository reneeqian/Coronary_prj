# Requirements Traceability Matrix

| Requirement ID | Title | Linked Tests | Evidence Artifacts | Status |
|----------------|-------------|--------------|--------------------|--------|
| ALG-FR-01 | Volume-Based Inference |  |  | UNTESTED |
| ALG-FR-02 | Output Shape Consistency |  |  | UNTESTED |
| ALG-NFR-01 | Reproducible Model Loading |  |  | UNTESTED |
| DEP-FR-01 | Deployment Wrapper Interface |  |  | UNTESTED |
| DEP-NFR-01 | No Hardcoded Paths |  |  | UNTESTED |
| ING-FR-01 | Dataset Root Validation |  |  | UNTESTED |
| ING-FR-02 | Deterministic Patient Enumeration |  |  | UNTESTED |
| ING-FR-03 | Graceful Failure on Missing Dataset |  |  | UNTESTED |
| ING-FR-04 | CT Slice Ordering by Spatial Position | tests/test_coca_ingestor_synthetic.py::test_slices_sorted_by_z | tests_test_coca_ingestor_synthetic.py_test_slices_sorted_by_z_20260219_125258_719369.json | PASS |
| ING-FR-05 | Hounsfield Unit Rescale Application | tests/test_coca_ingestor_synthetic.py::test_hounsfield_rescale_applied | tests_test_coca_ingestor_synthetic.py_test_hounsfield_rescale_applied_20260219_125258_721068.json | PASS |
| ING-FR-06 | Annotation Index Remapping |  |  | UNTESTED |
| SAF-FR-01 | Input Validation |  |  | UNTESTED |
| SAF-FR-02 | Failure Mode Logging |  |  | UNTESTED |
| SAF-FR-03 | Annotation Bounds Validation |  |  | UNTESTED |
| SAF-FR-04 | Mandatory DICOM Spatial Metadata Validation |  |  | UNTESTED |
| SYS-FR-01 | End-to-End Inference Pipeline |  |  | UNTESTED |
| SYS-NFR-01 | Deterministic Behavior |  |  | UNTESTED |


---
Total Requirements: 17

Tested: 2

Failures: 0
