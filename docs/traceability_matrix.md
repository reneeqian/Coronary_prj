<!-- AUTO-GENERATED FILE. DO NOT EDIT MANUALLY. -->

# Requirements Traceability Matrix

| Requirement ID | Title | Linked Tests | Evidence Artifacts | Status |
|----------------|-------------|--------------|--------------------|--------|
| DAT-001 | Dataset Validation | tests/test_coca_dataset.py::test_required_subdirectories_present | coca_dataset_structure_20260305_122753_021719.json | PASS |
| DAT-002 | Deterministic Data Access | tests/test_coca_dataset.py::test_deterministic_dataset_ordering | coca_ingestor_determinism_20260305_122753_487708.json | PASS |
| DAT-003 | Patient Index Boundary Handling |  |  | UNTESTED |
| DAT-004 | Successful Data Ingestion | tests/test_coca_ingestor_contract.py::test_ingest_ct_volumes_from_root, tests/test_coca_ingestor_synthetic.py::test_annotation_out_of_bounds_raises, tests/test_coca_ingestor_synthetic.py::test_hounsfield_rescale_applied, tests/test_coca_ingestor_synthetic.py::test_slices_sorted_by_z | coca_ingestor_contract_20260305_122753_609224.json, tests_test_coca_ingestor_synthetic.py_test_annotation_out_of_bounds_raises_20260305_122753_641093.json, tests_test_coca_ingestor_synthetic.py_test_hounsfield_rescale_applied_20260305_122753_638982.json, tests_test_coca_ingestor_synthetic.py_test_slices_sorted_by_z_20260305_122753_637273.json | PASS |
| DAT-005 | Data Ingestion Failure Modes | tests/test_coca_ingestor_contract.py::test_graceful_failure_on_missing_data | coca_ingestor_missing_data_20260305_122753_611637.json | PASS |
| DAT-006 | Lazy Patient Access |  |  | UNTESTED |
| DAT-007 | Volume and Slice Index Boundary Handling |  |  | UNTESTED |
| DAT-008 | Deterministic Slice Retrieval |  |  | UNTESTED |
| DAT-009 | Annotation Geometry Integrity |  |  | UNTESTED |
| DAT-010 | Optional Annotation Handling |  |  | UNTESTED |
| DOC-001 | Machine-Readable Requirements Definition | tests/test_project_structure.py::test_project_documentation_structure | project_documentation_structure_20260305_122753_668168.json | PASS |
| DOC-002 | Basic Project Documentation | tests/test_project_structure.py::test_project_documentation_structure | project_documentation_structure_20260305_122753_668168.json | PASS |
| SYS-001 | Deterministic System Behavior |  |  | UNTESTED |
| SYS-002 | Controlled Data Splitting |  |  | UNTESTED |
| SYS-003 | Traceable Verification |  |  | UNTESTED |
| TRN-001 | Controlled Model Initialization |  |  | UNTESTED |
| TRN-002 | Training Artifact Generation |  |  | UNTESTED |
| VER-001 | Automated Verification Execution |  |  | UNTESTED |
| VER-002 | Evidence Capture |  |  | UNTESTED |


---
Total Requirements: 19

Tested: 6

Failures: 0
