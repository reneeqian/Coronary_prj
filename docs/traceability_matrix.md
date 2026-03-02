<!-- AUTO-GENERATED FILE. DO NOT EDIT MANUALLY. -->

# Requirements Traceability Matrix

| Requirement ID | Title | Linked Tests | Evidence Artifacts | Status |
|----------------|-------------|--------------|--------------------|--------|
| DAT-001 | Dataset Validation |  |  | UNTESTED |
| DAT-002 | Deterministic Data Access |  |  | UNTESTED |
| DAT-003 | Boundary Condition Handling |  |  | UNTESTED |
| DAT-004 | Successful Data Ingestion | tests/test_coca_ingestor_synthetic.py::test_annotation_out_of_bounds_raises, tests/test_coca_ingestor_synthetic.py::test_hounsfield_rescale_applied, tests/test_coca_ingestor_synthetic.py::test_slices_sorted_by_z | tests_test_coca_ingestor_synthetic.py_test_annotation_out_of_bounds_raises_20260302_133824_544979.json, tests_test_coca_ingestor_synthetic.py_test_hounsfield_rescale_applied_20260302_133824_543613.json, tests_test_coca_ingestor_synthetic.py_test_slices_sorted_by_z_20260302_133824_541961.json | PASS |
| DAT-005 | Data Ingestion Failure Modes |  |  | UNTESTED |
| DOC-001 | Machine-Readable Requirements Definition | tests/test_project_structure.py::test_required_project_documentation_exists | project_documentation_presence_20260302_133824_545617.json | PASS |
| DOC-002 | Basic Project Documentation | tests/test_project_structure.py::test_required_project_documentation_exists | project_documentation_presence_20260302_133824_545617.json | PASS |
| SYS-001 | Deterministic System Behavior |  |  | UNTESTED |
| SYS-002 | Controlled Data Splitting |  |  | UNTESTED |
| SYS-003 | Traceable Verification |  |  | UNTESTED |
| TRN-001 | Controlled Model Initialization |  |  | UNTESTED |
| TRN-002 | Training Artifact Generation |  |  | UNTESTED |
| VER-001 | Automated Verification Execution |  |  | UNTESTED |
| VER-002 | Evidence Capture |  |  | UNTESTED |


---
Total Requirements: 14

Tested: 3

Failures: 0
