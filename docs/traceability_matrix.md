<!-- AUTO-GENERATED FILE. DO NOT EDIT MANUALLY. -->

# Requirements Traceability Matrix

| Requirement ID | Title | Linked Tests | Evidence Artifacts | Status |
|----------------|-------------|--------------|--------------------|--------|
| DAT-001 | Dataset Validation | tests/test_coca_dataset.py::test_required_subdirectories_present, tests/test_coca_ingestor_defensive_paths.py::test_missing_patient_root, tests/test_coca_ingestor_edge_cases.py::test_no_patient_directories, tests/test_coca_ingestor_synthetic.py::test_dataset_structure_validation | coca_dataset_structure_20260305_143226_113075.json | PASS |
| DAT-002 | Deterministic Data Access | tests/test_coca_dataset.py::test_deterministic_dataset_ordering, tests/test_coca_ingestor_edge_cases.py::test_slice_determinism | coca_ingestor_determinism_20260305_143226_604974.json | PASS |
| DAT-003 | Patient Index Boundary Handling | tests/test_coca_ingestor_synthetic.py::test_invalid_patient_id_raises |  | LINKED |
| DAT-004 | Successful Data Ingestion | tests/test_coca_ingestor_contract.py::test_ingest_ct_volumes_from_root, tests/test_coca_ingestor_edge_cases.py::test_get_slice_success, tests/test_coca_ingestor_edge_cases.py::test_missing_image_position_patient, tests/test_coca_ingestor_synthetic.py::test_annotation_out_of_bounds_raises, tests/test_coca_ingestor_synthetic.py::test_hounsfield_rescale_applied, tests/test_coca_ingestor_synthetic.py::test_ingest_dataset_multiple_patients, tests/test_coca_ingestor_synthetic.py::test_slices_sorted_by_z | coca_ingestor_contract_20260305_143226_769306.json, tests_test_coca_ingestor_synthetic.py_test_annotation_out_of_bounds_raises_20260305_143226_800183.json, tests_test_coca_ingestor_synthetic.py_test_hounsfield_rescale_applied_20260305_143226_797974.json, tests_test_coca_ingestor_synthetic.py_test_slices_sorted_by_z_20260305_143226_796099.json | PASS |
| DAT-005 | Data Ingestion Failure Modes | tests/test_coca_ingestor_contract.py::test_graceful_failure_on_missing_data, tests/test_coca_ingestor_defensive_paths.py::test_annotation_xml_parse_failure, tests/test_coca_ingestor_defensive_paths.py::test_invalid_dicom_file, tests/test_coca_ingestor_defensive_paths.py::test_missing_image_position_patient, tests/test_coca_ingestor_defensive_paths.py::test_patient_without_series, tests/test_coca_ingestor_defensive_paths.py::test_series_without_dicoms, tests/test_coca_ingestor_edge_cases.py::test_no_dicom_files, tests/test_coca_ingestor_edge_cases.py::test_no_series_directories, tests/test_coca_ingestor_synthetic.py::test_missing_dicom_files | coca_ingestor_missing_data_20260305_143226_772327.json | PASS |
| DAT-006 | Lazy Patient Access | tests/test_coca_ingestor_synthetic.py::test_get_patient_api, tests/test_coca_ingestor_synthetic.py::test_get_volume_api, tests/test_coca_ingestor_synthetic.py::test_lazy_patient_loading |  | LINKED |
| DAT-007 | Volume and Slice Index Boundary Handling | tests/test_coca_dataset.py::test_slice_index_out_of_bounds, tests/test_coca_ingestor_edge_cases.py::test_annotation_slice_out_of_bounds, tests/test_coca_ingestor_synthetic.py::test_slice_index_out_of_bounds |  | LINKED |
| DAT-008 | Deterministic Slice Retrieval | tests/test_coca_ingestor_synthetic.py::test_deterministic_slice_retrieval |  | LINKED |
| DAT-009 | Annotation Geometry Integrity | tests/test_annotation_geometry_integrity.py::test_invalid_polygon_skipped, tests/test_annotation_geometry_integrity.py::test_valid_annotation_geometry, tests/test_coca_ingestor_edge_cases.py::test_roi_with_insufficient_points_ignored, tests/test_coca_ingestor_synthetic.py::test_annotation_with_missing_image_index |  | LINKED |
| DAT-010 | Optional Annotation Handling | tests/test_annotation_geometry_integrity.py::test_missing_annotation_file_returns_empty, tests/test_coca_ingestor_edge_cases.py::test_dataset_without_annotations |  | LINKED |
| DOC-001 | Machine-Readable Requirements Definition | tests/test_project_structure.py::test_project_documentation_structure | project_documentation_structure_20260305_143226_827020.json | PASS |
| DOC-002 | Basic Project Documentation | tests/test_project_structure.py::test_project_documentation_structure | project_documentation_structure_20260305_143226_827020.json | PASS |
| SYS-001 | Deterministic System Behavior | tests/test_annotation_geometry_integrity.py::test_volume_sorted_by_z_position |  | LINKED |
| SYS-002 | Controlled Data Splitting |  |  | UNTESTED |
| SYS-003 | Traceable Verification |  |  | UNTESTED |
| TRN-001 | Controlled Model Initialization |  |  | UNTESTED |
| TRN-002 | Training Artifact Generation |  |  | UNTESTED |
| VER-001 | Automated Verification Execution |  |  | UNTESTED |
| VER-002 | Evidence Capture |  |  | UNTESTED |


---
Total Requirements: 19

Tested: 13

Failures: 0
