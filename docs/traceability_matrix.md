<!-- AUTO-GENERATED FILE. DO NOT EDIT MANUALLY. -->

# Requirements Traceability Matrix

| Requirement ID | Title | Linked Tests | Evidence Artifacts | Status |
|----------------|-------------|--------------|--------------------|--------|
| DAT-001 | Dataset Validation | tests/test_coca_dataset.py::test_required_subdirectories_present, tests/test_coca_ingestor_defensive_paths.py::test_missing_patient_root, tests/test_coca_ingestor_edge_cases.py::test_no_patient_directories, tests/test_coca_ingestor_synthetic.py::test_dataset_structure_validation | coca_dataset_structure_20260312_110701_042778.json | PASS |
| DAT-002 | Deterministic Data Access | tests/test_coca_dataset.py::test_deterministic_dataset_ordering, tests/test_coca_ingestor_edge_cases.py::test_slice_determinism | coca_ingestor_determinism_20260312_110702_192264.json | PASS |
| DAT-003 | Patient Index Boundary Handling | tests/test_coca_ingestor_synthetic.py::test_invalid_patient_id_raises |  | LINKED |
| DAT-004 | Successful Patient Ingestion | tests/test_coca_dataset.py::test_get_sample_on_real_dataset, tests/test_coca_ingestor_contract.py::test_ingest_ct_volumes_from_root, tests/test_coca_ingestor_contract.py::test_patient_sample_contract, tests/test_coca_ingestor_edge_cases.py::test_get_slice_success, tests/test_coca_ingestor_edge_cases.py::test_missing_image_position_patient, tests/test_coca_ingestor_synthetic.py::test_annotation_out_of_bounds_raises, tests/test_coca_ingestor_synthetic.py::test_get_sample_generates_image_and_mask, tests/test_coca_ingestor_synthetic.py::test_get_sample_multiple_rois_same_slice, tests/test_coca_ingestor_synthetic.py::test_get_sample_no_annotations_returns_empty, tests/test_coca_ingestor_synthetic.py::test_hounsfield_rescale_applied, tests/test_coca_ingestor_synthetic.py::test_ingest_dataset_multiple_patients, tests/test_coca_ingestor_synthetic.py::test_slices_sorted_by_z | coca_ingestor_contract_20260312_110702_622076.json, tests_test_coca_ingestor_synthetic.py_test_annotation_out_of_bounds_raises_20260312_110702_694711.json, tests_test_coca_ingestor_synthetic.py_test_hounsfield_rescale_applied_20260312_110702_691606.json, tests_test_coca_ingestor_synthetic.py_test_slices_sorted_by_z_20260312_110702_688532.json | PASS |
| DAT-005 | Data Ingestion Failure Modes | tests/test_coca_ingestor_contract.py::test_graceful_failure_on_missing_data, tests/test_coca_ingestor_defensive_paths.py::test_annotation_xml_parse_failure, tests/test_coca_ingestor_defensive_paths.py::test_invalid_dicom_file, tests/test_coca_ingestor_defensive_paths.py::test_missing_image_position_patient, tests/test_coca_ingestor_defensive_paths.py::test_patient_without_series, tests/test_coca_ingestor_defensive_paths.py::test_series_without_dicoms, tests/test_coca_ingestor_edge_cases.py::test_no_dicom_files, tests/test_coca_ingestor_edge_cases.py::test_no_series_directories, tests/test_coca_ingestor_synthetic.py::test_missing_dicom_files | coca_ingestor_missing_data_20260312_110702_627547.json | PASS |
| DAT-006 | Lazy Patient Access | tests/test_coca_dataset.py::test_get_sample_on_real_dataset, tests/test_coca_ingestor_synthetic.py::test_get_patient_api, tests/test_coca_ingestor_synthetic.py::test_get_sample_generates_image_and_mask, tests/test_coca_ingestor_synthetic.py::test_get_volume_api, tests/test_coca_ingestor_synthetic.py::test_lazy_patient_loading | coca_get_sample_validation_20260312_110702_396118.json | PASS |
| DAT-007 | Dataset Element Boundary Handling | tests/test_coca_dataset.py::test_slice_index_out_of_bounds, tests/test_coca_ingestor_edge_cases.py::test_annotation_slice_out_of_bounds, tests/test_coca_ingestor_synthetic.py::test_get_sample_skips_invalid_slice_annotations, tests/test_coca_ingestor_synthetic.py::test_slice_index_out_of_bounds |  | LINKED |
| DAT-008 | Deterministic Patient Sample Retrieval | tests/test_coca_ingestor_synthetic.py::test_deterministic_slice_retrieval |  | LINKED |
| DAT-009 | Annotation Geometry Integrity | tests/test_annotation_geometry_integrity.py::test_invalid_polygon_skipped, tests/test_annotation_geometry_integrity.py::test_valid_annotation_geometry, tests/test_coca_ingestor_edge_cases.py::test_roi_with_insufficient_points_ignored, tests/test_coca_ingestor_synthetic.py::test_annotation_with_missing_image_index |  | LINKED |
| DAT-010 | Optional Annotation Handling | tests/test_annotation_geometry_integrity.py::test_missing_annotation_file_returns_empty, tests/test_coca_ingestor_edge_cases.py::test_dataset_without_annotations |  | LINKED |
| DAT-011 | Dataset Partition Generation |  |  | UNTESTED |
| DAT-012 | CT slices must be sorted by ImagePositionPatient[2] before volume construction | tests/test_coca_ingestor_synthetic.py::test_ct_volume_sorted_by_z_position |  | LINKED |
| DAT-013 | Annotation Rasterization |  |  | UNTESTED |
| DOC-001 | Machine-Readable Requirements Definition | tests/test_project_structure.py::test_project_documentation_structure | project_documentation_structure_20260312_110702_746813.json | PASS |
| DOC-002 | Basic Project Documentation | tests/test_project_structure.py::test_project_documentation_structure | project_documentation_structure_20260312_110702_746813.json | PASS |
| DOC-003 | Traceability Documentation |  |  | UNTESTED |
| MOD-001 | Model Artifact Generation |  |  | UNTESTED |
| MOD-002 | Model Artifact Persistence |  |  | UNTESTED |
| MOD-003 | Model Evaluation Capability |  |  | UNTESTED |
| SYS-001 | Deterministic System Behavior | tests/test_annotation_geometry_integrity.py::test_volume_sorted_by_z_position |  | LINKED |
| SYS-002 | Controlled Data Splitting |  |  | UNTESTED |
| SYS-003 | Traceable Verification |  |  | UNTESTED |
| SYS-004 | Coronary Model Development |  |  | UNTESTED |
| SYS-005 | Model Improvement Capability |  |  | UNTESTED |
| SYS-006 | Dataset Task Encapsulation |  |  | UNTESTED |
| TRN-001 | Controlled Model Initialization |  |  | UNTESTED |
| TRN-002 | Training Artifact Generation |  |  | UNTESTED |
| TRN-003 | Coronary Model Training |  |  | UNTESTED |
| TRN-004 | Coronary Model Retraining |  |  | UNTESTED |
| TRN-005 | Coronary Dataset Training Interface |  |  | UNTESTED |
| VER-001 | Automated Verification Execution |  |  | UNTESTED |
| VER-002 | Evidence Capture |  |  | UNTESTED |
| VER-003 | Model Performance Verification |  |  | UNTESTED |


---
Total Requirements: 33

Tested: 14

Failures: 0
