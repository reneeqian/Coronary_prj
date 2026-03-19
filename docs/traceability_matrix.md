<!-- AUTO-GENERATED FILE. DO NOT EDIT MANUALLY. -->

# Requirements Traceability Matrix

## Requirement Coverage

**Coverage:** 17.1% (7 / 41 requirements tested)

## Code Coverage

**Line Coverage:** 92.9%

Detailed uncovered lines saved in `artifacts/coverage/uncovered_lines.txt`

| Requirement ID | Title | Linked Tests | Evidence Artifacts | Status |
|----------------|-------------|--------------|--------------------|--------|
| DAT-001 | Dataset Validation | tests/test_coca_dataset.py::test_required_subdirectories_present, tests/test_coca_ingestor_defensive_paths.py::test_missing_patient_root, tests/test_coca_ingestor_edge_cases.py::test_no_patient_directories, tests/test_coca_ingestor_synthetic.py::test_dataset_structure_validation |  | PASS |
| DAT-002 | Deterministic Data Access | tests/test_coca_dataset.py::test_deterministic_dataset_ordering, tests/test_coca_ingestor_edge_cases.py::test_slice_determinism |  | PASS |
| DAT-003 | Patient Index Boundary Handling | tests/test_coca_ingestor_synthetic.py::test_invalid_patient_id_raises |  | UNTESTED |
| DAT-004 | Successful Patient Ingestion | tests/test_coca_dataset.py::test_get_sample_on_real_dataset, tests/test_coca_ingestor_contract.py::test_ingest_ct_volumes_from_root, tests/test_coca_ingestor_contract.py::test_patient_sample_contract, tests/test_coca_ingestor_edge_cases.py::test_get_slice_success, tests/test_coca_ingestor_edge_cases.py::test_missing_image_position_patient, tests/test_coca_ingestor_synthetic.py::test_annotation_out_of_bounds_raises, tests/test_coca_ingestor_synthetic.py::test_get_sample_generates_image_and_mask, tests/test_coca_ingestor_synthetic.py::test_get_sample_multiple_rois_same_slice, tests/test_coca_ingestor_synthetic.py::test_get_sample_no_annotations_returns_empty, tests/test_coca_ingestor_synthetic.py::test_hounsfield_rescale_applied, tests/test_coca_ingestor_synthetic.py::test_ingest_dataset_multiple_patients, tests/test_coca_ingestor_synthetic.py::test_slices_sorted_by_z |  | PASS |
| DAT-005 | Data Ingestion Failure Modes | tests/test_coca_ingestor_contract.py::test_graceful_failure_on_missing_data, tests/test_coca_ingestor_defensive_paths.py::test_annotation_xml_parse_failure, tests/test_coca_ingestor_defensive_paths.py::test_invalid_dicom_file, tests/test_coca_ingestor_defensive_paths.py::test_missing_image_position_patient, tests/test_coca_ingestor_defensive_paths.py::test_patient_without_series, tests/test_coca_ingestor_defensive_paths.py::test_series_without_dicoms, tests/test_coca_ingestor_edge_cases.py::test_no_dicom_files, tests/test_coca_ingestor_edge_cases.py::test_no_series_directories, tests/test_coca_ingestor_synthetic.py::test_missing_dicom_files |  | PASS |
| DAT-006 | Lazy Patient Access | tests/test_coca_dataset.py::test_get_sample_on_real_dataset, tests/test_coca_ingestor_synthetic.py::test_get_patient_api, tests/test_coca_ingestor_synthetic.py::test_get_sample_generates_image_and_mask, tests/test_coca_ingestor_synthetic.py::test_get_volume_api, tests/test_coca_ingestor_synthetic.py::test_lazy_patient_loading |  | PASS |
| DAT-007 | Dataset Element Boundary Handling | tests/test_coca_dataset.py::test_slice_index_out_of_bounds, tests/test_coca_ingestor_edge_cases.py::test_annotation_slice_out_of_bounds, tests/test_coca_ingestor_synthetic.py::test_get_sample_skips_invalid_slice_annotations, tests/test_coca_ingestor_synthetic.py::test_slice_index_out_of_bounds |  | UNTESTED |
| DAT-008 | Deterministic Patient Sample Retrieval | tests/test_coca_ingestor_synthetic.py::test_deterministic_slice_retrieval |  | UNTESTED |
| DAT-009 | Annotation Geometry Integrity | tests/test_annotation_geometry_integrity.py::test_invalid_polygon_skipped, tests/test_annotation_geometry_integrity.py::test_valid_annotation_geometry, tests/test_coca_ingestor_edge_cases.py::test_roi_with_insufficient_points_ignored, tests/test_coca_ingestor_synthetic.py::test_annotation_with_missing_image_index |  | UNTESTED |
| DAT-010 | Optional Annotation Handling | tests/test_annotation_geometry_integrity.py::test_missing_annotation_file_returns_empty, tests/test_coca_ingestor_edge_cases.py::test_dataset_without_annotations |  | UNTESTED |
| DAT-011 | Dataset Partition Generation |  |  | UNTESTED |
| DAT-012 | CT slices must be sorted by ImagePositionPatient[2] before volume construction | tests/test_coca_ingestor_synthetic.py::test_ct_volume_sorted_by_z_position |  | UNTESTED |
| DAT-013 | Annotation Rasterization |  |  | UNTESTED |
| DOC-001 | Machine-Readable Requirements Definition | tests/test_project_structure.py::test_project_documentation_structure |  | PASS |
| DOC-002 | Basic Project Documentation | tests/test_project_structure.py::test_project_documentation_structure |  | PASS |
| DOC-003 | Traceability Documentation |  |  | UNTESTED |
| INF-001 | Inference Capability |  |  | UNTESTED |
| INF-002 | Inference Determinism |  |  | UNTESTED |
| MOD-001 | Model Artifact Generation |  |  | UNTESTED |
| MOD-002 | Model Artifact Persistence |  |  | UNTESTED |
| MOD-003 | Model Evaluation Capability |  |  | UNTESTED |
| REP-001 | Training Report Generation |  |  | UNTESTED |
| REP-002 | Visualization Support |  |  | UNTESTED |
| SYS-001 | Deterministic System Behavior | tests/test_annotation_geometry_integrity.py::test_volume_sorted_by_z_position |  | UNTESTED |
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
| TSK-001 | Task Definition Interface |  |  | UNTESTED |
| TSK-002 | Coronary Calcium Detection Task |  |  | UNTESTED |
| TSK-003 | Task Determinism |  |  | UNTESTED |
| TSK-004 | Calcium Thresholding Support |  |  | UNTESTED |
| VER-001 | Automated Verification Execution |  |  | UNTESTED |
| VER-002 | Evidence Capture |  |  | UNTESTED |
| VER-003 | Model Performance Verification |  |  | UNTESTED |


---

## Untested Requirements

- DAT-003
- DAT-007
- DAT-008
- DAT-009
- DAT-010
- DAT-011
- DAT-012
- DAT-013
- DOC-003
- INF-001
- INF-002
- MOD-001
- MOD-002
- MOD-003
- REP-001
- REP-002
- SYS-001
- SYS-002
- SYS-003
- SYS-004
- SYS-005
- SYS-006
- TRN-001
- TRN-002
- TRN-003
- TRN-004
- TRN-005
- TSK-001
- TSK-002
- TSK-003
- TSK-004
- VER-001
- VER-002
- VER-003


---
Total Requirements: 41

Tested: 7

Failures: 0
