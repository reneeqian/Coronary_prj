<!-- AUTO-GENERATED FILE. DO NOT EDIT MANUALLY. -->

# Requirements Traceability Matrix

## Requirement Coverage

**Coverage:** 100.0% (48 / 48 requirements tested)

## Code Coverage

**Line Coverage:** 92.2%

Detailed uncovered lines saved in `artifacts/coverage/uncovered_lines.txt`

## Forge Code Health

**Overall Score:** 95.2%  **Grade:** A

*Generated at 2026-05-04T02:24:14.695446+00:00*

| Collector | Score | Status |
|-----------|-------|--------|
| Test Metrics | 92.2% | ok |
| Complexity | — | radon not found — install it with: pip install radon |
| Dependency Health | — | Could not parse pip-audit output |
| Requirements Coverage | 100.0% | ok |
| Static Analysis | 92.3% | ok |
| Type Coverage | 100.0% | ok |
| Dead Code | 100.0% | ok |
| Mutation Testing | — | Disabled by default; set [collectors.mutation_testing] enabled = true in forge.toml |

| Requirement ID | Title | Linked Tests | Evidence Artifacts | Status |
|----------------|-------------|--------------|--------------------|--------|
| DAT-001 | Dataset Validation | tests/test_coca_dataset.py::test_required_subdirectories_present, tests/test_coca_ingestor_defensive_paths.py::test_missing_patient_root, tests/test_coca_ingestor_edge_cases.py::test_no_patient_directories, tests/test_coca_ingestor_synthetic.py::test_dataset_structure_validation | DAT001_missing_patient_root_20260503_192410_250014.json, DAT001_no_patient_directories_20260503_192410_272970.json, coca_dataset_structure_20260503_192409_536664.json | PASS |
| DAT-002 | Deterministic Data Access | tests/test_coca_dataset.py::test_deterministic_dataset_ordering, tests/test_coca_ingestor_edge_cases.py::test_slice_determinism, tests/test_coca_nongated_ingestor.py::test_list_patient_ids_sorted_numerically | coca_ingestor_determinism_20260503_192409_966258.json | PASS |
| DAT-003 | Patient Index Boundary Handling | tests/test_coca_ingestor_synthetic.py::test_invalid_patient_id_raises |  | LINKED |
| DAT-004 | Successful Patient Ingestion | tests/test_coca_dataset.py::test_get_sample_on_real_dataset, tests/test_coca_ingestor_contract.py::test_ingest_ct_volumes_from_root, tests/test_coca_ingestor_contract.py::test_patient_sample_contract, tests/test_coca_ingestor_edge_cases.py::test_get_slice_success, tests/test_coca_ingestor_edge_cases.py::test_missing_image_position_patient, tests/test_coca_ingestor_synthetic.py::test_annotation_out_of_bounds_raises, tests/test_coca_ingestor_synthetic.py::test_get_sample_generates_image_and_mask, tests/test_coca_ingestor_synthetic.py::test_get_sample_multiple_rois_same_slice, tests/test_coca_ingestor_synthetic.py::test_get_sample_no_annotations_returns_empty, tests/test_coca_ingestor_synthetic.py::test_hounsfield_rescale_applied, tests/test_coca_ingestor_synthetic.py::test_ingest_dataset_multiple_patients, tests/test_coca_ingestor_synthetic.py::test_slices_sorted_by_z, tests/test_coca_nongated_ingestor.py::test_annotations_always_none, tests/test_coca_nongated_ingestor.py::test_hounsfield_rescale_applied, tests/test_coca_nongated_ingestor.py::test_slices_z_sorted_on_load, tests/test_coronary_calcium_task.py::test_coca_gated_ingestor_skips_dicom_without_image_positionpatient | DAT004_missing_image_position_20260503_192410_266937.json, coca_ingestor_contract_20260503_192410_244139.json, tests_test_coca_ingestor_synthetic.py_test_annotation_out_of_bounds_raises_20260503_192410_282883.json, tests_test_coca_ingestor_synthetic.py_test_hounsfield_rescale_applied_20260503_192410_280675.json, tests_test_coca_ingestor_synthetic.py_test_slices_sorted_by_z_20260503_192410_278427.json | PASS |
| DAT-005 | Data Ingestion Failure Modes | tests/test_coca_ingestor_contract.py::test_graceful_failure_on_missing_data, tests/test_coca_ingestor_defensive_paths.py::test_annotation_xml_parse_failure, tests/test_coca_ingestor_defensive_paths.py::test_invalid_dicom_file, tests/test_coca_ingestor_defensive_paths.py::test_missing_image_position_patient, tests/test_coca_ingestor_defensive_paths.py::test_patient_without_series, tests/test_coca_ingestor_defensive_paths.py::test_series_without_dicoms, tests/test_coca_ingestor_edge_cases.py::test_no_dicom_files, tests/test_coca_ingestor_edge_cases.py::test_no_series_directories, tests/test_coca_ingestor_synthetic.py::test_missing_dicom_files | DAT005_annotation_xml_parse_failure_20260503_192410_256828.json, DAT005_invalid_dicom_file_20260503_192410_258621.json, DAT005_missing_image_position_patient_20260503_192410_254732.json, DAT005_no_dicom_files_20260503_192410_271747.json, DAT005_no_series_directories_20260503_192410_270322.json, DAT005_patient_without_series_20260503_192410_251384.json, DAT005_series_without_dicoms_20260503_192410_252830.json, coca_ingestor_missing_data_20260503_192410_246795.json | PASS |
| DAT-006 | Lazy Patient Access | tests/test_coca_dataset.py::test_get_sample_on_real_dataset, tests/test_coca_ingestor_synthetic.py::test_get_patient_api, tests/test_coca_ingestor_synthetic.py::test_get_sample_generates_image_and_mask, tests/test_coca_ingestor_synthetic.py::test_get_volume_api, tests/test_coca_ingestor_synthetic.py::test_lazy_patient_loading | coca_get_sample_validation_20260503_192410_112791.json | PASS |
| DAT-007 | Dataset Element Boundary Handling | tests/test_coca_dataset.py::test_slice_index_out_of_bounds, tests/test_coca_ingestor_edge_cases.py::test_annotation_slice_out_of_bounds, tests/test_coca_ingestor_synthetic.py::test_get_sample_skips_invalid_slice_annotations, tests/test_coca_ingestor_synthetic.py::test_slice_index_out_of_bounds |  | LINKED |
| DAT-008 | Deterministic Patient Sample Retrieval | tests/test_coca_ingestor_synthetic.py::test_deterministic_slice_retrieval, tests/test_coca_nongated_ingestor.py::test_deterministic_patient_loading |  | LINKED |
| DAT-009 | Annotation Geometry Integrity | tests/test_annotation_geometry_integrity.py::test_invalid_polygon_skipped, tests/test_annotation_geometry_integrity.py::test_valid_annotation_geometry, tests/test_coca_ingestor_edge_cases.py::test_roi_with_insufficient_points_ignored, tests/test_coca_ingestor_synthetic.py::test_annotation_with_missing_image_index |  | LINKED |
| DAT-010 | Optional Annotation Handling | tests/test_annotation_geometry_integrity.py::test_missing_annotation_file_returns_empty, tests/test_coca_ingestor_edge_cases.py::test_dataset_without_annotations |  | LINKED |
| DAT-011 | Dataset Partition Generation | tests/test_coca_ingestor_synthetic.py::test_ingest_dataset_multiple_patients, tests/test_data_splitting.py::test_datasource_partition_assignment, tests/test_data_splitting.py::test_deterministic_holdout_split_generates_three_partitions |  | LINKED |
| DAT-012 | CT slices must be sorted by ImagePositionPatient[2] before volume construction | tests/test_coca_ingestor_synthetic.py::test_ct_volume_sorted_by_z_position |  | LINKED |
| DAT-013 | Annotation Rasterization | tests/test_annotation_rasterization.py::test_contour_is_rasterized_into_binary_mask, tests/test_annotation_rasterization.py::test_rasterized_mask_aligns_with_source_dimensions, tests/test_coronary_calcium_task.py::test_coronary_calcium_task_yields_masks_for_annotated_slices |  | LINKED |
| DAT-014 | Domain-Safe Ingestion Errors | tests/test_annotation_rasterization.py::test_ingestor_raises_dataset_structure_error_not_runtime_error, tests/test_annotation_rasterization.py::test_missing_patient_directory_raises_dataset_structure_error |  | LINKED |
| DAT-015 | Nongated Dataset Structure Validation | tests/test_coca_nongated_ingestor.py::test_empty_dataset_root_raises, tests/test_coca_nongated_ingestor.py::test_missing_image_position_patient_warns_and_skips, tests/test_coca_nongated_ingestor.py::test_missing_patient_directory_raises, tests/test_coca_nongated_ingestor.py::test_no_dicom_files_in_patient_dir_raises, tests/test_coca_nongated_ingestor.py::test_scores_file_not_found_raises, tests/test_coca_nongated_ingestor.py::test_scores_xlsx_missing_required_column_raises, tests/test_coca_nongated_ingestor.py::test_spacing_fallback_on_missing_dicom_metadata |  | LINKED |
| DAT-016 | Nongated Score Loading | tests/test_coca_nongated_ingestor.py::test_blank_score_cell_treated_as_zero, tests/test_coca_nongated_ingestor.py::test_get_sample_returns_volume_and_score_array, tests/test_coca_nongated_ingestor.py::test_missing_score_entry_uses_zero_fill, tests/test_coca_nongated_ingestor.py::test_missing_score_entry_warns_via_report, tests/test_coca_nongated_ingestor.py::test_scores_attached_to_patient_metadata |  | LINKED |
| DOC-001 | Machine-Readable Requirements Definition | tests/test_project_structure.py::test_project_documentation_structure |  | LINKED |
| DOC-002 | Basic Project Documentation | tests/test_project_structure.py::test_project_documentation_structure |  | LINKED |
| DOC-003 | Traceability Documentation | tests/test_reporting_and_traceability.py::test_traceability_matrix_can_be_generated |  | LINKED |
| INF-001 | Inference Capability | tests/test_model_persistence.py::test_inference_produces_output_on_new_data |  | LINKED |
| INF-002 | Inference Determinism | tests/test_model_persistence.py::test_inference_is_deterministic_for_same_input |  | LINKED |
| MOD-001 | Model Artifact Generation | tests/test_models.py::test_small_segmentation_cnn_output_shape, tests/test_models.py::test_unet2d_configurable_channels, tests/test_models.py::test_unet2d_gradient_flows_through_network, tests/test_models.py::test_unet2d_output_shape_matches_input, tests/test_models.py::test_unet2d_produces_finite_outputs |  | LINKED |
| MOD-002 | Model Artifact Persistence | tests/test_model_persistence.py::test_model_state_dict_can_be_saved_and_loaded |  | LINKED |
| MOD-003 | Model Evaluation Capability | tests/test_model_persistence.py::test_model_evaluation_on_held_out_partition |  | LINKED |
| MOD-004 | Calcium Score Regression Model | tests/test_models.py::test_calcium_score_regressor_accepts_any_spatial_size, tests/test_models.py::test_calcium_score_regressor_configurable_base_channels, tests/test_models.py::test_calcium_score_regressor_configurable_outputs, tests/test_models.py::test_calcium_score_regressor_finite_output_on_random_input, tests/test_models.py::test_calcium_score_regressor_gradient_flows, tests/test_models.py::test_calcium_score_regressor_output_shape |  | LINKED |
| REP-001 | Training Report Generation | tests/test_reporting_and_traceability.py::test_training_report_generated_contains_metrics |  | LINKED |
| REP-002 | Visualization Support | tests/test_reporting_and_traceability.py::test_visualization_figures_can_be_generated |  | LINKED |
| REP-003 | PDF Report Export | tests/test_status_report.py::test_status_report_prints_model_testing_section, tests/test_status_report.py::test_status_report_prints_training_section, tests/test_status_report.py::test_status_report_prints_tuning_section |  | LINKED |
| REP-004 | Status Report | tests/test_status_report.py::test_status_report_handles_no_runs, tests/test_status_report.py::test_status_report_prints_model_testing_section, tests/test_status_report.py::test_status_report_prints_training_section, tests/test_status_report.py::test_status_report_prints_tuning_section |  | LINKED |
| SYS-001 | Deterministic System Behavior | tests/test_annotation_geometry_integrity.py::test_volume_sorted_by_z_position |  | LINKED |
| SYS-002 | Controlled Data Splitting | tests/test_data_splitting.py::test_deterministic_holdout_split_generates_three_partitions, tests/test_data_splitting.py::test_split_is_reproducible_with_same_seed |  | LINKED |
| SYS-003 | Traceable Verification | tests/test_reporting_and_traceability.py::test_traceability_matrix_can_be_generated |  | LINKED |
| SYS-004 | Coronary Model Development | tests/test_system_architecture.py::test_coronary_task_and_ingestor_exist_and_are_importable |  | LINKED |
| SYS-005 | Model Improvement Capability | tests/test_system_architecture.py::test_coronary_task_and_ingestor_exist_and_are_importable |  | LINKED |
| SYS-006 | Dataset Task Encapsulation | tests/test_system_architecture.py::test_ingestor_and_task_are_in_project_not_toolkit |  | LINKED |
| TRN-001 | Controlled Model Initialization | tests/test_model_persistence.py::test_training_initialization_is_deterministic |  | LINKED |
| TRN-002 | Training Artifact Generation | tests/test_model_persistence.py::test_training_artifacts_generated_after_training |  | LINKED |
| TRN-003 | Coronary Model Training | tests/test_coronary_calcium_task.py::test_coronary_calcium_task_compute_loss_returns_finite_scalar, tests/test_coronary_calcium_task.py::test_coronary_calcium_task_loss_near_zero_on_perfect_prediction, tests/test_coronary_calcium_task.py::test_coronary_calcium_task_loss_penalises_wrong_predictions, tests/test_nongated_calcium_score_task.py::test_compute_loss_is_finite_scalar, tests/test_nongated_calcium_score_task.py::test_gradients_flow_through_loss |  | LINKED |
| TRN-004 | Coronary Model Retraining | tests/test_model_persistence.py::test_model_can_be_retrained_with_updated_config |  | LINKED |
| TRN-005 | Coronary Dataset Training Interface | tests/test_data_splitting.py::test_datasource_exposes_training_samples_via_task |  | LINKED |
| TSK-001 | Task Definition Interface | tests/test_coronary_calcium_task.py::test_coronary_calcium_task_yields_masks_for_annotated_slices, tests/test_system_architecture.py::test_coronary_task_implements_toolkit_interface |  | LINKED |
| TSK-002 | Coronary Calcium Detection Task | tests/test_coronary_calcium_task.py::test_coronary_calcium_task_ignores_short_contours, tests/test_coronary_calcium_task.py::test_coronary_calcium_task_input_is_hu_normalised, tests/test_coronary_calcium_task.py::test_coronary_calcium_task_yields_masks_for_annotated_slices |  | LINKED |
| TSK-003 | Task Determinism | tests/test_system_architecture.py::test_task_output_is_deterministic_for_same_input |  | LINKED |
| TSK-004 | Calcium Thresholding Support | tests/test_coronary_calcium_task.py::test_coronary_calcium_task_input_is_hu_normalised, tests/test_nongated_calcium_score_task.py::test_input_hu_normalised_above_window, tests/test_nongated_calcium_score_task.py::test_input_hu_normalised_below_window |  | LINKED |
| TSK-005 | Nongated Calcium Score Regression Task | tests/test_nongated_calcium_score_task.py::test_compute_loss_higher_for_wrong_prediction, tests/test_nongated_calcium_score_task.py::test_compute_loss_is_finite_scalar, tests/test_nongated_calcium_score_task.py::test_compute_loss_zero_on_perfect_prediction, tests/test_nongated_calcium_score_task.py::test_input_hu_normalised_above_window, tests/test_nongated_calcium_score_task.py::test_input_hu_normalised_below_window, tests/test_nongated_calcium_score_task.py::test_input_tensor_shape, tests/test_nongated_calcium_score_task.py::test_input_wl40_maps_to_zero, tests/test_nongated_calcium_score_task.py::test_missing_metadata_defaults_to_zero_score, tests/test_nongated_calcium_score_task.py::test_target_broadcast_identical_across_slices, tests/test_nongated_calcium_score_task.py::test_target_is_log1p_of_vessel_scores, tests/test_nongated_calcium_score_task.py::test_target_tensor_shape, tests/test_nongated_calcium_score_task.py::test_yields_one_sample_per_slice, tests/test_nongated_calcium_score_task.py::test_zero_score_target_is_zero |  | LINKED |
| VER-001 | Automated Verification Execution | tests/test_reporting_and_traceability.py::test_test_suite_exists_and_contains_test_functions |  | LINKED |
| VER-002 | Evidence Capture | tests/test_reporting_and_traceability.py::test_evidence_report_can_be_saved_and_loaded |  | LINKED |
| VER-003 | Model Performance Verification | tests/test_reporting_and_traceability.py::test_segmentation_evaluator_produces_performance_metrics |  | LINKED |


---


---
Total Requirements: 48

Tested: 48

Failures: 0
