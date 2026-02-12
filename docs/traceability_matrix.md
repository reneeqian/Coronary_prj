# Requirements Traceability Matrix

| Requirement ID | Description | Linked Tests | Evidence Artifacts | Status |
|----------------|-------------|--------------|--------------------|--------|
| ALG-FR-01 | The algorithm shall accept a 3D CT volume and output CAC predictions. |  |  | UNTESTED |
| ALG-FR-02 | The output prediction shall maintain consistent spatial dimensions relative to input volume. |  |  | UNTESTED |
| ALG-NFR-01 | Model weights shall load deterministically without modification to inference logic. |  |  | UNTESTED |
| DEP-FR-01 | The system shall expose a callable inference interface for integration into external workflows. |  |  | UNTESTED |
| DEP-NFR-01 | The deployment module shall not rely on hardcoded file paths. |  |  | UNTESTED |
| ING-FR-01 | The ingestion module shall validate dataset directory structure before attempting ingestion. | tests/test_coca_dataset.py::test_ING_FR_01_required_subdirectories_present, tests/test_coca_ingestor_contract.py::test_ING_FR_01_ingest_ct_volumes_from_root | coca_dataset_structure_20260212_120431_799172.json, coca_ingestor_contract_20260212_120432_268613.json | PASS |
| ING-FR-02 | The ingestion module shall enumerate patient IDs in a deterministic order. | tests/test_coca_dataset.py::test_ING_FR_02_deterministic_dataset_ordering | coca_ingestor_determinism_20260212_120432_188107.json | PASS |
| ING-FR-03 | The ingestion module shall raise a controlled exception if dataset root is missing. | tests/test_coca_ingestor_contract.py::test_ING_FR_03_graceful_failure_on_missing_data | coca_ingestor_missing_data_20260212_120432_273267.json | PASS |
| SAF-FR-01 | The system shall validate input tensor dimensions and spacing. |  |  | UNTESTED |
| SAF-FR-02 | Critical failures shall produce structured error messages. |  |  | UNTESTED |
| SYS-FR-01 | The system shall accept a CT study directory and produce a deterministic CAC detection output. |  |  | UNTESTED |
| SYS-NFR-01 | Given identical inputs and model weights, the system shall produce identical outputs. |  |  | UNTESTED |
