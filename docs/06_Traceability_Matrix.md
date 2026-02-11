# Traceability Matrix (Demonstrative)

This matrix demonstrates how requirement traceability would be structured
in a regulated medical software environment.

| Requirement | Module | Test | Status |
|-------------|--------|------|--------|
| CAC-FR-01 | COCAGatedIngestor | test_CAC_FR_01_ingest_ct_volumes_from_root | Pass/Skip |
| CAC-FR-02 | COCAGatedIngestor | test_CAC_FR_01_deterministic_dataset_ordering | Pass/Skip |
| CAC-FR-07 | Ingestor | test_CAC_FR_01_graceful_failure_on_missing_data | Pass |
| CAC-DR-01 | PatientSample | Contract enforcement test | Pass |
| CAC-SR-01 | Determinism | Determinism test | Pass |

This project demonstrates traceable engineering rigor but is not a regulated product.
