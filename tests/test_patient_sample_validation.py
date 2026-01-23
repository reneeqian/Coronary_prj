from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestors.coca_gated_ingestor import COCAGatedIngestor
from src.validators.patient_sample_validator import validate_patient_sample


def test_coca_patient_sample_validation():
    print("\n=== Running PatientSample validation test ===")

    dataset_root = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "coca"
        / "cocacoronarycalciumandchestcts-2"
        / "Gated_release_final"
    )

    ingestor = COCAGatedIngestor(dataset_root=dataset_root)

    sample = ingestor.ingest_patient("0")

    report = validate_patient_sample(
        sample,
        require_annotations=True,
    )

    # --- print report for now ---
    report.print_summary()

    # --- test condition ---
    assert not report.has_errors, (
        "PatientSample validation failed:\n"
        + report.to_string()
    )
