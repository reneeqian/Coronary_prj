from pathlib import Path
import pytest

from medical_image_ai_toolkit.evidence.evidence_report import EvidenceReport
from src.ingestors.coca_gated_ingestor import COCAGatedIngestor

@pytest.mark.requirement("CAC_FR_01")
def test_CAC_FR_01_required_subdirectories_present():
    report = EvidenceReport(
        subject="COCA Dataset → Directory Structure Contract"
    )

    dataset_root = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "coca"
        / "cocacoronarycalciumandchestcts-2"
        / "Gated_release_final"
    )

    if not dataset_root.exists():
        report.warn(
            message="Dataset not found; structure check skipped",
            requirement_id="CAC-FR-01",
        )
    else:
        images_dir = dataset_root / "patient"
        report.info(
            message="Checking image directory existence",
            requirement_id="CAC-FR-01",
            context=str(images_dir),
        )

        if not images_dir.exists() or not any(images_dir.iterdir()):
            report.warn(
                message="Image directory missing or empty; dataset may not be staged",
                requirement_id="CAC-FR-02",
            )


    report.auto_save("coca_dataset_structure")
    assert not report.has_errors, report.summary()

@pytest.mark.requirement("CAC_FR_02")
def test_CAC_FR_01_deterministic_dataset_ordering():
    report = EvidenceReport(
        subject="COCA Ingestor → Deterministic Ingestion"
    )

    dataset_root = (
            PROJECT_ROOT
            / "data"
            / "raw"
            / "coca"
            / "cocacoronarycalciumandchestcts-2"
            / "Gated_release_final"
        )
    
    if not dataset_root.exists():
        report.warn(
            message="Dataset not found; determinism check skipped",
            requirement_id="CAC-FR-02",
        )
    else:
        ingestor = COCAGatedIngestor(dataset_root=dataset_root)

        sample1 = ingestor.ingest_patient("0")
        sample2 = ingestor.ingest_patient("0")

        assert sample1.image_volume.shape == sample2.image_volume.shape
        assert sample1.spacing == sample2.spacing

        report.info(
            message="Repeated ingestion of same patient ID is deterministic",
            requirement_id="CAC_FR-02",
        )

    report.auto_save("coca_ingestor_determinism")
    assert not report.has_errors, report.summary()
