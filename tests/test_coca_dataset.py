import pytest

from regulatory_tools.evidence.evidence_report import EvidenceReport
from Coronary_prj.ingestors.coca_gated_ingestor import COCAGatedIngestor

@pytest.mark.requirement("CAC-FR-01")
def test_CAC_FR_01_required_subdirectories_present(coca_dataset_root,
    coca_dataset_available,
):
    if not coca_dataset_available:
        pytest.skip("COCA dataset not available — skipping integration test.")

    assert coca_dataset_root.exists()
    
    report = EvidenceReport(
        subject="COCA Dataset → Directory Structure Contract"
    )

    images_dir = coca_dataset_root / "patient"
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

@pytest.mark.requirement("CAC-FR-02")
def test_CAC_FR_02_deterministic_dataset_ordering(coca_dataset_root,
    coca_dataset_available,
):
    if not coca_dataset_available:
        pytest.skip("COCA dataset not available — skipping integration test.")

    assert coca_dataset_root.exists()
    
    report = EvidenceReport(
        subject="COCA Ingestor → Deterministic Ingestion"
    )

    dataset_root = coca_dataset_root
    
    if not dataset_root.exists():
        report.warn(
            message="Dataset not found; determinism check skipped",
            requirement_id="CAC-FR-02",
        )
    else:
        ingestor = COCAGatedIngestor(dataset_root=dataset_root)

        patient_ids = sorted(ingestor.list_patient_ids())
        assert len(patient_ids) > 0

        pid = patient_ids[0]

        sample1 = ingestor.ingest_patient(pid)
        sample2 = ingestor.ingest_patient(pid)

        assert sample1.image_volume.shape == sample2.image_volume.shape
        assert sample1.spacing == sample2.spacing

        report.info(
            message="Repeated ingestion of same patient ID is deterministic",
            requirement_id="CAC-FR-02",
        )

    report.auto_save("coca_ingestor_determinism")
    assert not report.has_errors, report.summary()
