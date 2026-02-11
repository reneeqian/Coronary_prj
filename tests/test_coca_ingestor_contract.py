from pathlib import Path
import numpy as np
import pytest

from src.ingestors.coca_gated_ingestor import COCAGatedIngestor
from medical_image_ai_toolkit.contracts.patient_sample_contract import enforce_patient_sample_contract
from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample
from medical_image_ai_toolkit.evidence.evidence_report import EvidenceReport


def _make_dummy_patient_sample() -> PatientSample:
    return PatientSample(
        patient_id="DUMMY-COCA-001",
        image_volume=np.zeros((16, 64, 64), dtype=np.float32),
        spacing=(1.0, 1.0, 1.0),
        annotations=None,
    )

@pytest.mark.requirement("CAC-FR-01")
@pytest.mark.requirement("CAC-DR-01")
@pytest.mark.requirement("CAC-DR-02")
@pytest.mark.requirement("CAC-DR-03")
@pytest.mark.requirement("MIT-VRF-01")

def test_CAC_FR_01_ingest_ct_volumes_from_root():
    report = EvidenceReport(subject="COCA Ingestor → PatientSample Contract")

    dataset_root = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "coca"
        / "cocacoronarycalciumandchestcts-2"
        / "Gated_release_final"
    )

    if dataset_root.exists():
        report.info(
            message="Using real COCA dataset", 
            requirement_id="CAC-FR-01",
            context=str(dataset_root))
        ingestor = COCAGatedIngestor(dataset_root=dataset_root)
        sample = ingestor.ingest_patient("0")
    else:
        report.warn(
            message="COCA dataset not found, using dummy sample",
            requirement_id="CAC-FR-01")
        sample = _make_dummy_patient_sample()

    contract_report = enforce_patient_sample_contract(
        sample,
        require_annotations=False,
    )

    report.issues.extend(contract_report.issues)
    report.auto_save("coca_ingestor_contract")

    assert not report.has_errors, report.summary()


def test_CAC_FR_01_graceful_failure_on_missing_data():
    report = EvidenceReport(
        subject="COCA Ingestor → Missing Dataset Failure Mode"
    )

    fake_root = PROJECT_ROOT / "data" / "raw" / "coca" / "DOES_NOT_EXIST"

    try:
        ingestor = COCAGatedIngestor(dataset_root=fake_root)
        ingestor.ingest_patient("0")
        report.error(
            message="Ingestor did not fail on missing dataset",
            requirement_id="CAC-FR-01",
        )
    except Exception as e:
        report.info(
            message="Ingestor failed as expected",
            requirement_id="CAC-FR-01",
            context=str(e),
        )

    report.auto_save("coca_ingestor_missing_data")
    assert not report.has_errors, report.summary()