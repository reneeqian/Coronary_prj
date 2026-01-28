from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestors.coca_gated_ingestor import COCAGatedIngestor
from src.medimg_training.src.contracts.patient_sample_contract import enforce_patient_sample_contract
from src.medimg_training.src.dataobjects.patient_sample import PatientSample
from src.medimg_training.evidence.evidence_report import EvidenceReport


def _make_dummy_patient_sample() -> PatientSample:
    return PatientSample(
        patient_id="DUMMY-COCA-001",
        image_volume=np.zeros((16, 64, 64), dtype=np.float32),
        spacing=(1.0, 1.0, 1.0),
        annotations=None,
    )


def test_coca_ingestor_produces_valid_patient_sample():
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
        report.info("Using real COCA dataset", context=str(dataset_root))
        ingestor = COCAGatedIngestor(dataset_root=dataset_root)
        sample = ingestor.ingest_patient("0")
    else:
        report.warn("COCA dataset not found, using dummy sample")
        sample = _make_dummy_patient_sample()

    contract_report = enforce_patient_sample_contract(
        sample,
        require_annotations=False,
    )

    report.issues.extend(contract_report.issues)
    report.auto_save("coca_ingestor_contract")

    assert not report.has_errors, report.summary()
