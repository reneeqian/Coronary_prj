import numpy as np

from src.dataobjects.patient_sample import PatientSample
from src.contracts.patient_sample_contract import enforce_patient_sample_contract
from evidence.evidence_report import EvidenceReport


def test_patient_sample_contract_with_dummy_data():
    report = EvidenceReport(subject="PatientSample Contract (Dummy)")

    sample = PatientSample(
        patient_id="DUMMY-001",
        image_volume=np.zeros((16, 64, 64), dtype=np.float32),
        spacing=(1.0, 1.0, 1.0),
        annotations=None,
    )

    contract_report = enforce_patient_sample_contract(
        sample,
        require_annotations=False,
    )

    report.issues.extend(contract_report.issues)
    report.auto_save("patient_sample_contract_dummy")

    assert not report.has_errors, report.summary()
