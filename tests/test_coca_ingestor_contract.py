import numpy as np
import pytest
from medical_image_ai_toolkit.dataobjects.patient_sample_contract import (
    enforce_patient_sample_contract,
)
from regulatory_tools.evidence.evidence_report import EvidenceReport

from Coronary_prj.ingestors.coca_gated_ingestor import COCAGatedIngestor, DatasetStructureError


@pytest.mark.requires_dataset
@pytest.mark.requirement("DAT-004")
def test_ingest_ct_volumes_from_root(
    coca_dataset_root,
    coca_dataset_available,
    request,
    evidence_output_dir,
):
    if not coca_dataset_available:
        pytest.skip("COCA dataset not available — skipping integration test.")

    assert coca_dataset_root.exists()

    report = EvidenceReport(
        subject="COCA Ingestor → PatientSample Contract",
        test_id=request.node.nodeid,
    )

    dataset_root = coca_dataset_root

    if dataset_root.exists():
        report.info(
            message="Using real COCA dataset", requirement_tag="DAT-004", context=str(dataset_root)
        )
        ingestor = COCAGatedIngestor(dataset_root=dataset_root)
        sample = ingestor.load_patient_sample("0")

    contract_report = enforce_patient_sample_contract(
        sample,
        require_annotations=False,
    )

    report.issues.extend(contract_report.issues)
    report.auto_save("coca_ingestor_contract", evidence_output_dir)

    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DAT-005")
def test_graceful_failure_on_missing_data(tmp_path, request, evidence_output_dir):
    report = EvidenceReport(
        subject="COCA Ingestor → Missing Dataset Failure Mode",
        test_id=request.node.nodeid,
    )

    fake_root = tmp_path / "does_not_exist"

    ingestor = COCAGatedIngestor(dataset_root=fake_root)

    with pytest.raises(DatasetStructureError) as exc:
        ingestor.ingest_patient("0")

    report.info(
        message="Ingestor failed as expected",
        requirement_tag="DAT-005",
        context=str(exc.value),
    )

    report.auto_save("coca_ingestor_missing_data", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DAT-004")
def test_patient_sample_contract(tmp_path):

    # create minimal dataset structure
    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)

    (patient_dir / "slice1.dcm").write_text("fake")

    class FakeDicom:
        ImagePositionPatient = [0.0, 0.0, 0.0]
        PixelSpacing = [1.0, 1.0]
        SliceThickness = 1.0
        RescaleSlope = 1.0
        RescaleIntercept = 0.0
        pixel_array = np.zeros((2, 2), dtype=np.float32)

    from unittest.mock import patch

    with patch("pydicom.dcmread", return_value=FakeDicom()):
        ingestor = COCAGatedIngestor(tmp_path)

        patient = ingestor.load_patient_sample("0")

        assert patient.patient_id == "0"
        assert patient.image_volume.ndim == 3
