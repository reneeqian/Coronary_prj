import numpy as np
import pytest

from Coronary_prj.ingestors.coca_gated_ingestor import COCAGatedIngestor, DatasetStructureError
from medical_image_ai_toolkit.dataobjects.patient_sample_contract import enforce_patient_sample_contract
from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample
from regulatory_tools.evidence.evidence_report import EvidenceReport


@pytest.mark.requires_dataset
@pytest.mark.requirement("DAT-004")
def test_ingest_ct_volumes_from_root(coca_dataset_root,
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
            message="Using real COCA dataset", 
            requirement_id="DAT-004",
            context=str(dataset_root))
        ingestor = COCAGatedIngestor(dataset_root=dataset_root)
        sample = ingestor.ingest_patient("0")

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
        requirement_id="DAT-005",
        context=str(exc.value),
    )

    report.auto_save("coca_ingestor_missing_data", evidence_output_dir)
    assert not report.has_errors, report.summary()
