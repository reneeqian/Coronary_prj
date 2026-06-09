from unittest.mock import patch

import numpy as np
import pytest
from regulatory_tools.evidence.evidence_report import EvidenceReport

from Coronary_prj.ingestors.coca_gated_ingestor import (
    COCAGatedIngestor,
)

# =============================================================================
# Synthetic DICOM Helpers
# =============================================================================


class SimpleDicom:
    def __init__(self, z=0):
        self.ImagePositionPatient = [0.0, 0.0, float(z)]
        self.PixelSpacing = [1.0, 1.0]
        self.SliceThickness = 1.0
        self.RescaleSlope = 1.0
        self.RescaleIntercept = 0.0
        self.pixel_array = np.zeros((2, 2), dtype=np.float32)


@pytest.mark.requires_dataset
@pytest.mark.requirement("DAT-001")
def test_required_subdirectories_present(
    coca_dataset_root,
    coca_dataset_available,
    request,
    evidence_output_dir,
):
    if not coca_dataset_available:
        pytest.skip("COCA dataset not available — skipping integration test.")

    assert coca_dataset_root.exists()

    report = EvidenceReport(
        subject="COCA Dataset → Directory Structure Contract",
        test_id=request.node.nodeid,
    )

    images_dir = coca_dataset_root / "patient"
    report.info(
        message="Checking image directory existence",
        requirement_tag="DAT-001",
        context=str(images_dir),
    )

    if not images_dir.exists():
        report.error(
            message="Patient directory missing",
            requirement_tag="DAT-001",
        )
    elif not any(images_dir.iterdir()):
        report.error(
            message="Patient directory exists but is empty",
            requirement_tag="DAT-001",
        )

    report.auto_save("coca_dataset_structure", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requires_dataset
@pytest.mark.requirement("DAT-002")
def test_deterministic_dataset_ordering(
    coca_dataset_root,
    coca_dataset_available,
    request,
    evidence_output_dir,
):
    if not coca_dataset_available:
        pytest.skip("COCA dataset not available — skipping integration test.")

    assert coca_dataset_root.exists()

    report = EvidenceReport(
        subject="COCA Ingestor → Deterministic Ingestion",
        test_id=request.node.nodeid,
    )

    dataset_root = coca_dataset_root

    if not dataset_root.exists():
        report.warn(
            message="Dataset not found; determinism check skipped",
            requirement_tag="DAT-002",
        )
    else:
        ingestor = COCAGatedIngestor(dataset_root=dataset_root)

        patient_ids = sorted(ingestor.list_patient_ids())
        assert len(patient_ids) > 0

        pid = patient_ids[0]

        sample1 = ingestor.ingest_patient(pid)
        sample2 = ingestor.ingest_patient(pid)

        assert np.array_equal(sample1.image_volume, sample2.image_volume), (
            "Image volume differs across repeated ingestion"
        )
        assert sample1.spacing == sample2.spacing

        report.info(
            message="Repeated ingestion of same patient ID is deterministic",
            requirement_tag="DAT-002",
        )

    report.auto_save("coca_ingestor_determinism", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DAT-007")
def test_slice_index_out_of_bounds(tmp_path):

    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)

    (patient_dir / "a.dcm").write_text("fake")
    (patient_dir / "b.dcm").write_text("fake")

    with patch("pydicom.dcmread", return_value=SimpleDicom()):
        ingestor = COCAGatedIngestor(dataset_root=tmp_path)

        patient = ingestor.load_patient_sample("0")

        # volume only has 2 slices → index 10 is invalid
        with pytest.raises(IndexError):
            _ = patient.image_volume[10]


@pytest.mark.requires_dataset
@pytest.mark.requirement("DAT-006")
@pytest.mark.requirement("DAT-004")
def test_get_sample_on_real_dataset(
    coca_dataset_root,
    coca_dataset_available,
    request,
    evidence_output_dir,
):

    if not coca_dataset_available:
        pytest.skip("COCA dataset not available")

    report = EvidenceReport(
        subject="COCA Ingestor → get_sample Real Data Validation",
        test_id=request.node.nodeid,
    )

    ingestor = COCAGatedIngestor(coca_dataset_root)

    patient_ids = ingestor.list_patient_ids()
    pid = patient_ids[0]

    X, Y = ingestor.get_sample(pid)

    report.info(
        message=f"Generated {X.shape[0]} training slices",
        requirement_tag="DAT-006",
    )

    assert X.ndim == 3
    assert Y.ndim == 3
    assert X.shape == Y.shape

    if X.shape[0] > 0:
        assert np.sum(Y) > 0

    report.auto_save("coca_get_sample_validation", evidence_output_dir)
