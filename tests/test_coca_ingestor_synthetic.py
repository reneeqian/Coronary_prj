import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from Coronary_prj.ingestors.coca_gated_ingestor import COCAGatedIngestor
from Coronary_prj.ingestors.coca_gated_ingestor import DatasetStructureError
from regulatory_tools.evidence.evidence_report import EvidenceReport


# =============================================================================
# Synthetic DICOM Helpers
# =============================================================================

class FakeDicom:
    def __init__(self, z, pixel_value):
        self.ImagePositionPatient = [0.0, 0.0, float(z)]
        self.PixelSpacing = [0.5, 0.5]
        self.SliceThickness = 1.0
        self.RescaleSlope = 1.0
        self.RescaleIntercept = 0.0
        self.pixel_array = np.full((4, 4), pixel_value, dtype=np.float32)


class SimpleDicom:
    def __init__(self, z):
        self.ImagePositionPatient = [0.0, 0.0, float(z)]
        self.PixelSpacing = [1.0, 1.0]
        self.SliceThickness = 1.0
        self.RescaleSlope = 1.0
        self.RescaleIntercept = 0.0
        self.pixel_array = np.zeros((2, 2), dtype=np.float32)


# =============================================================================
# ING-FR-04 — Slice Sorting
# =============================================================================

@pytest.mark.requirement("ING-FR-04")
def test_slices_sorted_by_z(tmp_path, request, evidence_output_dir):

    report = EvidenceReport(
        subject="COCA Gated Ingestor → Slice Sorting",
        test_id=request.node.nodeid,
    )

    patient_dir = tmp_path / "0"
    series_dir = patient_dir / "seriesA"
    series_dir.mkdir(parents=True)

    files = []
    for name in ["a.dcm", "b.dcm", "c.dcm"]:
        f = series_dir / name
        f.write_text("fake")
        files.append(f)

    fake_dicoms = {
        str(files[0]): FakeDicom(10, 10),
        str(files[1]): FakeDicom(5, 5),
        str(files[2]): FakeDicom(7, 7),
    }

    def fake_dcmread(path):
        return fake_dicoms[str(path)]

    with patch("pydicom.dcmread", side_effect=fake_dcmread):
        ingestor = COCAGatedIngestor(dataset_root=tmp_path)
        sample = ingestor.ingest_patient("0")

    volume = sample.image_volume

    assert volume[0, 0, 0] == 5
    assert volume[1, 0, 0] == 7
    assert volume[2, 0, 0] == 10

    # Attach artifact
    artifact_path = evidence_output_dir / "sorted_volume.npy"
    np.save(artifact_path, volume)
    
    report.info(
        "Slices correctly sorted by z-position",
        requirement_id="ING-FR-04"
    )

    report.auto_save(request.node.nodeid, evidence_output_dir)



# =============================================================================
# ING-FR-05 — HU Rescale
# =============================================================================

@pytest.mark.requirement("ING-FR-05")
def test_hounsfield_rescale_applied(tmp_path, request, evidence_output_dir):

    report = EvidenceReport(
        subject="COCA Gated Ingestor → HU Rescale",
        test_id=request.node.nodeid,
    )

    patient_dir = tmp_path / "0"
    series_dir = patient_dir / "seriesA"
    series_dir.mkdir(parents=True)

    f = series_dir / "a.dcm"
    f.write_text("fake")

    class RescaleDicom:
        ImagePositionPatient = [0, 0, 0]
        PixelSpacing = [1.0, 1.0]
        SliceThickness = 1.0
        RescaleSlope = 2.0
        RescaleIntercept = -1000.0
        pixel_array = np.ones((2, 2), dtype=np.float32) * 100

    with patch("pydicom.dcmread", return_value=RescaleDicom()):
        ingestor = COCAGatedIngestor(dataset_root=tmp_path)
        sample = ingestor.ingest_patient("0")

    assert np.all(sample.image_volume == -800.0)

    artifact_path = evidence_output_dir / "rescaled_volume.npy"
    np.save(artifact_path, sample.image_volume)
    
    report.info(
        "Hounsfield rescale applied correctly",
        requirement_id="ING-FR-05"
    )

    report.auto_save(request.node.nodeid, evidence_output_dir)



# =============================================================================
# ING-FR-07 — Bounds Validation
# =============================================================================

@pytest.mark.requirement("ING-FR-07")
def test_annotation_out_of_bounds_raises(tmp_path, request, evidence_output_dir):

    report = EvidenceReport(
        subject="COCA Gated Ingestor → Annotation Bounds Validation",
        test_id=request.node.nodeid,
    )

    patient_dir = tmp_path / "0"
    series_dir = patient_dir / "seriesA"
    series_dir.mkdir(parents=True)

    f = series_dir / "a.dcm"
    f.write_text("fake")

    annotation_file = patient_dir / "annotations.txt"
    annotation_file.write_text("5,stenosis\n")

    with patch("pydicom.dcmread", return_value=SimpleDicom(0)):
        ingestor = COCAGatedIngestor(dataset_root=tmp_path)

        with pytest.raises(DatasetStructureError):
            ingestor.ingest_patient("0")
    
    report.info(
        "Out-of-bounds annotation correctly rejected",
        requirement_id="ING-FR-07"
    )

    report.auto_save(request.node.nodeid, evidence_output_dir)
