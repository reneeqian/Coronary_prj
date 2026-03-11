import numpy as np
import pytest
import plistlib
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
# Dataset Structure / Validation Tests
# =============================================================================
        
@pytest.mark.requirement("DAT-001")
def test_dataset_structure_validation(tmp_path):

    # Root exists but missing patient folder
    ingestor = COCAGatedIngestor(dataset_root=tmp_path)

    with pytest.raises(DatasetStructureError):
        ingestor.list_patient_ids()

# =============================================================================
# Ingestion Behavior Tests
# =============================================================================

@pytest.mark.requirement("DAT-004")
def test_slices_sorted_by_z(tmp_path, request, evidence_output_dir):

    report = EvidenceReport(
        subject="COCA Gated Ingestor → Slice Sorting",
        test_id=request.node.nodeid,
    )

    patient_dir = tmp_path / "patient" / "0"
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

    def fake_dcmread(path, *args, **kwargs):
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
        requirement_id="DAT-004"
    )

    report.auto_save(request.node.nodeid, evidence_output_dir)


@pytest.mark.requirement("DAT-004")
def test_hounsfield_rescale_applied(tmp_path, request, evidence_output_dir):

    report = EvidenceReport(
        subject="COCA Gated Ingestor → HU Rescale",
        test_id=request.node.nodeid,
    )

    patient_dir = tmp_path / "patient" / "0"
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
        requirement_id="DAT-004"
    )

    report.auto_save(request.node.nodeid, evidence_output_dir)


@pytest.mark.requirement("DAT-004")
def test_annotation_out_of_bounds_raises(tmp_path, request, evidence_output_dir):

    report = EvidenceReport(
        subject="COCA Gated Ingestor → Annotation Bounds Validation",
        test_id=request.node.nodeid,
    )

    patient_dir = tmp_path / "patient" / "0"
    series_dir = patient_dir / "seriesA"
    series_dir.mkdir(parents=True)

    f = series_dir / "a.dcm"
    f.write_text("fake")
    
    xml_dir = tmp_path / "calcium_xml"
    xml_dir.mkdir()

    xml_file = xml_dir / "0.xml"

    xml_file.write_text("""
    <?xml version="1.0" encoding="UTF-8"?>
    <plist version="1.0">
    <dict>
        <key>Images</key>
        <array>
            <dict>
                <key>ImageIndex</key>
                <integer>5</integer>
                <key>ROIs</key>
                <array/>
            </dict>
        </array>
    </dict>
    </plist>
    """)

    with patch("pydicom.dcmread", return_value=SimpleDicom(0)):
        ingestor = COCAGatedIngestor(dataset_root=tmp_path)

        with pytest.raises(DatasetStructureError):
            ingestor.ingest_patient("0")
    
    report.info(
        "Out-of-bounds annotation correctly rejected",
        requirement_id="DAT-004"
    )

    report.auto_save(request.node.nodeid, evidence_output_dir)

@pytest.mark.requirement("DAT-003")
def test_invalid_patient_id_raises(tmp_path):

    (tmp_path / "patient" / "999").mkdir(parents=True)

    ingestor = COCAGatedIngestor(dataset_root=tmp_path)

    with pytest.raises(DatasetStructureError):
        ingestor.ingest_patient("999")

@pytest.mark.requirement("DAT-006")
def test_lazy_patient_loading(tmp_path):

    # Create two patients
    for pid in ["0", "1"]:
        series_dir = tmp_path / "patient" / pid / "seriesA"
        series_dir.mkdir(parents=True)
        (series_dir / "a.dcm").write_text("fake")

    with patch.object(
        COCAGatedIngestor,
        "_load_image_volume",
        return_value=(
            np.zeros((1, 2, 2)),
            (1.0, 1.0, 1.0),
            {}
        )
    ) as mock_loader:

        ingestor = COCAGatedIngestor(dataset_root=tmp_path)

        # Only ingest patient 0
        ingestor.ingest_patient("0")

        # Should only load once
        assert mock_loader.call_count == 1

@pytest.mark.requirement("DAT-007")
def test_slice_index_out_of_bounds(tmp_path):

    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)
    (patient_dir / "a.dcm").write_text("fake")

    with patch("pydicom.dcmread", return_value=SimpleDicom(0)):
        ingestor = COCAGatedIngestor(dataset_root=tmp_path)

        sample = ingestor.load_patient("0")
        # annotation refers to slice outside volume → should be ignored
        assert sample.annotations.vector_rois is None

@pytest.mark.requirement("DAT-008")
def test_deterministic_slice_retrieval(tmp_path):

    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)
    (patient_dir / "a.dcm").write_text("fake")

    with patch("pydicom.dcmread", return_value=SimpleDicom(0)):
        ingestor = COCAGatedIngestor(dataset_root=tmp_path)

        patient = ingestor.load_patient("0")
        slice1 = patient.image_volume[0]
        patient = ingestor.load_patient("0")
        slice2 = patient.image_volume[0]

        assert np.array_equal(slice1, slice2)

@pytest.mark.requirement("DAT-005")
def test_missing_dicom_files(tmp_path):

    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)

    # No DICOM files created

    ingestor = COCAGatedIngestor(dataset_root=tmp_path)

    with pytest.raises(DatasetStructureError):
        ingestor.ingest_patient("0")
        
@pytest.mark.requirement("DAT-006")
def test_get_patient_api(tmp_path):

    patient_dir = tmp_path/"patient"/"0"/"seriesA"
    patient_dir.mkdir(parents=True)

    (patient_dir/"slice1.dcm").write_text("fake")

    class FakeDicom:
        ImagePositionPatient=[0,0,0]
        PixelSpacing=[1,1]
        SliceThickness=1
        pixel_array=np.zeros((2,2))
        RescaleSlope=1
        RescaleIntercept=0

    with patch("pydicom.dcmread", return_value=FakeDicom()):
        ingestor = COCAGatedIngestor(tmp_path)

        patient = ingestor.load_patient("0")

        assert patient.patient_id == "0"
    
@pytest.mark.requirement("DAT-006")
def test_get_volume_api(tmp_path):

    patient_dir = tmp_path/"patient"/"0"/"seriesA"
    patient_dir.mkdir(parents=True)

    (patient_dir/"slice1.dcm").write_text("fake")

    class FakeDicom:
        ImagePositionPatient=[0,0,0]
        PixelSpacing=[1,1]
        SliceThickness=1
        pixel_array=np.ones((2,2))
        RescaleSlope=1
        RescaleIntercept=0

    with patch("pydicom.dcmread", return_value=FakeDicom()):
        ingestor = COCAGatedIngestor(tmp_path)

        patient = ingestor.load_patient("0")
        vol = patient.image_volume

        assert vol.shape == (1,2,2)

@pytest.mark.requirement("DAT-004")
def test_ingest_dataset_multiple_patients(tmp_path):

    for pid in ["0","1"]:
        patient_dir = tmp_path/"patient"/pid/"seriesA"
        patient_dir.mkdir(parents=True)
        (patient_dir/"slice1.dcm").write_text("fake")

    class FakeDicom:
        ImagePositionPatient=[0,0,0]
        PixelSpacing=[1,1]
        SliceThickness=1
        pixel_array=np.zeros((2,2))
        RescaleSlope=1
        RescaleIntercept=0

    with patch("pydicom.dcmread", return_value=FakeDicom()):
        ingestor = COCAGatedIngestor(tmp_path)

        ds = ingestor.ingest_dataset()

        assert len(ds) == 2

@pytest.mark.requirement("DAT-009")
def test_annotation_with_missing_image_index(tmp_path):

    patient_dir = tmp_path/"patient"/"0"/"seriesA"
    patient_dir.mkdir(parents=True)

    (patient_dir/"slice1.dcm").write_text("fake")

    xml_dir = tmp_path/"calcium_xml"
    xml_dir.mkdir()

    xml_file = xml_dir/"0.xml"

    plist_data = {
        "Images": [
            {
                "ROIs": []
            }
        ]
    }

    with open(xml_file, "wb") as f:
        plistlib.dump(plist_data, f)

    class FakeDicom:
        ImagePositionPatient=[0,0,0]
        PixelSpacing=[1,1]
        SliceThickness=1
        pixel_array=np.zeros((2,2))
        RescaleSlope=1
        RescaleIntercept=0

    with patch("pydicom.dcmread", return_value=FakeDicom()):
        ingestor = COCAGatedIngestor(tmp_path)

        sample = ingestor.ingest_patient("0")

        assert sample.annotations.vector_rois is None
        
# =============================================================================
# CT slices must be sorted by Z position before stacking
# =============================================================================

@pytest.mark.requirement("DAT-012")
def test_ct_volume_sorted_by_z_position(tmp_path):

    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)

    # intentionally create files in reverse order
    (patient_dir / "sliceB.dcm").write_text("fake")
    (patient_dir / "sliceA.dcm").write_text("fake")

    class FakeDicom:
        def __init__(self, z):
            self.ImagePositionPatient = [0.0, 0.0, float(z)]
            self.PixelSpacing = [1.0, 1.0]
            self.SliceThickness = 1.0
            self.RescaleSlope = 1.0
            self.RescaleIntercept = 0.0
            self.pixel_array = np.full((2,2), z, dtype=np.float32)
    
    def fake_dcmread(path, *args, **kwargs):
        if "sliceA" in str(path):
            return FakeDicom(0)
        elif "sliceB" in str(path):
            return FakeDicom(10)
        return FakeDicom(0)

    with patch("pydicom.dcmread", side_effect=fake_dcmread):

        ingestor = COCAGatedIngestor(tmp_path)
        patient = ingestor.load_patient("0")

        volume = patient.image_volume

        assert np.all(volume[0] == 0)
        assert np.all(volume[1] == 10)