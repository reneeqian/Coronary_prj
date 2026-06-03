import plistlib
from unittest.mock import patch

import numpy as np
import pytest
from regulatory_tools.evidence.evidence_report import EvidenceReport

from Coronary_prj.ingestors.coca_gated_ingestor import COCAGatedIngestor, DatasetStructureError

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
def test_dataset_structure_validation(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="DAT-001: missing patient/ directory raises DatasetStructureError")
    ingestor = COCAGatedIngestor(dataset_root=tmp_path)
    raised = False
    try:
        ingestor.list_patient_ids()
    except DatasetStructureError:
        raised = True
    if not raised:
        report.error("Expected DatasetStructureError when patient/ dir is absent; none raised", "DAT-001")
    report.info("DatasetStructureError raised when dataset root has no patient/ directory", "DAT-001")
    report.auto_save("DAT001_dataset_structure_validation", evidence_output_dir)
    assert not report.has_errors, report.summary()

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
        requirement_tag="DAT-004"
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
        requirement_tag="DAT-004"
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
        requirement_tag="DAT-004"
    )

    report.auto_save(request.node.nodeid, evidence_output_dir)

@pytest.mark.requirement("DAT-003")
def test_invalid_patient_id_raises(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="DAT-003: patient directory with no DICOM series raises DatasetStructureError")
    (tmp_path / "patient" / "999").mkdir(parents=True)
    ingestor = COCAGatedIngestor(dataset_root=tmp_path)
    raised = False
    try:
        ingestor.ingest_patient("999")
    except DatasetStructureError:
        raised = True
    if not raised:
        report.error("Expected DatasetStructureError for patient dir with no series subdirs; none raised", "DAT-003")
    report.info("DatasetStructureError raised for patient directory with no series subdirectories", "DAT-003")
    report.auto_save("DAT003_invalid_patient_id", evidence_output_dir)
    assert not report.has_errors, report.summary()

@pytest.mark.requirement("DAT-006")
def test_lazy_patient_loading(tmp_path, evidence_output_dir):

    report = EvidenceReport(subject="DAT-006: COCAGatedIngestor loads each patient volume exactly once")

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
        ingestor.ingest_patient("0")
        report.info(f"_load_image_volume called {mock_loader.call_count} time(s) for 1 patient ingestion", "DAT-006")

    report.auto_save("dat006_lazy_patient_loading", evidence_output_dir)
    assert not report.has_errors, report.summary()
    assert mock_loader.call_count == 1

@pytest.mark.requirement("DAT-007")
def test_slice_index_out_of_bounds(tmp_path, evidence_output_dir):

    report = EvidenceReport(subject="DAT-007: annotation referencing out-of-bounds slice is silently ignored")

    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)
    (patient_dir / "a.dcm").write_text("fake")

    with patch("pydicom.dcmread", return_value=SimpleDicom(0)):
        ingestor = COCAGatedIngestor(dataset_root=tmp_path)
        sample = ingestor.load_patient_sample("0")

    report.info(f"vector_rois={sample.annotations.vector_rois!r} (expected None — out-of-bounds annotation ignored)", "DAT-007")
    report.auto_save("dat007_slice_index_out_of_bounds", evidence_output_dir)
    assert not report.has_errors, report.summary()
    assert sample.annotations.vector_rois is None

@pytest.mark.requirement("DAT-008")
def test_deterministic_slice_retrieval(tmp_path, evidence_output_dir):

    report = EvidenceReport(subject="DAT-008: loading the same patient twice returns identical slice data")

    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)
    (patient_dir / "a.dcm").write_text("fake")

    with patch("pydicom.dcmread", return_value=SimpleDicom(0)):
        ingestor = COCAGatedIngestor(dataset_root=tmp_path)
        patient = ingestor.load_patient_sample("0")
        slice1 = patient.image_volume[0]
        patient = ingestor.load_patient_sample("0")
        slice2 = patient.image_volume[0]

    match = np.array_equal(slice1, slice2)
    report.info(f"Two consecutive loads returned array-equal slice[0]: {match}", "DAT-008")
    report.auto_save("dat008_deterministic_slice_retrieval", evidence_output_dir)
    assert not report.has_errors, report.summary()
    assert match

@pytest.mark.requirement("DAT-005")
def test_missing_dicom_files(tmp_path, evidence_output_dir):

    report = EvidenceReport(subject="DAT-005: patient series dir with no DICOM files raises DatasetStructureError")

    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)

    ingestor = COCAGatedIngestor(dataset_root=tmp_path)

    with pytest.raises(DatasetStructureError):
        ingestor.ingest_patient("0")

    report.info("DatasetStructureError raised when patient series directory has no .dcm files", "DAT-005")
    report.auto_save("dat005_missing_dicom_files", evidence_output_dir)
    assert not report.has_errors, report.summary()

@pytest.mark.requirement("DAT-006")
def test_get_patient_api(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="DAT-006: load_patient_sample returns PatientSample with correct patient_id")
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
        patient = ingestor.load_patient_sample("0")

    if patient.patient_id != "0":
        report.error(f"Expected patient_id='0', got '{patient.patient_id}'", "DAT-006")
    report.info(f"load_patient_sample returned PatientSample with patient_id='{patient.patient_id}'", "DAT-006")
    report.auto_save("DAT006_get_patient_api", evidence_output_dir)
    assert not report.has_errors, report.summary()
    assert patient.patient_id == "0"

@pytest.mark.requirement("DAT-006")
def test_get_volume_api(tmp_path, evidence_output_dir):

    report = EvidenceReport(subject="DAT-006: load_patient_sample returns a volume with expected shape")

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
        patient = ingestor.load_patient_sample("0")
        vol = patient.image_volume

    report.info(f"image_volume.shape={vol.shape} (expected (1,2,2))", "DAT-006")
    report.auto_save("dat006_get_volume_api", evidence_output_dir)
    assert not report.has_errors, report.summary()
    assert vol.shape == (1, 2, 2)

@pytest.mark.requirement("DAT-004")
def test_ingest_dataset_multiple_patients(tmp_path, evidence_output_dir):

    report = EvidenceReport(subject="DAT-004: ingest_dataset loads all patients from the dataset")

    for pid in ["0", "1"]:
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

    report.info(f"ingest_dataset() returned dataset with len={len(ds)} (expected 2)", "DAT-004")
    report.auto_save("dat004_ingest_dataset_multiple_patients", evidence_output_dir)
    assert not report.has_errors, report.summary()
    assert len(ds) == 2


@pytest.mark.requirement("DAT-011")
def test_ingest_dataset_enumerates_all_patient_ids(tmp_path, evidence_output_dir):

    report = EvidenceReport(subject="DAT-011: ingest_dataset enumerates all patient IDs without loading volumes")

    for pid in ["0", "1"]:
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
        patient_ids = ingestor.list_patient_ids()

    report.info(f"list_patient_ids() returned {patient_ids} (expected ['0', '1'])", "DAT-011")
    report.auto_save("dat011_ingest_dataset_enumerates_patient_ids", evidence_output_dir)
    assert not report.has_errors, report.summary()
    assert set(patient_ids) == {"0", "1"}

@pytest.mark.requirement("DAT-009")
def test_annotation_with_missing_image_index(tmp_path, evidence_output_dir):

    report = EvidenceReport(subject="DAT-009: annotation entry missing ImageIndex is treated as no annotation")

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

    report.info(f"vector_rois={sample.annotations.vector_rois!r} (expected None — entry without ImageIndex ignored)", "DAT-009")
    report.auto_save("dat009_annotation_missing_image_index", evidence_output_dir)
    assert not report.has_errors, report.summary()
    assert sample.annotations.vector_rois is None

# =============================================================================
# CT slices must be sorted by Z position before stacking
# =============================================================================

@pytest.mark.requirement("DAT-012")
def test_ct_volume_sorted_by_z_position(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="DAT-012: CT volume slices are sorted by Z position")
    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)

    # intentionally create files in reverse order
    (patient_dir / "sliceB.dcm").write_text("fake")
    (patient_dir / "sliceA.dcm").write_text("fake")

    class _FakeDicom:
        def __init__(self, z):
            self.ImagePositionPatient = [0.0, 0.0, float(z)]
            self.PixelSpacing = [1.0, 1.0]
            self.SliceThickness = 1.0
            self.RescaleSlope = 1.0
            self.RescaleIntercept = 0.0
            self.pixel_array = np.full((2,2), z, dtype=np.float32)

    def fake_dcmread(path, *args, **kwargs):
        if "sliceA" in str(path):
            return _FakeDicom(0)
        elif "sliceB" in str(path):
            return _FakeDicom(10)
        return _FakeDicom(0)

    with patch("pydicom.dcmread", side_effect=fake_dcmread):
        ingestor = COCAGatedIngestor(tmp_path)
        patient = ingestor.load_patient_sample("0")
        volume = patient.image_volume

    if not np.all(volume[0] == 0) or not np.all(volume[1] == 10):
        report.error(f"Volume not sorted by Z: slice0={volume[0,0,0]}, slice1={volume[1,0,0]}", "DAT-012")
    report.info(f"Slices sorted by Z: slice0 pixel={volume[0,0,0]:.0f} (z=0), slice1 pixel={volume[1,0,0]:.0f} (z=10)", "DAT-012")
    report.auto_save("DAT012_ct_volume_sorted_by_z", evidence_output_dir)
    assert not report.has_errors, report.summary()
    assert np.all(volume[0] == 0)
    assert np.all(volume[1] == 10)


def _write_single_roi_xml(xml_dir: "Path", patient_id: str) -> None:
    """Write a minimal calcium XML with one ROI on ImageIndex 1."""
    import plistlib as _pl
    xml_dir.mkdir(exist_ok=True)
    plist_data = {
        "Images": [{
            "ImageIndex": 1,
            "ROIs": [{
                "Name": "calcium",
                "NumberOfPoints": 4,
                "Point_px": ["(0,0)", "(1,0)", "(1,1)", "(0,1)"],
            }],
        }]
    }
    with open(xml_dir / f"{patient_id}.xml", "wb") as f:
        _pl.dump(plist_data, f)


@pytest.mark.requirement("DAT-004")
def test_get_sample_generates_image_and_mask(tmp_path, evidence_output_dir):

    report = EvidenceReport(subject="DAT-004: get_sample returns (image, mask) arrays with non-zero mask for annotated patient")

    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)
    (patient_dir / "slice1.dcm").write_text("fake")

    _write_single_roi_xml(tmp_path / "calcium_xml", "0")

    class FakeDicom:
        ImagePositionPatient=[0,0,0]
        PixelSpacing=[1,1]
        SliceThickness=1
        pixel_array=np.ones((4,4))
        RescaleSlope=1
        RescaleIntercept=0

    with patch("pydicom.dcmread", return_value=FakeDicom()):
        ingestor = COCAGatedIngestor(tmp_path)
        X, Y = ingestor.get_sample("0")

    report.info(f"get_sample: X.shape={X.shape}, Y.shape={Y.shape}, np.sum(Y)={np.sum(Y):.0f}", "DAT-004")
    report.auto_save("dat004_get_sample_generates_image_and_mask", evidence_output_dir)
    assert not report.has_errors, report.summary()
    assert X.shape == (1, 4, 4)
    assert Y.shape == (1, 4, 4)
    assert np.sum(Y) > 0


@pytest.mark.requirement("DAT-006")
def test_get_sample_returns_single_annotated_slice(tmp_path, evidence_output_dir):

    report = EvidenceReport(subject="DAT-006: get_sample returns exactly one image/mask pair for a single-slice annotated patient")

    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)
    (patient_dir / "slice1.dcm").write_text("fake")

    _write_single_roi_xml(tmp_path / "calcium_xml", "0")

    class FakeDicom:
        ImagePositionPatient=[0,0,0]
        PixelSpacing=[1,1]
        SliceThickness=1
        pixel_array=np.ones((4,4))
        RescaleSlope=1
        RescaleIntercept=0

    with patch("pydicom.dcmread", return_value=FakeDicom()):
        ingestor = COCAGatedIngestor(tmp_path)
        X, Y = ingestor.get_sample("0")

    report.info(f"get_sample: returned X with {X.shape[0]} slice(s) (expected 1)", "DAT-006")
    report.auto_save("dat006_get_sample_returns_single_slice", evidence_output_dir)
    assert not report.has_errors, report.summary()
    assert X.shape[0] == 1

@pytest.mark.requirement("DAT-004")
def test_get_sample_no_annotations_returns_empty(tmp_path, evidence_output_dir):

    report = EvidenceReport(subject="DAT-004: get_sample returns empty arrays when patient has no annotations")

    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)

    (patient_dir / "slice1.dcm").write_text("fake")

    class FakeDicom:
        ImagePositionPatient=[0,0,0]
        PixelSpacing=[1,1]
        SliceThickness=1
        pixel_array=np.ones((4,4))
        RescaleSlope=1
        RescaleIntercept=0

    with patch("pydicom.dcmread", return_value=FakeDicom()):
        ingestor = COCAGatedIngestor(tmp_path)
        X, Y = ingestor.get_sample("0")

    report.info(f"No annotations: X.shape[0]={X.shape[0]}, Y.shape[0]={Y.shape[0]} (both expected 0)", "DAT-004")
    report.auto_save("dat004_get_sample_no_annotations_returns_empty", evidence_output_dir)
    assert not report.has_errors, report.summary()
    assert X.shape[0] == 0
    assert Y.shape[0] == 0

@pytest.mark.requirement("DAT-004")
def test_get_sample_multiple_rois_same_slice(tmp_path, evidence_output_dir):

    report = EvidenceReport(subject="DAT-004: multiple ROIs on the same slice are merged into one mask")

    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)

    (patient_dir / "slice1.dcm").write_text("fake")

    xml_dir = tmp_path / "calcium_xml"
    xml_dir.mkdir()

    xml_file = xml_dir / "0.xml"

    plist_data = {
        "Images": [
            {
                "ImageIndex": 1,
                "ROIs": [
                    {
                        "Name":"calcium",
                        "NumberOfPoints":4,
                        "Point_px":["(0,0)","(1,0)","(1,1)","(0,1)"],
                    },
                    {
                        "Name":"calcium",
                        "NumberOfPoints":4,
                        "Point_px":["(2,2)","(3,2)","(3,3)","(2,3)"],
                    },
                ]
            }
        ]
    }

    with open(xml_file, "wb") as f:
        plistlib.dump(plist_data, f)

    class FakeDicom:
        ImagePositionPatient=[0,0,0]
        PixelSpacing=[1,1]
        SliceThickness=1
        pixel_array=np.ones((4,4))
        RescaleSlope=1
        RescaleIntercept=0

    with patch("pydicom.dcmread", return_value=FakeDicom()):
        ingestor = COCAGatedIngestor(tmp_path)
        X, Y = ingestor.get_sample("0")

    report.info(f"Two ROIs merged: X.shape[0]={X.shape[0]}, np.sum(Y)={np.sum(Y):.0f} (expected >2)", "DAT-004")
    report.auto_save("dat004_get_sample_multiple_rois_same_slice", evidence_output_dir)
    assert not report.has_errors, report.summary()
    assert X.shape[0] == 1
    assert np.sum(Y) > 2

@pytest.mark.requirement("DAT-007")
def test_get_sample_skips_invalid_slice_annotations(tmp_path, evidence_output_dir):

    report = EvidenceReport(subject="DAT-007: annotation with out-of-bounds ImageIndex is skipped; get_sample returns empty arrays")

    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)

    (patient_dir / "slice1.dcm").write_text("fake")

    xml_dir = tmp_path / "calcium_xml"
    xml_dir.mkdir()

    xml_file = xml_dir / "0.xml"

    plist_data = {
        "Images":[
            {
                "ImageIndex":10,
                "ROIs":[]
            }
        ]
    }

    with open(xml_file,"wb") as f:
        plistlib.dump(plist_data,f)

    class FakeDicom:
        ImagePositionPatient=[0,0,0]
        PixelSpacing=[1,1]
        SliceThickness=1
        pixel_array=np.ones((4,4))
        RescaleSlope=1
        RescaleIntercept=0

    with patch("pydicom.dcmread", return_value=FakeDicom()):
        ingestor = COCAGatedIngestor(tmp_path)
        X, Y = ingestor.get_sample("0")

    report.info(f"Out-of-bounds annotation skipped: X.shape[0]={X.shape[0]}, Y.shape[0]={Y.shape[0]} (both expected 0)", "DAT-007")
    report.auto_save("dat007_get_sample_skips_invalid_slice_annotations", evidence_output_dir)
    assert not report.has_errors, report.summary()
    assert X.shape[0] == 0
    assert Y.shape[0] == 0
