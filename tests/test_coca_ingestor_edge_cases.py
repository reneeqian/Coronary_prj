from unittest.mock import patch

import numpy as np
import pytest
from regulatory_tools.evidence.evidence_report import EvidenceReport

from Coronary_prj.ingestors.coca_gated_ingestor import (
    COCAGatedIngestor,
    DatasetStructureError,
)

# =============================================================================
# Synthetic DICOM
# =============================================================================

class SimpleDicom:
    def __init__(self, z):
        self.ImagePositionPatient = [0.0, 0.0, float(z)]
        self.PixelSpacing = [1.0, 1.0]
        self.SliceThickness = 1.0
        self.RescaleSlope = 1.0
        self.RescaleIntercept = 0.0
        self.pixel_array = np.zeros((2, 2), dtype=np.float32)


# =============================================================================
# DAT-007 — Annotation slice index outside volume bounds
# =============================================================================

@pytest.mark.requirement("DAT-007")
def test_annotation_slice_out_of_bounds(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Annotation slice index outside volume is ignored")

    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)

    (patient_dir / "slice1.dcm").write_text("fake")

    xml_dir = tmp_path / "calcium_xml"
    xml_dir.mkdir()

    xml_file = xml_dir / "0.xml"

    xml_file.write_text(
        """<?xml version="1.0"?>
<plist version="1.0">
<dict>
    <key>Images</key>
    <array>
        <dict>
            <key>ImageIndex</key>
            <integer>99</integer>
            <key>ROIs</key>
            <array/>
        </dict>
    </array>
</dict>
</plist>
"""
    )

    with patch("pydicom.dcmread", return_value=SimpleDicom(0)):
        ingestor = COCAGatedIngestor(tmp_path)
        sample = ingestor.load_patient_sample("0")

    if sample.annotations.vector_rois is not None:
        report.error(
            "Expected vector_rois=None for out-of-bounds annotation slice, "
            f"got: {sample.annotations.vector_rois}",
            "DAT-007",
        )

    report.auto_save("DAT007_annotation_slice_out_of_bounds", evidence_output_dir)
    assert not report.has_errors, report.summary()


# =============================================================================
# DAT-009 — Invalid ROI geometry ignored (<3 points)
# =============================================================================

@pytest.mark.requirement("DAT-009")
def test_roi_with_insufficient_points_ignored(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="ROI with fewer than 3 points is ignored")

    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)

    (patient_dir / "slice1.dcm").write_text("fake")

    xml_dir = tmp_path / "calcium_xml"
    xml_dir.mkdir()

    xml_file = xml_dir / "0.xml"

    xml_file.write_text(
        """<?xml version="1.0"?>
<plist version="1.0">
<dict>
    <key>Images</key>
    <array>
        <dict>
            <key>ImageIndex</key>
            <integer>1</integer>
            <key>ROIs</key>
            <array>
                <dict>
                    <key>Name</key>
                    <string>LAD</string>
                    <key>NumberOfPoints</key>
                    <integer>2</integer>
                    <key>Point_px</key>
                    <array>
                        <string>(1,1)</string>
                        <string>(2,1)</string>
                    </array>
                </dict>
            </array>
        </dict>
    </array>
</dict>
</plist>
"""
    )

    with patch("pydicom.dcmread", return_value=SimpleDicom(0)):
        ingestor = COCAGatedIngestor(tmp_path)
        sample = ingestor.load_patient_sample("0")

    if sample.annotations.vector_rois is not None:
        report.error(
            "Expected vector_rois=None for sub-3-point ROI, "
            f"got: {sample.annotations.vector_rois}",
            "DAT-009",
        )

    report.auto_save("DAT009_roi_insufficient_points_ignored", evidence_output_dir)
    assert not report.has_errors, report.summary()


# =============================================================================
# DAT-010 — Dataset without annotation files
# =============================================================================

@pytest.mark.requirement("DAT-010")
def test_dataset_without_annotations(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Dataset with no annotation files returns empty annotations")

    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)

    (patient_dir / "slice1.dcm").write_text("fake")

    with patch("pydicom.dcmread", return_value=SimpleDicom(0)):
        ingestor = COCAGatedIngestor(tmp_path)
        sample = ingestor.load_patient_sample("0")

    if sample.annotations.vector_rois is not None:
        report.error(
            "Expected vector_rois=None when no annotation file exists, "
            f"got: {sample.annotations.vector_rois}",
            "DAT-010",
        )

    report.auto_save("DAT010_dataset_without_annotations", evidence_output_dir)
    assert not report.has_errors, report.summary()


# =============================================================================
# DAT-004 — Missing required DICOM metadata
# =============================================================================

@pytest.mark.requirement("DAT-004")
def test_missing_image_position_patient(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="DICOM missing ImagePositionPatient raises DatasetStructureError")

    class BadDicom:
        PixelSpacing = [1.0, 1.0]
        SliceThickness = 1.0

    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)

    (patient_dir / "slice1.dcm").write_text("fake")

    with patch("pydicom.dcmread", return_value=BadDicom()):
        ingestor = COCAGatedIngestor(tmp_path)

        try:
            ingestor.load_patient_sample("0")
            report.error("Expected DatasetStructureError; none raised", "DAT-004")
        except DatasetStructureError:
            report.info("DatasetStructureError raised as expected", "DAT-004")

    report.auto_save("DAT004_missing_image_position", evidence_output_dir)
    assert not report.has_errors, report.summary()


# =============================================================================
# DAT-004 — Successful slice retrieval with HU conversion
# =============================================================================

@pytest.mark.requirement("DAT-004")
def test_get_slice_success(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Slice retrieved with correct HU conversion")

    class GoodDicom:
        def __init__(self):
            self.ImagePositionPatient = [0, 0, 0]
            self.PixelSpacing = [1.0, 1.0]
            self.SliceThickness = 1.0
            self.pixel_array = np.ones((2, 2), dtype=np.float32)
            self.RescaleSlope = 2
            self.RescaleIntercept = 5

    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)

    (patient_dir / "slice1.dcm").write_text("fake")

    with patch("pydicom.dcmread", return_value=GoodDicom()):
        ingestor = COCAGatedIngestor(tmp_path)
        patient = ingestor.load_patient_sample("0")
        slice_img = patient.image_volume[0]

    if slice_img.shape != (2, 2):
        report.error(f"Slice shape wrong: {slice_img.shape}", "DAT-004")
    if not np.all(slice_img == 7):
        report.error(f"HU conversion wrong: expected 7, got {slice_img[0,0]}", "DAT-004")

    report.auto_save("DAT004_get_slice_success", evidence_output_dir)
    assert not report.has_errors, report.summary()


# =============================================================================
# DAT-005 — Missing gated series directory
# =============================================================================

@pytest.mark.requirement("DAT-005")
def test_no_series_directories(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Patient with no series directories raises DatasetStructureError")

    patient_dir = tmp_path / "patient" / "0"
    patient_dir.mkdir(parents=True)

    ingestor = COCAGatedIngestor(tmp_path)

    try:
        ingestor.load_patient_sample("0")
        report.error("Expected DatasetStructureError; none raised", "DAT-005")
    except DatasetStructureError:
        report.info("DatasetStructureError raised as expected", "DAT-005")

    report.auto_save("DAT005_no_series_directories", evidence_output_dir)
    assert not report.has_errors, report.summary()


# =============================================================================
# DAT-005 — No DICOM files
# =============================================================================

@pytest.mark.requirement("DAT-005")
def test_no_dicom_files(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Series with no DICOM files raises DatasetStructureError")

    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)

    ingestor = COCAGatedIngestor(tmp_path)

    try:
        ingestor.load_patient_sample("0")
        report.error("Expected DatasetStructureError; none raised", "DAT-005")
    except DatasetStructureError:
        report.info("DatasetStructureError raised as expected", "DAT-005")

    report.auto_save("DAT005_no_dicom_files", evidence_output_dir)
    assert not report.has_errors, report.summary()


# =============================================================================
# DAT-001 — Dataset structure validation
# =============================================================================

@pytest.mark.requirement("DAT-001")
def test_no_patient_directories(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Empty patient root raises DatasetStructureError")

    (tmp_path / "patient").mkdir()

    ingestor = COCAGatedIngestor(tmp_path)

    try:
        ingestor.list_patient_ids()
        report.error("Expected DatasetStructureError; none raised", "DAT-001")
    except DatasetStructureError:
        report.info("DatasetStructureError raised as expected", "DAT-001")

    report.auto_save("DAT001_no_patient_directories", evidence_output_dir)
    assert not report.has_errors, report.summary()


# =============================================================================
# DAT-002 — Deterministic slice retrieval
# =============================================================================

@pytest.mark.requirement("DAT-002")
def test_slice_determinism(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Repeated slice retrieval produces identical arrays")

    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)

    (patient_dir / "slice1.dcm").write_text("fake")

    with patch("pydicom.dcmread", return_value=SimpleDicom(0)):
        ingestor = COCAGatedIngestor(tmp_path)

        patient = ingestor.load_patient_sample("0")
        s1 = patient.image_volume[0]

        patient = ingestor.load_patient_sample("0")
        s2 = patient.image_volume[0]

    if not np.array_equal(s1, s2):
        report.error("Repeated load returned different slice arrays", "DAT-002")

    report.auto_save("DAT002_slice_determinism", evidence_output_dir)
    assert not report.has_errors, report.summary()
