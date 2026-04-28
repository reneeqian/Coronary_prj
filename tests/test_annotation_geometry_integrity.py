import plistlib
from unittest.mock import patch

import numpy as np
import pytest
from medical_image_ai_toolkit.dataobjects.annotation_bundle import VectorROI
from regulatory_tools.evidence.evidence_report import EvidenceReport

from Coronary_prj.ingestors.coca_gated_ingestor import (
    COCAGatedIngestor,
)


class SimpleDicom:
    ImagePositionPatient = [0, 0, 0]
    PixelSpacing = [1.0, 1.0]
    SliceThickness = 1.0
    RescaleSlope = 1.0
    RescaleIntercept = 0.0
    pixel_array = np.zeros((4, 4), dtype=np.float32)


@pytest.mark.requirement("DAT-009")
def test_valid_annotation_geometry(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Valid annotation geometry: polygon shape and dtype")

    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)

    (patient_dir / "a.dcm").write_text("fake")

    xml_dir = tmp_path / "calcium_xml"
    xml_dir.mkdir()

    xml_file = xml_dir / "0.xml"

    xml_file.write_text(
    """<?xml version="1.0" encoding="UTF-8"?>
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
                        <integer>4</integer>
                        <key>Point_px</key>
                        <array>
                            <string>(1,1)</string>
                            <string>(2,1)</string>
                            <string>(2,2)</string>
                            <string>(1,2)</string>
                        </array>
                    </dict>
                </array>
            </dict>
        </array>
    </dict>
    </plist>
    """
    )

    with patch("pydicom.dcmread", return_value=SimpleDicom()):
        ingestor = COCAGatedIngestor(dataset_root=tmp_path)
        sample = ingestor.ingest_patient("0")

    annotations = sample.annotations

    if annotations.vector_rois is None:
        report.error("vector_rois is None; expected parsed ROIs", "DAT-009")
    else:
        for slice_idx, rois in annotations.vector_rois.items():
            if not isinstance(slice_idx, int):
                report.error(f"slice_idx is not int: {type(slice_idx)}", "DAT-009")
            for roi in rois:
                if not isinstance(roi, VectorROI):
                    report.error(f"ROI is not VectorROI: {type(roi)}", "DAT-009")
                if roi.contour_px.shape[1] != 2:
                    report.error(f"contour_px has wrong column count: {roi.contour_px.shape}", "DAT-009")
                if roi.contour_px.shape[0] < 3:
                    report.error(f"contour_px has fewer than 3 points: {roi.contour_px.shape}", "DAT-009")
                if roi.contour_px.dtype != np.float32:
                    report.error(f"contour_px dtype wrong: {roi.contour_px.dtype}", "DAT-009")
                if np.isnan(roi.contour_px).any():
                    report.error("contour_px contains NaN values", "DAT-009")

    report.auto_save("DAT009_valid_annotation_geometry", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DAT-009")
def test_invalid_polygon_skipped(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="ROI with <3 points is skipped")

    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)

    (patient_dir / "a.dcm").write_text("fake")

    xml_dir = tmp_path / "calcium_xml"
    xml_dir.mkdir()

    xml_file = xml_dir / "0.xml"

    plist_data = {
        "Images": [
            {
                "ImageIndex": 1,
                "ROIs": [
                    {
                        "Name": "LAD",
                        "NumberOfPoints": 2,
                        "Point_px": [
                            "(1,1)",
                            "(2,1)"
                        ]
                    }
                ]
            }
        ]
    }

    with open(xml_file, "wb") as f:
        plistlib.dump(plist_data, f)

    with patch("pydicom.dcmread", return_value=SimpleDicom()):
        ingestor = COCAGatedIngestor(dataset_root=tmp_path)
        sample = ingestor.ingest_patient("0")

    if sample.annotations.vector_rois is not None:
        report.error(
            "Expected vector_rois=None for sub-3-point polygon, "
            f"got: {sample.annotations.vector_rois}",
            "DAT-009",
        )

    report.auto_save("DAT009_invalid_polygon_skipped", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DAT-010")
def test_missing_annotation_file_returns_empty(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Missing annotation file returns empty annotations")

    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)

    (patient_dir / "a.dcm").write_text("fake")

    with patch("pydicom.dcmread", return_value=SimpleDicom()):
        ingestor = COCAGatedIngestor(dataset_root=tmp_path)
        sample = ingestor.ingest_patient("0")

    if sample.annotations.vector_rois is not None:
        report.error(
            "Expected vector_rois=None when annotation file absent, "
            f"got: {sample.annotations.vector_rois}",
            "DAT-010",
        )

    report.auto_save("DAT010_missing_annotation_file_empty", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("SYS-001")
def test_volume_sorted_by_z_position(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Volume slices are sorted by Z position")

    class FakeDicom:
        def __init__(self, z):
            self.ImagePositionPatient = [0, 0, z]
            self.PixelSpacing = [1, 1]
            self.SliceThickness = 1
            self.pixel_array = np.full((2, 2), z)
            self.RescaleSlope = 1
            self.RescaleIntercept = 0

    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)

    (patient_dir / "a.dcm").write_text("fake")
    (patient_dir / "b.dcm").write_text("fake")
    (patient_dir / "c.dcm").write_text("fake")

    fake_dicoms = {
        "a.dcm": FakeDicom(5),
        "b.dcm": FakeDicom(1),
        "c.dcm": FakeDicom(3),
    }

    def fake_dcmread(path, *args, **kwargs):
        return fake_dicoms[path.name]

    with patch("pydicom.dcmread", side_effect=fake_dcmread):
        ingestor = COCAGatedIngestor(tmp_path)
        patient = ingestor.load_patient_sample("0")
        vol = patient.image_volume

    z_vals = list(vol[:, 0, 0])
    if z_vals != [1, 3, 5]:
        report.error(f"Volume not sorted by Z: got {z_vals}, expected [1, 3, 5]", "SYS-001")

    report.auto_save("SYS001_volume_sorted_by_z", evidence_output_dir)
    assert not report.has_errors, report.summary()
