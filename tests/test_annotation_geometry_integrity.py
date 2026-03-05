import numpy as np
import pytest
from unittest.mock import patch
from pathlib import Path

from Coronary_prj.ingestors.coca_gated_ingestor import (
    COCAGatedIngestor,
    DatasetStructureError,
)

from medical_image_ai_toolkit.dataobjects.annotation_bundle import VectorROI


class SimpleDicom:
    ImagePositionPatient = [0, 0, 0]
    PixelSpacing = [1.0, 1.0]
    SliceThickness = 1.0
    RescaleSlope = 1.0
    RescaleIntercept = 0.0
    pixel_array = np.zeros((4, 4), dtype=np.float32)


@pytest.mark.requirement("DAT-009")
def test_valid_annotation_geometry(tmp_path):
    """
    Ensure parsed ROIs contain valid polygon geometry.
    """

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

    assert annotations.vector_rois is not None

    for slice_idx, rois in annotations.vector_rois.items():

        assert isinstance(slice_idx, int)

        for roi in rois:

            assert isinstance(roi, VectorROI)

            # geometry checks
            assert roi.contour_px.shape[1] == 2
            assert roi.contour_px.shape[0] >= 3
            assert roi.contour_px.dtype == np.float32

            assert not np.isnan(roi.contour_px).any()

@pytest.mark.requirement("DAT-009")
def test_invalid_polygon_skipped(tmp_path):

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

    with patch("pydicom.dcmread", return_value=SimpleDicom()):
        ingestor = COCAGatedIngestor(dataset_root=tmp_path)
        sample = ingestor.ingest_patient("0")

    assert sample.annotations.vector_rois is None
    
@pytest.mark.requirement("DAT-010")
def test_missing_annotation_file_returns_empty(tmp_path):

    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)

    (patient_dir / "a.dcm").write_text("fake")

    with patch("pydicom.dcmread", return_value=SimpleDicom()):
        ingestor = COCAGatedIngestor(dataset_root=tmp_path)
        sample = ingestor.ingest_patient("0")

    assert sample.annotations.vector_rois is None