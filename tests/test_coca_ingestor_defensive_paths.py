from unittest.mock import patch

import numpy as np
import pytest
from regulatory_tools.evidence.evidence_report import EvidenceReport

from Coronary_prj.ingestors.base_ingestor import BaseIngestor
from Coronary_prj.ingestors.coca_gated_ingestor import (
    COCAGatedIngestor,
    DatasetStructureError,
)


# ---------------------------------------------------------------------------
# BaseIngestor abstract method contract (DAT-001)
# ---------------------------------------------------------------------------


class _CallSuperIngestor(BaseIngestor):
    """Minimal concrete subclass that delegates all methods to super() to
    exercise the NotImplementedError bodies of the abstract methods."""

    def list_patient_ids(self):
        return super().list_patient_ids()

    def load_patient_sample(self, patient_id: str):
        return super().load_patient_sample(patient_id)

    def get_sample(self, patient_id: str):
        return super().get_sample(patient_id)


@pytest.mark.requirement("DAT-001")
def test_base_ingestor_abstract_methods_raise(evidence_output_dir):
    report = EvidenceReport(
        subject="BaseIngestor abstract method bodies raise NotImplementedError"
    )
    ingestor = _CallSuperIngestor()

    for method_name, args in [
        ("list_patient_ids", []),
        ("load_patient_sample", ["any"]),
        ("get_sample", ["any"]),
    ]:
        try:
            getattr(ingestor, method_name)(*args)
            report.error(
                f"{method_name} did not raise NotImplementedError", "DAT-001"
            )
        except NotImplementedError:
            report.info(
                f"{method_name} raises NotImplementedError as expected", "DAT-001"
            )

    report.auto_save("DAT001_base_ingestor_abstract_methods", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DAT-001")
def test_missing_patient_root(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Missing patient root raises DatasetStructureError")

    ingestor = COCAGatedIngestor(tmp_path)

    try:
        ingestor.list_patient_ids()
        report.error("Expected DatasetStructureError; none raised", "DAT-001")
    except DatasetStructureError:
        report.info("DatasetStructureError raised as expected", "DAT-001")

    report.auto_save("DAT001_missing_patient_root", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DAT-005")
def test_patient_without_series(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Patient directory without series raises DatasetStructureError")

    patient_dir = tmp_path / "patient" / "0"
    patient_dir.mkdir(parents=True)

    ingestor = COCAGatedIngestor(tmp_path)

    try:
        ingestor.ingest_patient("0")
        report.error("Expected DatasetStructureError; none raised", "DAT-005")
    except DatasetStructureError:
        report.info("DatasetStructureError raised as expected", "DAT-005")

    report.auto_save("DAT005_patient_without_series", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DAT-005")
def test_series_without_dicoms(tmp_path, evidence_output_dir):
    report = EvidenceReport(
        subject="Series directory with no DICOM files raises DatasetStructureError"
    )

    series = tmp_path / "patient" / "0" / "seriesA"
    series.mkdir(parents=True)

    ingestor = COCAGatedIngestor(tmp_path)

    try:
        ingestor.ingest_patient("0")
        report.error("Expected DatasetStructureError; none raised", "DAT-005")
    except DatasetStructureError:
        report.info("DatasetStructureError raised as expected", "DAT-005")

    report.auto_save("DAT005_series_without_dicoms", evidence_output_dir)
    assert not report.has_errors, report.summary()


class BadDicom:
    PixelSpacing = [1, 1]
    SliceThickness = 1
    pixel_array = np.zeros((2, 2))


@pytest.mark.requirement("DAT-005")
def test_missing_image_position_patient(tmp_path, evidence_output_dir):
    report = EvidenceReport(
        subject="DICOM missing ImagePositionPatient raises DatasetStructureError"
    )

    series = tmp_path / "patient" / "0" / "seriesA"
    series.mkdir(parents=True)

    (series / "a.dcm").write_text("fake")

    with patch("pydicom.dcmread", return_value=BadDicom()):
        ingestor = COCAGatedIngestor(tmp_path)

        try:
            ingestor.ingest_patient("0")
            report.error("Expected DatasetStructureError; none raised", "DAT-005")
        except DatasetStructureError:
            report.info("DatasetStructureError raised as expected", "DAT-005")

    report.auto_save("DAT005_missing_image_position_patient", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DAT-005")
def test_annotation_xml_parse_failure(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Malformed annotation XML raises DatasetStructureError")

    series = tmp_path / "patient" / "0" / "seriesA"
    series.mkdir(parents=True)

    (series / "a.dcm").write_text("fake")

    xml_dir = tmp_path / "calcium_xml"
    xml_dir.mkdir()

    xml_file = xml_dir / "0.xml"
    xml_file.write_text("THIS IS NOT XML")

    class SimpleDicom:
        ImagePositionPatient = [0, 0, 0]
        PixelSpacing = [1, 1]
        SliceThickness = 1
        pixel_array = np.zeros((2, 2))

    with patch("pydicom.dcmread", return_value=SimpleDicom()):
        ingestor = COCAGatedIngestor(tmp_path)

        try:
            ingestor.ingest_patient("0")
            report.error("Expected DatasetStructureError; none raised", "DAT-005")
        except DatasetStructureError:
            report.info("DatasetStructureError raised as expected", "DAT-005")

    report.auto_save("DAT005_annotation_xml_parse_failure", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DAT-005")
def test_invalid_dicom_file(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Invalid DICOM file raises DatasetStructureError")

    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)

    (patient_dir / "slice1.dcm").write_text("fake")

    from pydicom.errors import InvalidDicomError

    with patch("pydicom.dcmread", side_effect=InvalidDicomError()):
        ingestor = COCAGatedIngestor(tmp_path)

        try:
            ingestor.load_patient_sample("0")
            report.error("Expected DatasetStructureError; none raised", "DAT-005")
        except DatasetStructureError:
            report.info("DatasetStructureError raised as expected", "DAT-005")

    report.auto_save("DAT005_invalid_dicom_file", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DAT-001")
def test_list_patient_ids_os_error(tmp_path, evidence_output_dir):
    report = EvidenceReport(
        subject="list_patient_ids wraps OSError into DatasetStructureError"
    )
    # Place a regular FILE at the 'patient/' path so that iterdir()
    # raises NotADirectoryError (a subclass of OSError).
    (tmp_path / "patient").write_text("not a directory")
    ingestor = COCAGatedIngestor(tmp_path)

    try:
        ingestor.list_patient_ids()
        report.error("Expected DatasetStructureError; none raised", "DAT-001")
    except DatasetStructureError:
        report.info("OSError wrapped into DatasetStructureError as expected", "DAT-001")

    report.auto_save("DAT001_list_patient_ids_os_error", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DAT-002")
def test_load_patient_sample_unexpected_exception(tmp_path, evidence_output_dir):
    report = EvidenceReport(
        subject="load_patient_sample wraps unexpected exceptions into DatasetStructureError"
    )
    patient_dir = tmp_path / "patient" / "P1"
    patient_dir.mkdir(parents=True)
    ingestor = COCAGatedIngestor(tmp_path)

    with patch.object(
        COCAGatedIngestor,
        "_resolve_gated_series_dir",
        side_effect=ValueError("unexpected internal error"),
    ):
        try:
            ingestor.load_patient_sample("P1")
            report.error("Expected DatasetStructureError; none raised", "DAT-002")
        except DatasetStructureError:
            report.info(
                "Unexpected exception wrapped into DatasetStructureError as expected",
                "DAT-002",
            )

    report.auto_save("DAT002_load_unexpected_exception", evidence_output_dir)
    assert not report.has_errors, report.summary()
