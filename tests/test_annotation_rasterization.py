import numpy as np
import pytest
from regulatory_tools.evidence.evidence_report import EvidenceReport

from Coronary_prj.ingestors.coca_gated_ingestor import COCAGatedIngestor, DatasetStructureError
from Coronary_prj.task_definitions.coronary_calcium_task import CoronaryCalciumTask


@pytest.mark.requirement("DAT-013")
def test_contour_is_rasterized_into_binary_mask(evidence_output_dir):
    report = EvidenceReport(subject="Annotation rasterization — polygon to mask")

    task = CoronaryCalciumTask()
    H, W = 16, 16

    # Square contour covering a known 4x4 region
    contour = np.array([[4, 4], [8, 4], [8, 8], [4, 8]], dtype=np.float32)
    rr, cc = task._polygon_to_mask(contour, H, W)

    if len(rr) == 0 or len(cc) == 0:
        report.error("Rasterized polygon produced no pixels", "DAT-013")
    else:
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[rr, cc] = 1

        # Centre pixel of the square must be foreground
        if mask[6, 6] != 1:
            report.error("Centre pixel inside contour is not foreground", "DAT-013")

        # Corner pixel outside the square must be background
        if mask[0, 0] != 0:
            report.error("Pixel outside contour is incorrectly marked foreground", "DAT-013")

        report.info(
            f"Polygon rasterized to {mask.sum()} foreground pixels; centre pixel=1, corner pixel=0",
            "DAT-013",
        )

    report.auto_save("DAT013_contour_rasterization", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DAT-013")
def test_rasterized_mask_aligns_with_source_dimensions(evidence_output_dir):
    report = EvidenceReport(subject="Annotation rasterization — mask shape matches source")

    task = CoronaryCalciumTask()

    for H, W in [(16, 16), (32, 64), (64, 32)]:
        contour = np.array([[2, 2], [H // 2, 2], [H // 2, W // 2], [2, W // 2]], dtype=np.float32)
        rr, cc = task._polygon_to_mask(contour, H, W)

        if len(rr) > 0 and (rr.max() >= H or cc.max() >= W):
            report.error(
                f"Rasterized pixels exceed image bounds for shape ({H}, {W}): "
                f"max_row={rr.max()}, max_col={cc.max()}",
                "DAT-013",
            )

    report.info(
        "Rasterized masks stay within bounds for shapes (16,16), (32,64), (64,32)", "DAT-013"
    )
    report.auto_save("DAT013_mask_dimension_alignment", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DAT-014")
def test_ingestor_raises_dataset_structure_error_not_runtime_error(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Domain-safe ingestion errors — DatasetStructureError")

    # Create a patient directory with a corrupted (non-DICOM) file
    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)
    (patient_dir / "bad.dcm").write_text("not a dicom file")

    ingestor = COCAGatedIngestor(dataset_root=tmp_path)

    raised_correct_type = False
    try:
        ingestor.load_patient_sample("0")
    except DatasetStructureError:
        raised_correct_type = True
    except Exception as e:
        report.error(
            f"Unexpected exception type {type(e).__name__} raised instead of "
            f"DatasetStructureError: {e}",
            "DAT-014",
        )

    if not raised_correct_type:
        report.error("No DatasetStructureError raised for a corrupted dataset", "DAT-014")

    report.info(
        "DatasetStructureError raised (not a raw exception) for corrupted DICOM input", "DAT-014"
    )
    report.auto_save("DAT014_domain_safe_ingestion_errors", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DAT-014")
def test_missing_patient_directory_raises_dataset_structure_error(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Domain-safe ingestion errors — missing patient directory")

    ingestor = COCAGatedIngestor(dataset_root=tmp_path)

    try:
        ingestor.load_patient_sample("nonexistent_patient")
        report.error("No exception raised for a nonexistent patient ID", "DAT-014")
    except DatasetStructureError:
        pass  # correct behaviour
    except Exception as e:
        report.error(f"Expected DatasetStructureError, got {type(e).__name__}: {e}", "DAT-014")

    report.info("DatasetStructureError raised for nonexistent patient directory", "DAT-014")
    report.auto_save("DAT014_missing_patient_directory", evidence_output_dir)
    assert not report.has_errors, report.summary()
