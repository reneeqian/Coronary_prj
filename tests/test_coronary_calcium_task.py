import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import patch

from Coronary_prj.ingestors.coca_gated_ingestor import COCAGatedIngestor
from Coronary_prj.ingestors.coca_gated_ingestor import DatasetStructureError
from Coronary_prj.task_definitions.coronary_calcium_task import CoronaryCalciumTask
from medical_image_ai_toolkit.dataobjects.annotation_bundle import AnnotationBundle, VectorROI
from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample
from regulatory_tools.evidence.evidence_report import EvidenceReport


class FakeReport:
    def __init__(self):
        self.warnings = []

    def warn(self, message, requirement_tag=None, context=None):
        self.warnings.append((message, requirement_tag, context))


@pytest.mark.requirement("DAT-004")
def test_coca_gated_ingestor_skips_dicom_without_image_positionpatient(tmp_path):
    patient_dir = tmp_path / "patient" / "0" / "seriesA"
    patient_dir.mkdir(parents=True)

    file_paths = [patient_dir / name for name in ["a.dcm", "b.dcm", "c.dcm"]]
    for path in file_paths:
        path.write_text("fake")

    class FakeDicomWithoutPosition:
        PixelSpacing = [1.0, 1.0]
        SliceThickness = 1.0
        RescaleSlope = 1.0
        RescaleIntercept = 0.0
        pixel_array = np.zeros((2, 2), dtype=np.float32)

    class FakeDicomWithPosition:
        def __init__(self, z, value):
            self.ImagePositionPatient = [0.0, 0.0, float(z)]
            self.PixelSpacing = [1.0, 1.0]
            self.SliceThickness = 1.0
            self.RescaleSlope = 1.0
            self.RescaleIntercept = 0.0
            self.pixel_array = np.full((2, 2), float(value), dtype=np.float32)

    metadata_map = {
        str(file_paths[0]): FakeDicomWithoutPosition(),
        str(file_paths[1]): FakeDicomWithPosition(10, 10),
        str(file_paths[2]): FakeDicomWithPosition(5, 5),
    }

    def fake_dcmread(path, *args, **kwargs):
        return metadata_map[str(path)]

    report = FakeReport()

    with patch("pydicom.dcmread", side_effect=fake_dcmread):
        ingestor = COCAGatedIngestor(dataset_root=tmp_path, report=report)
        sample = ingestor.load_patient_sample("0")

    assert sample.image_volume.shape == (2, 2, 2)
    assert np.all(sample.image_volume[0] == 5.0)
    assert np.all(sample.image_volume[1] == 10.0)
    assert len(report.warnings) == 1
    assert "Skipped" in report.warnings[0][0]


@pytest.mark.requirement("TSK-002")
def test_coronary_calcium_task_yields_masks_for_annotated_slices():
    task = CoronaryCalciumTask()
    volume = np.zeros((2, 4, 4), dtype=np.float32)
    volume[0] = np.arange(16, dtype=np.float32).reshape((4, 4))
    annotations = AnnotationBundle(
        vector_rois={
            0: [
                VectorROI(
                    slice_index=0,
                    contour_px=np.array(
                        [[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 3.0]],
                        dtype=np.float32,
                    ),
                    label="calcium",
                    metadata={"artery_name": "LAD"},
                )
            ]
        }
    )

    sample = PatientSample(
        patient_id="0",
        image_volume=volume,
        spacing=(1.0, 1.0, 1.0),
        annotations=annotations,
        metadata={}
    )

    outputs = list(task.generate_training_samples(sample))

    assert len(outputs) == 2
    assert outputs[0]["input"].shape == (1, 1, 4, 4)
    assert outputs[0]["target"].shape == (1, 1, 4, 4)
    assert outputs[1]["target"].sum().item() == 0.0
    assert outputs[0]["target"].sum().item() > 0.0


@pytest.mark.requirement("TSK-002")
def test_coronary_calcium_task_ignores_short_contours():
    task = CoronaryCalciumTask()
    volume = np.ones((1, 3, 3), dtype=np.float32)
    annotations = AnnotationBundle(
        vector_rois={
            0: [
                VectorROI(
                    slice_index=0,
                    contour_px=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
                    label="calcium",
                )
            ]
        }
    )

    sample = PatientSample(
        patient_id="0",
        image_volume=volume,
        spacing=(1.0, 1.0, 1.0),
        annotations=annotations,
        metadata={}
    )

    outputs = list(task.generate_training_samples(sample))

    assert len(outputs) == 1
    assert outputs[0]["target"].sum().item() == 0.0


@pytest.mark.requirement("TRN-003")
def test_coronary_calcium_task_compute_loss_returns_finite_scalar():
    task = CoronaryCalciumTask()
    prediction = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    target = torch.zeros((1, 1, 2, 2), dtype=torch.float32)

    loss = task.compute_loss(prediction, target)

    assert torch.isfinite(loss)
    assert loss.dim() == 0
    assert loss.item() > 0.0


@pytest.mark.requirement("TSK-002")
def test_coronary_calcium_task_input_is_hu_normalised(evidence_output_dir):
    report = EvidenceReport(subject="CoronaryCalciumTask HU window normalisation")

    task = CoronaryCalciumTask()
    volume = np.array(
        [
            np.full((4, 4), -2000.0, dtype=np.float32),
            np.full((4, 4), 3000.0, dtype=np.float32),
        ]
    )
    sample = PatientSample(
        patient_id="0",
        image_volume=volume,
        spacing=(1.0, 1.0, 1.0),
        annotations=None,
        metadata={},
    )

    outputs = list(task.generate_training_samples(sample))
    low_slice  = outputs[0]["input"]   # -2000 HU → clipped to -160 → normalised to -1.0
    high_slice = outputs[1]["input"]   # +3000 HU → clipped to +240 → normalised to +1.0

    if abs(low_slice.min().item() - (-1.0)) > 1e-4:
        report.error(
            f"Below-window HU not clamped correctly: got {low_slice.min().item():.4f}, expected -1.0",
            "TSK-002",
        )
    if abs(high_slice.max().item() - 1.0) > 1e-4:
        report.error(
            f"Above-window HU not clamped correctly: got {high_slice.max().item():.4f}, expected +1.0",
            "TSK-002",
        )

    report.auto_save("TSK002_hu_normalisation", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TRN-003")
def test_coronary_calcium_task_loss_near_zero_on_perfect_prediction(evidence_output_dir):
    report = EvidenceReport(subject="CoronaryCalciumTask combined loss near zero on perfect prediction")

    task = CoronaryCalciumTask()
    prediction = torch.full((1, 1, 4, 4), 10.0)   # sigmoid ≈ 1 — matches foreground
    target = torch.ones((1, 1, 4, 4))

    loss = task.compute_loss(prediction, target)

    if not torch.isfinite(loss):
        report.error("Combined loss is not finite on perfect prediction", "TRN-003")
    if loss.item() >= 0.1:
        report.error(
            f"Combined loss unexpectedly high on perfect prediction: {loss.item():.4f}",
            "TRN-003",
        )

    report.auto_save("TRN003_loss_perfect_prediction", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TRN-003")
def test_coronary_calcium_task_loss_penalises_wrong_predictions(evidence_output_dir):
    report = EvidenceReport(subject="CoronaryCalciumTask loss ordering: wrong > correct")

    task = CoronaryCalciumTask()
    target      = torch.ones((1, 1, 4, 4))
    correct_pred = torch.full((1, 1, 4, 4), 10.0)   # sigmoid ≈ 1 — matches target
    wrong_pred   = torch.full((1, 1, 4, 4), -10.0)  # sigmoid ≈ 0 — misses target

    loss_correct = task.compute_loss(correct_pred, target).item()
    loss_wrong   = task.compute_loss(wrong_pred, target).item()

    if loss_wrong <= loss_correct:
        report.error(
            f"Wrong prediction loss ({loss_wrong:.4f}) should exceed correct ({loss_correct:.4f})",
            "TRN-003",
        )

    report.auto_save("TRN003_loss_penalises_wrong_predictions", evidence_output_dir)
    assert not report.has_errors, report.summary()
