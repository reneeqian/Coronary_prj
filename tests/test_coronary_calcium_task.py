from unittest.mock import patch

import numpy as np
import pytest
import torch
from medical_image_ai_toolkit.dataobjects.annotation_bundle import AnnotationBundle, VectorROI
from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample
from regulatory_tools.evidence.evidence_report import EvidenceReport

from Coronary_prj.ingestors.coca_gated_ingestor import COCAGatedIngestor
from Coronary_prj.task_definitions.coronary_calcium_task import CoronaryCalciumTask


class FakeReport:
    def __init__(self):
        self.warnings = []

    def warn(self, message, requirement_tag=None, context=None):
        self.warnings.append((message, requirement_tag, context))


@pytest.mark.requirement("DAT-004")
def test_coca_gated_ingestor_skips_dicom_without_image_positionpatient(tmp_path, evidence_output_dir):
    from regulatory_tools.evidence.evidence_report import EvidenceReport
    report = EvidenceReport(subject="DAT-004: DICOMs missing ImagePositionPatient are skipped with a warning")
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

    fake_report = FakeReport()

    with patch("pydicom.dcmread", side_effect=fake_dcmread):
        ingestor = COCAGatedIngestor(dataset_root=tmp_path, report=fake_report)
        sample = ingestor.load_patient_sample("0")

    if sample.image_volume.shape != (2, 2, 2):
        report.error(f"Expected volume shape (2,2,2), got {sample.image_volume.shape}", "DAT-004")
    if len(fake_report.warnings) != 1 or "Skipped" not in fake_report.warnings[0][0]:
        report.error("Expected exactly 1 'Skipped' warning from ingestor", "DAT-004")
    report.info(
        f"Volume shape={sample.image_volume.shape} with 1 skipped DICOM and 1 'Skipped' warning emitted",
        "DAT-004",
    )
    report.auto_save("DAT004_skip_dicom_without_position", evidence_output_dir)
    assert not report.has_errors, report.summary()
    assert sample.image_volume.shape == (2, 2, 2)
    assert np.all(sample.image_volume[0] == 5.0)
    assert np.all(sample.image_volume[1] == 10.0)
    assert len(fake_report.warnings) == 1
    assert "Skipped" in fake_report.warnings[0][0]


def _make_annotated_sample():
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
    return PatientSample(
        patient_id="0",
        image_volume=volume,
        spacing=(1.0, 1.0, 1.0),
        annotations=annotations,
        metadata={},
    )


@pytest.mark.requirement("TSK-001")
def test_coronary_calcium_task_yields_one_sample_per_slice(evidence_output_dir):
    from regulatory_tools.evidence.evidence_report import EvidenceReport
    report = EvidenceReport(subject="TSK-001: CoronaryCalciumTask yields one sample per CT slice")
    outputs = list(CoronaryCalciumTask().generate_training_samples(_make_annotated_sample()))

    if len(outputs) != 2:
        report.error(f"Expected 2 samples for 2-slice volume, got {len(outputs)}", "TSK-001")

    report.info(f"generate_training_samples yielded {len(outputs)} samples for a 2-slice volume", "TSK-001")
    report.auto_save("TSK001_task_yields_one_sample_per_slice", evidence_output_dir)
    assert not report.has_errors, report.summary()
    assert len(outputs) == 2


@pytest.mark.requirement("TSK-002")
def test_coronary_calcium_task_yields_masks_for_annotated_slices(evidence_output_dir):
    from regulatory_tools.evidence.evidence_report import EvidenceReport
    report = EvidenceReport(subject="TSK-002: CoronaryCalciumTask input/target shapes are (1,1,H,W); unannotated slices get zero target")
    outputs = list(CoronaryCalciumTask().generate_training_samples(_make_annotated_sample()))

    if outputs[0]["input"].shape != (1, 1, 4, 4):
        report.error(f"Input shape wrong: {outputs[0]['input'].shape}", "TSK-002")
    if outputs[0]["target"].shape != (1, 1, 4, 4):
        report.error(f"Target shape wrong: {outputs[0]['target'].shape}", "TSK-002")
    if outputs[1]["target"].sum().item() != 0.0:
        report.error("Unannotated slice target is non-zero", "TSK-002")

    report.info(
        f"input shape={outputs[0]['input'].shape}, target shape={outputs[0]['target'].shape}; unannotated slice target sum=0",
        "TSK-002",
    )
    report.auto_save("TSK002_task_yields_masks_for_annotated_slices", evidence_output_dir)
    assert not report.has_errors, report.summary()
    assert outputs[0]["input"].shape == (1, 1, 4, 4)
    assert outputs[0]["target"].shape == (1, 1, 4, 4)
    assert outputs[1]["target"].sum().item() == 0.0


@pytest.mark.requirement("DAT-013")
def test_coronary_calcium_task_rasterizes_contour_to_nonzero_mask(evidence_output_dir):
    from regulatory_tools.evidence.evidence_report import EvidenceReport
    report = EvidenceReport(subject="DAT-013: CoronaryCalciumTask rasterizes contour annotations to non-zero binary mask")
    outputs = list(CoronaryCalciumTask().generate_training_samples(_make_annotated_sample()))

    if outputs[0]["target"].sum().item() == 0.0:
        report.error("Annotated slice target is all-zero — rasterization failed", "DAT-013")

    report.info(f"Contour rasterized to mask with sum={outputs[0]['target'].sum().item():.1f} (non-zero)", "DAT-013")
    report.auto_save("DAT013_task_rasterizes_contour_to_nonzero_mask", evidence_output_dir)
    assert not report.has_errors, report.summary()
    assert outputs[0]["target"].sum().item() > 0.0


@pytest.mark.requirement("TSK-002")
def test_coronary_calcium_task_ignores_short_contours(evidence_output_dir):
    from regulatory_tools.evidence.evidence_report import EvidenceReport
    report = EvidenceReport(subject="CoronaryCalciumTask ignores contours with fewer than 3 points")
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

    if len(outputs) != 1:
        report.error(f"Expected 1 sample, got {len(outputs)}", "TSK-002")
    if outputs and outputs[0]["target"].sum().item() != 0.0:
        report.error("Short contour produced non-zero mask — should have been ignored", "TSK-002")
    report.info("2-point contour ignored; target is all-zero for the annotated slice", "TSK-002")
    report.auto_save("TSK002_ignore_short_contours", evidence_output_dir)
    assert not report.has_errors, report.summary()
    assert len(outputs) == 1
    assert outputs[0]["target"].sum().item() == 0.0


@pytest.mark.requirement("TRN-003")
def test_coronary_calcium_task_compute_loss_returns_finite_scalar(evidence_output_dir):
    from regulatory_tools.evidence.evidence_report import EvidenceReport
    report = EvidenceReport(subject="CoronaryCalciumTask compute_loss returns finite scalar")
    task = CoronaryCalciumTask()
    prediction = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    target = torch.zeros((1, 1, 2, 2), dtype=torch.float32)

    loss = task.compute_loss(prediction, target)

    if not torch.isfinite(loss):
        report.error("compute_loss returned non-finite value", "TRN-003")
    if loss.dim() != 0:
        report.error(f"compute_loss returned non-scalar (dim={loss.dim()})", "TRN-003")
    report.info(f"compute_loss returned finite scalar loss={loss.item():.4f}", "TRN-003")
    report.auto_save("TRN003_compute_loss_finite_scalar", evidence_output_dir)
    assert not report.has_errors, report.summary()
    assert torch.isfinite(loss)
    assert loss.dim() == 0
    assert loss.item() > 0.0


@pytest.mark.requirement("TSK-002")
def test_coronary_calcium_task_input_is_hu_normalised(evidence_output_dir):
    report = EvidenceReport(subject="TSK-002: CoronaryCalciumTask normalises input HU values to [-1, +1] range")

    task = CoronaryCalciumTask()
    volume = np.array([
        np.full((4, 4), -2000.0, dtype=np.float32),
        np.full((4, 4), 3000.0, dtype=np.float32),
    ])
    sample = PatientSample(patient_id="0", image_volume=volume, spacing=(1.0, 1.0, 1.0), annotations=None, metadata={})
    outputs = list(task.generate_training_samples(sample))
    low_val = outputs[0]["input"].min().item()
    high_val = outputs[1]["input"].max().item()

    if abs(low_val - (-1.0)) > 1e-4:
        report.error(f"Below-window HU not clamped to -1.0: got {low_val:.4f}", "TSK-002")
    if abs(high_val - 1.0) > 1e-4:
        report.error(f"Above-window HU not clamped to +1.0: got {high_val:.4f}", "TSK-002")

    report.info(f"HU window normalised: below-window min={low_val:.4f}, above-window max={high_val:.4f}", "TSK-002")
    report.auto_save("TSK002_task_input_hu_normalised", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TSK-004")
def test_coronary_calcium_task_applies_cardiac_hu_window(evidence_output_dir):
    report = EvidenceReport(subject="TSK-004: CoronaryCalciumTask applies the cardiac HU window [-160, 240]")

    task = CoronaryCalciumTask()
    volume = np.array([
        np.full((4, 4), -2000.0, dtype=np.float32),
        np.full((4, 4), 3000.0, dtype=np.float32),
    ])
    sample = PatientSample(patient_id="0", image_volume=volume, spacing=(1.0, 1.0, 1.0), annotations=None, metadata={})
    outputs = list(task.generate_training_samples(sample))
    low_val = outputs[0]["input"].min().item()
    high_val = outputs[1]["input"].max().item()

    if abs(low_val - (-1.0)) > 1e-4 or abs(high_val - 1.0) > 1e-4:
        report.error(f"Cardiac HU window not applied: low={low_val:.4f}, high={high_val:.4f}", "TSK-004")

    report.info(f"Cardiac HU window applied: -2000 HU → {low_val:.4f}, +3000 HU → {high_val:.4f}", "TSK-004")
    report.auto_save("TSK004_task_applies_cardiac_hu_window", evidence_output_dir)
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

    report.info(f"Combined loss on perfect prediction={loss.item():.4f} (finite, < 0.1)", "TRN-003")
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

    report.info(f"loss(wrong)={loss_wrong:.4f} > loss(correct)={loss_correct:.4f} — loss penalises errors", "TRN-003")
    report.auto_save("TRN003_loss_penalises_wrong_predictions", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TSK-006")
def test_gated_task_all_slices_receive_target_tensor(evidence_output_dir):
    report = EvidenceReport(
        subject="CoronaryCalciumTask broadcast design — every slice receives a target mask"
    )

    task = CoronaryCalciumTask()
    n_slices = 4
    volume = np.zeros((n_slices, 8, 8), dtype=np.float32)
    contour = np.array([[1.0, 1.0], [6.0, 1.0], [6.0, 6.0], [1.0, 6.0]], dtype=np.float32)
    roi = VectorROI(slice_index=1, contour_px=contour, label="calcium")
    sample = PatientSample(
        patient_id="P_tsk006",
        image_volume=volume,
        spacing=(1.0, 1.0, 1.0),
        annotations=AnnotationBundle(
            vector_rois={1: [roi]}, segmentation_masks=None, label_map={"calcium": 1}
        ),
        metadata={},
    )

    outputs = list(task.generate_training_samples(sample))

    if len(outputs) != n_slices:
        report.error(
            f"Expected {n_slices} samples (one per slice), got {len(outputs)}", "TSK-006"
        )
    else:
        for i, s in enumerate(outputs):
            if "target" not in s:
                report.error(f"Slice {i} has no 'target' key", "TSK-006")
            elif s["target"].shape[-2:] != (8, 8):
                report.error(f"Slice {i} target has wrong spatial shape: {s['target'].shape}", "TSK-006")

    report.info(
        f"All {n_slices} slices received a 'target' tensor with spatial shape (8,8)",
        "TSK-006",
    )
    report.auto_save("TSK006_gated_broadcast", evidence_output_dir)
    assert not report.has_errors, report.summary()
