"""
Tests for NongatedCalciumScoreTask.

Verifies that generate_training_samples yields the correct number of samples,
that inputs are HU-normalised identically to CoronaryCalciumTask, that targets
are log1p-transformed per-vessel scores broadcast across all slices, and that
compute_loss behaves correctly for regression.
"""
import math

import numpy as np
import pytest
import torch
from medical_image_ai_toolkit.dataobjects.annotation_bundle import AnnotationBundle
from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample
from regulatory_tools.evidence.evidence_report import EvidenceReport

from Coronary_prj.task_definitions.nongated_calcium_score_task import NongatedCalciumScoreTask


def _make_patient(
    n_slices: int = 3,
    H: int = 4,
    W: int = 4,
    lca: float = 0.0,
    lad: float = 0.0,
    lcx: float = 0.0,
    rca: float = 0.0,
    hu_value: float = 40.0,
) -> PatientSample:
    volume = np.full((n_slices, H, W), hu_value, dtype=np.float32)
    return PatientSample(
        patient_id="test",
        image_volume=volume,
        spacing=(1.0, 1.0, 1.0),
        annotations=AnnotationBundle(vector_rois=None),
        metadata={"lca": lca, "lad": lad, "lcx": lcx, "rca": rca},
    )


# ---------------------------------------------------------------------------
# Sample generation (TSK-005)
# ---------------------------------------------------------------------------

@pytest.mark.requirement("TSK-005")
def test_yields_one_sample_per_slice():
    """generate_training_samples yields exactly one sample per slice."""
    task = NongatedCalciumScoreTask()
    patient = _make_patient(n_slices=5)
    samples = list(task.generate_training_samples(patient))
    assert len(samples) == 5


@pytest.mark.requirement("TSK-005")
def test_input_tensor_shape():
    """Each yielded input has shape (1, 1, H, W)."""
    task = NongatedCalciumScoreTask()
    patient = _make_patient(n_slices=2, H=8, W=8)
    samples = list(task.generate_training_samples(patient))
    for s in samples:
        assert s["input"].shape == (1, 1, 8, 8)


@pytest.mark.requirement("TSK-005")
def test_target_tensor_shape():
    """Each yielded target has shape (4,) — one value per vessel."""
    task = NongatedCalciumScoreTask()
    patient = _make_patient(n_slices=3)
    samples = list(task.generate_training_samples(patient))
    for s in samples:
        assert s["target"].shape == (4,)


@pytest.mark.requirement("TSK-005")
def test_target_is_log1p_of_vessel_scores(evidence_output_dir):
    """Target values equal log1p(lca), log1p(lad), log1p(lcx), log1p(rca)."""
    report = EvidenceReport(subject="NongatedCalciumScoreTask target log1p transform")

    lca, lad, lcx, rca = 132.0, 1.91, 128.0, 37.5
    task = NongatedCalciumScoreTask()
    patient = _make_patient(n_slices=1, lca=lca, lad=lad, lcx=lcx, rca=rca)
    samples = list(task.generate_training_samples(patient))

    target = samples[0]["target"]
    expected = [math.log1p(v) for v in (lca, lad, lcx, rca)]

    for i, (got, exp) in enumerate(zip(target.tolist(), expected)):
        if abs(got - exp) > 1e-5:
            report.error(f"Vessel {i}: expected log1p={exp:.6f}, got {got:.6f}", "TSK-005")

    report.auto_save("TSK005_log1p_target", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TSK-005")
def test_target_broadcast_identical_across_slices():
    """All slices from the same patient share the identical target vector."""
    task = NongatedCalciumScoreTask()
    patient = _make_patient(n_slices=4, lca=10.0, lad=20.0)
    samples = list(task.generate_training_samples(patient))
    first = samples[0]["target"]
    for s in samples[1:]:
        assert torch.equal(s["target"], first)


@pytest.mark.requirement("TSK-005")
def test_zero_score_target_is_zero():
    """log1p(0) == 0 — patients with no calcium produce a zero target."""
    task = NongatedCalciumScoreTask()
    patient = _make_patient(n_slices=1, lca=0.0, lad=0.0, lcx=0.0, rca=0.0)
    samples = list(task.generate_training_samples(patient))
    assert torch.all(samples[0]["target"] == 0.0)


@pytest.mark.requirement("TSK-005")
def test_missing_metadata_defaults_to_zero_score():
    """PatientSample with no score metadata yields a zero target without raising."""
    task = NongatedCalciumScoreTask()
    patient = PatientSample(
        patient_id="x",
        image_volume=np.zeros((2, 4, 4), dtype=np.float32),
        spacing=(1.0, 1.0, 1.0),
        annotations=AnnotationBundle(vector_rois=None),
        metadata={},   # no score keys
    )
    samples = list(task.generate_training_samples(patient))
    assert len(samples) == 2
    assert torch.all(samples[0]["target"] == 0.0)


# ---------------------------------------------------------------------------
# HU normalisation (TSK-005, TSK-004)
# ---------------------------------------------------------------------------

@pytest.mark.requirement("TSK-005")
@pytest.mark.requirement("TSK-004")
def test_input_hu_normalised_below_window(evidence_output_dir):
    """Values below the cardiac window (-160 HU) are clamped to -1.0 after normalisation."""
    report = EvidenceReport(subject="NongatedCalciumScoreTask HU window — below")

    task = NongatedCalciumScoreTask()
    patient = _make_patient(n_slices=1, hu_value=-2000.0)
    samples = list(task.generate_training_samples(patient))
    val = samples[0]["input"].min().item()
    if abs(val - (-1.0)) > 1e-4:
        report.error(f"Expected -1.0 for below-window HU, got {val:.4f}", "TSK-005")

    report.auto_save("TSK005_hu_below_window", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TSK-005")
@pytest.mark.requirement("TSK-004")
def test_input_hu_normalised_above_window(evidence_output_dir):
    """Values above the cardiac window (+240 HU) are clamped to +1.0 after normalisation."""
    report = EvidenceReport(subject="NongatedCalciumScoreTask HU window — above")

    task = NongatedCalciumScoreTask()
    patient = _make_patient(n_slices=1, hu_value=3000.0)
    samples = list(task.generate_training_samples(patient))
    val = samples[0]["input"].max().item()
    if abs(val - 1.0) > 1e-4:
        report.error(f"Expected +1.0 for above-window HU, got {val:.4f}", "TSK-005")

    report.auto_save("TSK005_hu_above_window", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TSK-005")
def test_input_wl40_maps_to_zero():
    """WL=40 HU (centre of cardiac window) normalises to 0.0."""
    task = NongatedCalciumScoreTask()
    patient = _make_patient(n_slices=1, hu_value=40.0)
    samples = list(task.generate_training_samples(patient))
    assert abs(samples[0]["input"].mean().item()) < 1e-5


# ---------------------------------------------------------------------------
# Loss function (TSK-005, TRN-003)
# ---------------------------------------------------------------------------

@pytest.mark.requirement("TSK-005")
@pytest.mark.requirement("TRN-003")
def test_compute_loss_is_finite_scalar(evidence_output_dir):
    """compute_loss returns a finite scalar for arbitrary pred/target."""
    report = EvidenceReport(subject="NongatedCalciumScoreTask MSE loss is finite")

    task = NongatedCalciumScoreTask()
    pred = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    target = torch.tensor([1.5, 2.5, 3.5, 4.5])
    loss = task.compute_loss(pred, target)

    if not torch.isfinite(loss):
        report.error("Loss is not finite", "TSK-005")
    if loss.dim() != 0:
        report.error(f"Loss should be scalar, got dim={loss.dim()}", "TSK-005")

    report.auto_save("TSK005_loss_finite_scalar", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TSK-005")
def test_compute_loss_zero_on_perfect_prediction():
    """MSE loss is (near) zero when prediction exactly matches target."""
    task = NongatedCalciumScoreTask()
    pred = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    loss = task.compute_loss(pred, target)
    assert loss.item() < 1e-6


@pytest.mark.requirement("TSK-005")
def test_compute_loss_higher_for_wrong_prediction():
    """Loss is larger when prediction is further from target."""
    task = NongatedCalciumScoreTask()
    target = torch.tensor([2.0, 2.0, 2.0, 2.0])
    close_pred = torch.tensor([[2.1, 2.1, 2.1, 2.1]])
    far_pred   = torch.tensor([[10.0, 10.0, 10.0, 10.0]])

    loss_close = task.compute_loss(close_pred, target)
    loss_far   = task.compute_loss(far_pred,   target)

    assert loss_far.item() > loss_close.item()


@pytest.mark.requirement("TRN-003")
def test_gradients_flow_through_loss(evidence_output_dir):
    """Gradients propagate through compute_loss back to the model output."""
    report = EvidenceReport(subject="NongatedCalciumScoreTask gradient flow")

    task = NongatedCalciumScoreTask()
    pred = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    target = torch.tensor([0.5, 1.5, 2.5, 3.5])
    loss = task.compute_loss(pred, target)
    loss.backward()

    if pred.grad is None:
        report.error("No gradient flowed back through compute_loss", "TRN-003")

    report.auto_save("TRN003_nongated_gradient_flow", evidence_output_dir)
    assert not report.has_errors, report.summary()
