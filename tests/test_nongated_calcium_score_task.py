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
def test_yields_one_sample_per_slice(evidence_output_dir):
    """generate_training_samples yields exactly one sample per slice."""
    report = EvidenceReport(subject="NongatedCalciumScoreTask yields one sample per slice")
    task = NongatedCalciumScoreTask()
    patient = _make_patient(n_slices=5)
    samples = list(task.generate_training_samples(patient))
    if len(samples) != 5:
        report.error(f"Expected 5 samples, got {len(samples)}", "TSK-005")
    report.info(f"generate_training_samples yielded {len(samples)} samples for 5-slice patient", "TSK-005")
    report.auto_save("TSK005_yields_one_per_slice", evidence_output_dir)
    assert not report.has_errors, report.summary()
    assert len(samples) == 5


@pytest.mark.requirement("TSK-005")
def test_input_tensor_shape(evidence_output_dir):
    """Each yielded input has shape (1, 1, H, W)."""
    report = EvidenceReport(subject="NongatedCalciumScoreTask input tensor shape is (1,1,H,W)")
    task = NongatedCalciumScoreTask()
    patient = _make_patient(n_slices=2, H=8, W=8)
    samples = list(task.generate_training_samples(patient))
    for i, s in enumerate(samples):
        if s["input"].shape != (1, 1, 8, 8):
            report.error(f"Sample {i} input shape {s['input'].shape} != (1,1,8,8)", "TSK-005")
    report.info(f"All {len(samples)} samples have input shape (1,1,8,8)", "TSK-005")
    report.auto_save("TSK005_input_tensor_shape", evidence_output_dir)
    assert not report.has_errors, report.summary()
    for s in samples:
        assert s["input"].shape == (1, 1, 8, 8)


@pytest.mark.requirement("TSK-005")
def test_target_tensor_shape(evidence_output_dir):
    """Each yielded target has shape (4,) — one value per vessel."""
    report = EvidenceReport(subject="NongatedCalciumScoreTask target tensor shape is (4,)")
    task = NongatedCalciumScoreTask()
    patient = _make_patient(n_slices=3)
    samples = list(task.generate_training_samples(patient))
    for i, s in enumerate(samples):
        if s["target"].shape != (4,):
            report.error(f"Sample {i} target shape {s['target'].shape} != (4,)", "TSK-005")
    report.info(f"All {len(samples)} samples have target shape (4,) — one per vessel", "TSK-005")
    report.auto_save("TSK005_target_tensor_shape", evidence_output_dir)
    assert not report.has_errors, report.summary()
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

    report.info(
        f"Target values match log1p(scores): lca={target[0]:.4f}, lad={target[1]:.4f}, "
        f"lcx={target[2]:.4f}, rca={target[3]:.4f}",
        "TSK-005",
    )
    report.auto_save("TSK005_log1p_target", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TSK-005")
@pytest.mark.requirement("TSK-006")
def test_target_broadcast_identical_across_slices(evidence_output_dir):
    """All slices from the same patient share the identical target vector (TSK-006 broadcast design)."""
    report = EvidenceReport(subject="NongatedCalciumScoreTask — patient label broadcast to every slice")

    task = NongatedCalciumScoreTask()
    patient = _make_patient(n_slices=4, lca=10.0, lad=20.0)
    samples = list(task.generate_training_samples(patient))
    first = samples[0]["target"]
    for i, s in enumerate(samples[1:], start=1):
        if not torch.equal(s["target"], first):
            report.error(f"Slice {i} target differs from slice 0", "TSK-006")

    report.info(f"All {len(samples)} slices carry identical target vector (broadcast design)", "TSK-006")
    report.info(f"generate_training_samples produced {len(samples)} samples from 4-slice patient", "TSK-005")
    report.auto_save("TSK006_nongated_broadcast", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TSK-005")
def test_zero_score_target_is_zero(evidence_output_dir):
    """log1p(0) == 0 — patients with no calcium produce a zero target."""
    report = EvidenceReport(subject="NongatedCalciumScoreTask zero-calcium patient yields zero target")
    task = NongatedCalciumScoreTask()
    patient = _make_patient(n_slices=1, lca=0.0, lad=0.0, lcx=0.0, rca=0.0)
    samples = list(task.generate_training_samples(patient))
    if not torch.all(samples[0]["target"] == 0.0):
        report.error(f"Expected all-zero target for zero scores, got {samples[0]['target']}", "TSK-005")
    report.info("log1p(0)=0 — zero-calcium patient produces all-zero target vector", "TSK-005")
    report.auto_save("TSK005_zero_score_target", evidence_output_dir)
    assert not report.has_errors, report.summary()
    assert torch.all(samples[0]["target"] == 0.0)


@pytest.mark.requirement("TSK-005")
def test_missing_metadata_defaults_to_zero_score(evidence_output_dir):
    """PatientSample with no score metadata yields a zero target without raising."""
    report = EvidenceReport(subject="NongatedCalciumScoreTask missing metadata defaults to zero score")
    task = NongatedCalciumScoreTask()
    patient = PatientSample(
        patient_id="x",
        image_volume=np.zeros((2, 4, 4), dtype=np.float32),
        spacing=(1.0, 1.0, 1.0),
        annotations=AnnotationBundle(vector_rois=None),
        metadata={},   # no score keys
    )
    samples = list(task.generate_training_samples(patient))
    if len(samples) != 2:
        report.error(f"Expected 2 samples, got {len(samples)}", "TSK-005")
    if not torch.all(samples[0]["target"] == 0.0):
        report.error("Missing metadata did not default to zero target", "TSK-005")
    report.info("PatientSample with no score metadata yields zero target without raising", "TSK-005")
    report.auto_save("TSK005_missing_metadata_zero_score", evidence_output_dir)
    assert not report.has_errors, report.summary()
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

    report.info(f"Below-window HU (-2000) clamped to -1.0 after normalisation; got {val:.4f}", "TSK-005")
    report.info("Cardiac HU window applied by NongatedCalciumScoreTask", "TSK-004")
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

    report.info(f"Above-window HU (+3000) clamped to +1.0 after normalisation; got {val:.4f}", "TSK-005")
    report.info("Cardiac HU window applied by NongatedCalciumScoreTask", "TSK-004")
    report.auto_save("TSK005_hu_above_window", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TSK-005")
def test_input_wl40_maps_to_zero(evidence_output_dir):
    """WL=40 HU (centre of cardiac window) normalises to 0.0."""
    report = EvidenceReport(subject="NongatedCalciumScoreTask WL=40 HU normalises to 0.0")
    task = NongatedCalciumScoreTask()
    patient = _make_patient(n_slices=1, hu_value=40.0)
    samples = list(task.generate_training_samples(patient))
    mean_val = samples[0]["input"].mean().item()
    if abs(mean_val) >= 1e-5:
        report.error(f"Expected WL=40 HU to normalise to 0.0, got {mean_val:.6f}", "TSK-005")
    report.info(f"WL=40 HU (window centre) normalised to mean={mean_val:.6f} ≈ 0.0", "TSK-005")
    report.auto_save("TSK005_wl40_maps_to_zero", evidence_output_dir)
    assert not report.has_errors, report.summary()
    assert abs(mean_val) < 1e-5


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

    report.info(f"compute_loss returned finite scalar loss={loss.item():.4f}", "TSK-005")
    report.info("MSE loss is finite and scalar for arbitrary pred/target", "TRN-003")
    report.auto_save("TSK005_loss_finite_scalar", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TSK-005")
def test_compute_loss_zero_on_perfect_prediction(evidence_output_dir):
    """MSE loss is (near) zero when prediction exactly matches target."""
    report = EvidenceReport(subject="NongatedCalciumScoreTask MSE loss near zero on perfect prediction")
    task = NongatedCalciumScoreTask()
    pred = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    loss = task.compute_loss(pred, target)
    if loss.item() >= 1e-6:
        report.error(f"Expected loss≈0 on perfect prediction, got {loss.item():.2e}", "TSK-005")
    report.info(f"MSE loss on perfect prediction={loss.item():.2e} (near zero)", "TSK-005")
    report.auto_save("TSK005_loss_zero_on_perfect", evidence_output_dir)
    assert not report.has_errors, report.summary()
    assert loss.item() < 1e-6


@pytest.mark.requirement("TSK-005")
def test_compute_loss_higher_for_wrong_prediction(evidence_output_dir):
    """Loss is larger when prediction is further from target."""
    report = EvidenceReport(subject="NongatedCalciumScoreTask MSE loss is monotone in error magnitude")
    task = NongatedCalciumScoreTask()
    target = torch.tensor([2.0, 2.0, 2.0, 2.0])
    close_pred = torch.tensor([[2.1, 2.1, 2.1, 2.1]])
    far_pred   = torch.tensor([[10.0, 10.0, 10.0, 10.0]])

    loss_close = task.compute_loss(close_pred, target)
    loss_far   = task.compute_loss(far_pred,   target)

    if not (loss_far.item() > loss_close.item()):
        report.error(
            f"loss(far)={loss_far.item():.4f} should exceed loss(close)={loss_close.item():.4f}",
            "TSK-005",
        )
    report.info(
        f"loss(far)={loss_far.item():.4f} > loss(close)={loss_close.item():.4f} — loss is monotone in error",
        "TSK-005",
    )
    report.auto_save("TSK005_loss_higher_for_wrong", evidence_output_dir)
    assert not report.has_errors, report.summary()
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

    report.info("Gradient flowed from compute_loss back to model output (pred.grad is not None)", "TRN-003")
    report.auto_save("TRN003_nongated_gradient_flow", evidence_output_dir)
    assert not report.has_errors, report.summary()
