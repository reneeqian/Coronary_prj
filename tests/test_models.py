import pytest
import torch
from regulatory_tools.evidence.evidence_report import EvidenceReport

from Coronary_prj.models.calcium_score_regressor import CalciumScoreRegressor
from Coronary_prj.models.small_segmentation_cnn import SmallSegmentationCNN
from Coronary_prj.models.unet2d import UNet2D


@pytest.mark.requirement("MOD-001")
def test_small_segmentation_cnn_output_shape(evidence_output_dir):
    report = EvidenceReport(subject="SmallSegmentationCNN output shape")

    model = SmallSegmentationCNN()
    model.eval()
    x = torch.zeros(1, 1, 64, 64)

    with torch.no_grad():
        out = model(x)

    if out.shape != x.shape:
        report.error(f"Expected output shape {x.shape}, got {out.shape}", "MOD-001")

    report.auto_save("MOD001_small_cnn_output_shape", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("MOD-001")
def test_unet2d_output_shape_matches_input(evidence_output_dir):
    report = EvidenceReport(subject="UNet2D output shape preserves spatial dimensions")

    model = UNet2D(base_channels=8, depth=2)
    model.eval()
    x = torch.zeros(1, 1, 64, 64)

    with torch.no_grad():
        out = model(x)

    if out.shape != x.shape:
        report.error(f"Expected output shape {x.shape}, got {out.shape}", "MOD-001")

    report.auto_save("MOD001_unet2d_output_shape", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("MOD-001")
def test_unet2d_configurable_channels(evidence_output_dir):
    report = EvidenceReport(subject="UNet2D base_channels configuration")

    for base_ch in (8, 16, 32):
        model = UNet2D(base_channels=base_ch, depth=2)
        model.eval()
        x = torch.zeros(1, 1, 32, 32)
        with torch.no_grad():
            out = model(x)
        if out.shape != x.shape:
            report.error(
                f"UNet2D(base_channels={base_ch}) output shape mismatch: {out.shape}",
                "MOD-001",
            )

    report.auto_save("MOD001_unet2d_configurable_channels", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("MOD-001")
def test_unet2d_produces_finite_outputs(evidence_output_dir):
    report = EvidenceReport(subject="UNet2D outputs are finite on random input")

    model = UNet2D(base_channels=8, depth=2)
    model.eval()
    x = torch.randn(1, 1, 64, 64)

    with torch.no_grad():
        out = model(x)

    if not torch.isfinite(out).all():
        report.error("UNet2D produced non-finite output on random input", "MOD-001")

    report.auto_save("MOD001_unet2d_finite_outputs", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("MOD-001")
def test_unet2d_gradient_flows_through_network(evidence_output_dir):
    report = EvidenceReport(subject="UNet2D gradient flow verification")

    model = UNet2D(base_channels=8, depth=2)
    x = torch.randn(1, 1, 64, 64)
    out = model(x)
    loss = out.mean()
    loss.backward()

    no_grad_params = [
        name for name, p in model.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    if no_grad_params:
        report.error(
            f"Parameters received no gradient: {no_grad_params}", "MOD-001"
        )

    report.auto_save("MOD001_unet2d_gradient_flow", evidence_output_dir)
    assert not report.has_errors, report.summary()


# =============================================================================
# CalciumScoreRegressor tests (MOD-004)
# =============================================================================

@pytest.mark.requirement("MOD-004")
def test_calcium_score_regressor_output_shape(evidence_output_dir):
    """CalciumScoreRegressor output shape is (B, 4) for default num_outputs=4."""
    report = EvidenceReport(subject="CalciumScoreRegressor output shape")

    model = CalciumScoreRegressor(base_channels=8)
    model.eval()
    x = torch.zeros(1, 1, 64, 64)

    with torch.no_grad():
        out = model(x)

    if out.shape != (1, 4):
        report.error(f"Expected (1, 4), got {out.shape}", "MOD-004")

    report.auto_save("MOD004_regressor_output_shape", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("MOD-004")
def test_calcium_score_regressor_accepts_any_spatial_size(evidence_output_dir):
    """AdaptiveAvgPool2d allows the model to accept any H×W without error."""
    report = EvidenceReport(subject="CalciumScoreRegressor spatial size flexibility")

    model = CalciumScoreRegressor(base_channels=8)
    model.eval()

    for size in (32, 64, 128, 512):
        x = torch.zeros(1, 1, size, size)
        with torch.no_grad():
            out = model(x)
        if out.shape != (1, 4):
            report.error(f"Size {size}×{size}: expected (1,4), got {out.shape}", "MOD-004")

    report.auto_save("MOD004_regressor_spatial_flexibility", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("MOD-004")
def test_calcium_score_regressor_finite_output_on_random_input(evidence_output_dir):
    """CalciumScoreRegressor outputs finite values on random input."""
    report = EvidenceReport(subject="CalciumScoreRegressor finite outputs")

    model = CalciumScoreRegressor(base_channels=8)
    model.eval()
    x = torch.randn(2, 1, 64, 64)

    with torch.no_grad():
        out = model(x)

    if not torch.isfinite(out).all():
        report.error("Non-finite output on random input", "MOD-004")

    report.auto_save("MOD004_regressor_finite_output", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("MOD-004")
def test_calcium_score_regressor_gradient_flows(evidence_output_dir):
    """Gradients flow to all trainable parameters of CalciumScoreRegressor."""
    report = EvidenceReport(subject="CalciumScoreRegressor gradient flow")

    model = CalciumScoreRegressor(base_channels=8)
    x = torch.randn(1, 1, 32, 32)
    out = model(x)
    out.mean().backward()

    no_grad = [
        name for name, p in model.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    if no_grad:
        report.error(f"Parameters received no gradient: {no_grad}", "MOD-004")

    report.auto_save("MOD004_regressor_gradient_flow", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("MOD-004")
def test_calcium_score_regressor_configurable_outputs():
    """num_outputs parameter controls the output vector length."""
    for n in (1, 4, 5):
        model = CalciumScoreRegressor(base_channels=8, num_outputs=n)
        model.eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 1, 32, 32))
        assert out.shape == (1, n), f"num_outputs={n}: got {out.shape}"


@pytest.mark.requirement("MOD-004")
def test_calcium_score_regressor_configurable_base_channels():
    """base_channels parameter scales the model capacity without changing output shape."""
    for c in (8, 16, 32):
        model = CalciumScoreRegressor(base_channels=c)
        model.eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 1, 32, 32))
        assert out.shape == (1, 4), f"base_channels={c}: got {out.shape}"
