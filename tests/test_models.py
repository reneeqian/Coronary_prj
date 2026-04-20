import pytest
import torch

from regulatory_tools.evidence.evidence_report import EvidenceReport
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
