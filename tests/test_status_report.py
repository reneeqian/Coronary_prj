"""
Tests for Coronary_prj.reporting.status_report.

Each test writes minimal synthetic JSON artifacts into tmp_path and calls
status_report(project_root=tmp_path), capturing stdout with capsys to
verify that the expected fields appear in the output.
"""

from __future__ import annotations

import json

import pytest
from regulatory_tools.evidence.evidence_report import EvidenceReport

from Coronary_prj.reporting import status_report

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_training_run(root, name="20260427_100000_000000"):
    run_dir = root / "artifacts" / "training_runs" / name
    run_dir.mkdir(parents=True)
    data = {
        "metrics": {"final_loss": 0.0847, "epochs_trained": 3, "num_epochs": 50},
        "config": {
            "learning_rate": 0.001,
            "epochs": 50,
            "batch_size": 8,
            "device": "cpu",
            "optimizer": "Adam",
            "task": "CoronaryCalciumTask",
        },
        "datasource": {
            "train_ids": [f"P{i}" for i in range(42)],
            "val_ids": [f"V{i}" for i in range(9)],
            "test_ids": [f"T{i}" for i in range(87)],
        },
    }
    (run_dir / "training_report.json").write_text(json.dumps(data))
    return run_dir


def _write_tuning_run(root, name="20260427_183105"):
    run_dir = root / "artifacts" / "tuning_runs" / name
    run_dir.mkdir(parents=True)
    data = {
        "num_trials": 4,
        "best_trial": {
            "trial_id": 2,
            "params": {"learning_rate": 0.001, "base_channels": 32},
            "final_loss": 0.0847,
            "best_val_loss": 0.0901,
            "epochs_trained": 3,
            "early_stop_reason": None,
            "run_dir": str(run_dir / "trial_002"),
        },
        "trial_results": [],
    }
    (run_dir / "tuning_report.json").write_text(json.dumps(data))
    return run_dir


def _write_model_testing_run(root, name="20260427_190433_001000"):
    run_dir = root / "artifacts" / "model_testing_runs" / name
    run_dir.mkdir(parents=True)
    data = {
        "metrics": {
            "mean_loss": 0.0940,
            "num_test_patients": 87,
            "num_test_samples": 870,
            "tp": 500,
            "fp": 100,
            "fn": 80,
            "tn": 9320,
            "dice": 0.8470,
            "iou": 0.7350,
            "precision": 0.8910,
            "recall": 0.8080,
            "accuracy": 0.982,
        },
        "per_patient_results": [],
        "config": {},
    }
    (run_dir / "model_testing_report.json").write_text(json.dumps(data))
    return run_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.requirement("REP-004")
def test_status_report_handles_no_runs(tmp_path, evidence_output_dir, capsys):
    report = EvidenceReport(subject="status_report handles missing artifact directories")

    try:
        status_report(project_root=tmp_path)
    except Exception as e:
        report.error(f"status_report raised {type(e).__name__}: {e}", "REP-004")

    out = capsys.readouterr().out
    if "no runs found" not in out.lower():
        report.error("Expected 'no runs found' message when no artifacts exist", "REP-004")

    report.info(
        "status_report handles missing artifact directories without raising and prints 'no runs found'",
        "REP-004",
    )
    report.auto_save("REP004_status_report_no_runs", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("REP-003")
def test_status_report_prints_section_headers(tmp_path, evidence_output_dir, capsys):
    report = EvidenceReport(
        subject="REP-003: status_report produces TRAINING, TUNING, and MODEL TESTING section headers"
    )

    _write_training_run(tmp_path)
    _write_tuning_run(tmp_path)
    _write_model_testing_run(tmp_path)
    status_report(project_root=tmp_path)
    out = capsys.readouterr().out

    for header in ("TRAINING", "TUNING", "MODEL TESTING"):
        if header not in out:
            report.error(f"Section header '{header}' not found in status_report output", "REP-003")

    report.info(
        "status_report output contains TRAINING, TUNING, and MODEL TESTING section headers",
        "REP-003",
    )
    report.auto_save("REP003_status_report_section_headers", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("REP-004")
def test_status_report_prints_training_section(tmp_path, evidence_output_dir, capsys):
    report = EvidenceReport(
        subject="REP-004: status_report prints training metrics (final_loss, learning_rate)"
    )

    _write_training_run(tmp_path)
    status_report(project_root=tmp_path)
    out = capsys.readouterr().out

    for token, desc in [("0.0847", "final_loss value"), ("0.001", "learning_rate value")]:
        if token not in out:
            report.error(f"Expected '{token}' ({desc}) not found in output", "REP-004")

    report.info(
        "status_report printed TRAINING section with final_loss and learning_rate", "REP-004"
    )
    report.auto_save("REP004_status_report_training_section", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("REP-004")
def test_status_report_prints_tuning_section(tmp_path, evidence_output_dir, capsys):
    report = EvidenceReport(
        subject="REP-004: status_report prints tuning metrics (num_trials, best_val_loss)"
    )

    _write_tuning_run(tmp_path)
    status_report(project_root=tmp_path)
    out = capsys.readouterr().out

    for token, desc in [("4", "num_trials"), ("0.0901", "best_val_loss")]:
        if token not in out:
            report.error(f"Expected '{token}' ({desc}) not found in output", "REP-004")

    report.info("status_report printed TUNING section with num_trials and best_val_loss", "REP-004")
    report.auto_save("REP004_status_report_tuning_section", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("REP-004")
def test_status_report_prints_model_testing_section(tmp_path, evidence_output_dir, capsys):
    report = EvidenceReport(
        subject="REP-004: status_report prints model testing metrics (dice, iou, precision, recall)"
    )

    _write_model_testing_run(tmp_path)
    status_report(project_root=tmp_path)
    out = capsys.readouterr().out

    for token, desc in [
        ("MODEL TESTING", "MODEL TESTING section header"),
        ("0.8470", "dice value"),
        ("0.7350", "iou value"),
        ("0.8910", "precision value"),
        ("0.8080", "recall value"),
        ("87", "num_test_patients"),
    ]:
        if token not in out:
            report.error(f"Expected '{token}' ({desc}) not found in output", "REP-004")

    report.info(
        "status_report printed MODEL TESTING metrics: dice, iou, precision, recall, num_test_patients",
        "REP-004",
    )
    report.auto_save("REP004_status_report_model_testing_section", evidence_output_dir)
    assert not report.has_errors, report.summary()
