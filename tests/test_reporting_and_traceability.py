import json

import pytest
import torch
from medical_image_ai_toolkit.results.medical_image_training_results import (
    MedicalImageTrainingResults,
)
from medical_image_ai_toolkit.validation.segmentation_evaluator import SegmentationEvaluator
from medical_image_ai_toolkit.visualizations.model_testing_figures import plot_training_curve
from regulatory_tools.evidence.evidence_report import EvidenceReport
from regulatory_tools.traceability.generator import build_trace_matrix

from Coronary_prj.models.unet2d import UNet2D
from Coronary_prj.task_definitions.coronary_calcium_task import CoronaryCalciumTask


@pytest.mark.requirement("REP-001")
def test_training_report_generated_contains_metrics(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Training report generation")

    model = UNet2D(base_channels=4, depth=2)
    from medical_image_ai_toolkit.training.training_config import TrainingConfig

    config = TrainingConfig(task=CoronaryCalciumTask(), epochs=1, device="cpu")

    class _FakeDatasource:
        dataset_root = tmp_path
        partitions = {}

    results = MedicalImageTrainingResults(model, config, _FakeDatasource(), tmp_path)
    results.metrics = {"final_loss": 0.42, "epochs_trained": 1}
    results.history = [{"epoch": 1, "loss": 0.42}]

    report_path = results.generate_training_report()

    assert report_path.exists(), "training_report.json was not written"
    data = json.loads(report_path.read_text())
    assert "metrics" in data, "training_report.json missing 'metrics' key"
    assert data["metrics"].get("final_loss") == 0.42, (
        f"expected final_loss=0.42, got {data['metrics'].get('final_loss')}"
    )

    report.info(f"training_report.json at {report_path}: metrics.final_loss=0.42", "REP-001")
    report.auto_save("REP001_training_report_generation", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("REP-002")
def test_visualization_figures_can_be_generated(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Visualization support — training curve figure")

    history = [{"epoch": i + 1, "loss": 1.0 / (i + 1)} for i in range(5)]
    fig_path = tmp_path / "training_curve.png"

    result_path = plot_training_curve(history, fig_path)
    assert result_path.exists(), "plot_training_curve returned a path but no file was written"

    report.info(f"plot_training_curve wrote training curve figure to {result_path}", "REP-002")
    report.auto_save("REP002_visualization_support", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("VER-001")
def test_test_suite_exists_and_contains_test_functions(project_root, evidence_output_dir):
    report = EvidenceReport(subject="Automated verification execution — test suite existence")

    tests_dir = project_root / "tests"

    if not tests_dir.exists():
        report.error("tests/ directory does not exist", "VER-001")
    else:
        test_files = list(tests_dir.glob("test_*.py"))
        if not test_files:
            report.error("No test_*.py files found in tests/", "VER-001")
        else:
            has_test_fn = any("def test_" in f.read_text() for f in test_files)
            if not has_test_fn:
                report.error("No test functions (def test_*) found in any test file", "VER-001")
            else:
                report.info(
                    f"Test suite contains {len(test_files)} test_*.py files with at least one def test_* function",
                    "VER-001",
                )

    report.auto_save("VER001_test_suite_existence", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("VER-002")
def test_evidence_report_can_be_saved_and_loaded(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Evidence capture — auto_save produces loadable JSON")

    inner_report = EvidenceReport(subject="Synthetic test evidence")
    inner_report.info("test passed", requirement_tag="VER-002")
    inner_report.auto_save("synthetic_test_evidence", tmp_path)

    saved_files = list(tmp_path.glob("**/*.json"))
    if not saved_files:
        report.error("auto_save produced no JSON file", "VER-002")
    else:
        try:
            data = json.loads(saved_files[0].read_text())
            if "result" not in data:
                report.error("Saved evidence JSON missing 'result' key", "VER-002")
            else:
                report.info(
                    f"auto_save produced valid JSON evidence file with 'result' key at {saved_files[0]}",
                    "VER-002",
                )
        except json.JSONDecodeError as e:
            report.error(f"Saved evidence JSON is not valid JSON: {e}", "VER-002")

    report.auto_save("VER002_evidence_capture", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("VER-003")
def test_segmentation_evaluator_produces_performance_metrics(evidence_output_dir):
    report = EvidenceReport(subject="Model performance verification — segmentation metrics")

    evaluator = SegmentationEvaluator()

    pred = torch.zeros(1, 1, 8, 8)
    pred[0, 0, 2:6, 2:6] = 10.0
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 2:6, 2:6] = 1.0

    evaluator.update(pred, target)
    metrics = evaluator.aggregate()

    for key in ("dice", "iou", "precision", "recall"):
        if key not in metrics:
            report.error(f"Metric '{key}' missing from SegmentationEvaluator output", "VER-003")
        elif not (0.0 <= metrics[key] <= 1.0):
            report.error(f"Metric '{key}' = {metrics[key]} is outside [0, 1]", "VER-003")

    report.info(
        f"SegmentationEvaluator produced metrics: "
        f"dice={metrics.get('dice'):.4f}, iou={metrics.get('iou'):.4f}, "
        f"precision={metrics.get('precision'):.4f}, recall={metrics.get('recall'):.4f}",
        "VER-003",
    )
    report.auto_save("VER003_model_performance_verification", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("SYS-003")
def test_traceability_matrix_can_be_generated(project_root, evidence_output_dir):
    report = EvidenceReport(subject="Traceable verification — traceability matrix generation")

    requirements_yaml = project_root / "docs" / "requirements.yaml"
    evidence_root = project_root / "artifacts" / "evidence_runs"

    assert requirements_yaml.exists(), "docs/requirements.yaml not found"

    matrix = build_trace_matrix(requirements_yaml, evidence_root)
    assert matrix, "build_trace_matrix returned an empty matrix"

    report.info(
        f"build_trace_matrix returned {len(matrix)} rows from {requirements_yaml}", "SYS-003"
    )
    report.auto_save("SYS003_traceability_matrix_generation", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DOC-003")
def test_traceability_matrix_row_schema(project_root, evidence_output_dir):
    report = EvidenceReport(
        subject="Traceability matrix row schema — requirement_id field present in every row"
    )

    requirements_yaml = project_root / "docs" / "requirements.yaml"
    evidence_root = project_root / "artifacts" / "evidence_runs"

    matrix = build_trace_matrix(requirements_yaml, evidence_root)
    for row in matrix:
        assert "requirement_id" in row, f"Matrix row missing 'requirement_id' field: {row}"

    report.info(
        f"All {len(matrix)} traceability matrix rows contain 'requirement_id' field", "DOC-003"
    )
    report.auto_save("DOC003_traceability_matrix_row_schema", evidence_output_dir)
    assert not report.has_errors, report.summary()
