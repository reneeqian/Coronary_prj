"""Tests for risk control requirements RSK-001 through RSK-004."""
import math

import numpy as np
import pytest
import torch

from medical_image_ai_toolkit.dataobjects.annotation_bundle import AnnotationBundle
from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample
from regulatory_tools.evidence.evidence_report import EvidenceReport

from Coronary_prj.ingestors.coca_gated_ingestor import DatasetStructureError
from Coronary_prj.ingestors.coca_nongated_ingestor import COCANongatedIngestor
from Coronary_prj.task_definitions.coronary_calcium_task import CoronaryCalciumTask
from Coronary_prj.task_definitions.nongated_calcium_score_task import NongatedCalciumScoreTask
from Coronary_prj.thresholds import REGRESSION_MAX_MAE_AU, SEGMENTATION_MIN_DICE


def _make_sample(pid="p1", shape=(5, 32, 32), spacing=(1.0, 1.0, 1.0), metadata=None):
    vol = np.zeros(shape, dtype=np.float32)
    return PatientSample(
        patient_id=pid,
        image_volume=vol,
        spacing=spacing,
        annotations=AnnotationBundle(vector_rois=None),
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# RSK-001: segmentation threshold constant is meaningful
# ---------------------------------------------------------------------------

@pytest.mark.requirement("RSK-001")
def test_segmentation_threshold_constant_is_sensible(evidence_output_dir):
    report = EvidenceReport(subject="RSK-001: SEGMENTATION_MIN_DICE is a meaningful threshold in (0, 1)")
    if not (0.0 < SEGMENTATION_MIN_DICE < 1.0):
        report.error(f"SEGMENTATION_MIN_DICE={SEGMENTATION_MIN_DICE} is not in (0.0, 1.0)", "RSK-001")
    report.info(f"SEGMENTATION_MIN_DICE={SEGMENTATION_MIN_DICE} is in (0.0, 1.0)", "RSK-001")
    report.auto_save("RSK001_segmentation_threshold", evidence_output_dir)
    assert not report.has_errors, report.summary()
    assert 0.0 < SEGMENTATION_MIN_DICE < 1.0


# ---------------------------------------------------------------------------
# RSK-002: negative calcium score clamped before log1p
# ---------------------------------------------------------------------------

@pytest.mark.requirement("RSK-002")
def test_negative_score_clamped_before_log1p(evidence_output_dir):
    report = EvidenceReport(subject="RSK-002: negative calcium scores are clamped before log1p transform")
    sample = _make_sample(metadata={"lca": -5.0, "lad": 0.0, "lcx": 0.0, "rca": 0.0})
    task = NongatedCalciumScoreTask()
    slices = list(task.generate_training_samples(sample))
    if not slices:
        report.error("No training samples produced", "RSK-002")
    else:
        target = slices[0]["target"]
        if not torch.all(torch.isfinite(target)):
            report.error("target contains non-finite values after clamping", "RSK-002")
        if float(target[0]) < 0.0:
            report.error(f"LCA target={float(target[0]):.4f} is negative — clamping failed", "RSK-002")
        report.info(f"LCA target={float(target[0]):.4f} after clamping negative input -5.0; all finite", "RSK-002")
    report.auto_save("RSK002_negative_score_clamped", evidence_output_dir)
    assert not report.has_errors, report.summary()
    assert slices
    assert torch.all(torch.isfinite(slices[0]["target"]))
    assert float(slices[0]["target"][0]) >= 0.0


@pytest.mark.requirement("RSK-002")
def test_regression_threshold_constant_is_sensible(evidence_output_dir):
    report = EvidenceReport(subject="RSK-002: REGRESSION_MAX_MAE_AU is a positive threshold")
    if not (REGRESSION_MAX_MAE_AU > 0.0):
        report.error(f"REGRESSION_MAX_MAE_AU={REGRESSION_MAX_MAE_AU} is not positive", "RSK-002")
    report.info(f"REGRESSION_MAX_MAE_AU={REGRESSION_MAX_MAE_AU} > 0.0", "RSK-002")
    report.auto_save("RSK002_regression_threshold", evidence_output_dir)
    assert not report.has_errors, report.summary()
    assert REGRESSION_MAX_MAE_AU > 0.0


# ---------------------------------------------------------------------------
# RSK-003: ingestor raises DatasetStructureError, not raw exception
# ---------------------------------------------------------------------------

@pytest.mark.requirement("RSK-003")
def test_nongated_ingestor_raises_dataset_structure_error_on_missing_xlsx(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="RSK-003: ingestor raises DatasetStructureError on missing scores.xlsx")
    raised = False
    try:
        COCANongatedIngestor(tmp_path)
    except DatasetStructureError:
        raised = True
    if not raised:
        report.error("Expected DatasetStructureError when scores.xlsx is absent; none raised", "RSK-003")
    report.info("DatasetStructureError raised for missing scores.xlsx — ingestor fails safely", "RSK-003")
    report.auto_save("RSK003_missing_xlsx_error", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("RSK-003")
def test_ood_guard_warns_on_extreme_hu(evidence_output_dir):
    report = EvidenceReport(subject="OOD guard test")
    task = CoronaryCalciumTask(report=report)
    # Volume with mean HU far above the 400 threshold (e.g. all-900 = metal/bone artefact)
    vol = np.full((3, 32, 32), 900.0, dtype=np.float32)
    sample = PatientSample(
        patient_id="ood_test",
        image_volume=vol,
        spacing=(1.0, 1.0, 1.0),
        annotations=AnnotationBundle(vector_rois=None),
    )
    list(task.generate_training_samples(sample))
    ood_warnings = [i for i in report.issues if i.level == "WARN" and "OOD" in i.message]
    if not ood_warnings:
        report.error("Expected OOD WARN for slices with extreme HU; none found", "RSK-003")
    else:
        report.info(f"OOD guard issued {len(ood_warnings)} WARN(s) for extreme HU input (mean≫400)", "RSK-003")
    report.auto_save("rsk003_ood_guard", evidence_output_dir)
    assert ood_warnings, "Expected OOD WARN for slices with extreme HU"


# ---------------------------------------------------------------------------
# RSK-004: log1p target is always finite (no NaN from negative inputs)
# ---------------------------------------------------------------------------

@pytest.mark.requirement("RSK-004")
def test_log1p_target_is_always_finite(evidence_output_dir):
    report = EvidenceReport(subject="RSK-004: log1p target is finite for zero, large, and negative inputs")
    score_cases = [
        {"lca": 0.0, "lad": 0.0, "lcx": 0.0, "rca": 0.0},
        {"lca": 500.0, "lad": 1200.0, "lcx": 0.0, "rca": 300.0},
        {"lca": -10.0, "lad": -50.0, "lcx": -1.0, "rca": -100.0},
    ]
    for scores in score_cases:
        sample = _make_sample(metadata=scores)
        task = NongatedCalciumScoreTask()
        for s in task.generate_training_samples(sample):
            if not torch.all(torch.isfinite(s["target"])):
                report.error(f"Non-finite target for scores {scores}: {s['target']}", "RSK-004")
    report.info(f"log1p targets are finite for all {len(score_cases)} score cases including negatives", "RSK-004")
    report.auto_save("RSK004_log1p_finite", evidence_output_dir)
    assert not report.has_errors, report.summary()
    for scores in score_cases:
        sample = _make_sample(metadata=scores)
        task = NongatedCalciumScoreTask()
        for s in task.generate_training_samples(sample):
            assert torch.all(torch.isfinite(s["target"])), (
                f"Non-finite target for scores {scores}: {s['target']}"
            )
