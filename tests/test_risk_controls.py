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
def test_segmentation_threshold_constant_is_sensible():
    assert 0.0 < SEGMENTATION_MIN_DICE < 1.0


# ---------------------------------------------------------------------------
# RSK-002: negative calcium score clamped before log1p
# ---------------------------------------------------------------------------

@pytest.mark.requirement("RSK-002")
def test_negative_score_clamped_before_log1p():
    sample = _make_sample(metadata={"lca": -5.0, "lad": 0.0, "lcx": 0.0, "rca": 0.0})
    task = NongatedCalciumScoreTask()
    slices = list(task.generate_training_samples(sample))
    assert slices, "Expected at least one training sample"
    target = slices[0]["target"]
    assert torch.all(torch.isfinite(target)), "target contains non-finite values"
    assert float(target[0]) >= 0.0, "LCA target should be clamped to >= 0"


@pytest.mark.requirement("RSK-002")
def test_regression_threshold_constant_is_sensible():
    assert REGRESSION_MAX_MAE_AU > 0.0


# ---------------------------------------------------------------------------
# RSK-003: ingestor raises DatasetStructureError, not raw exception
# ---------------------------------------------------------------------------

@pytest.mark.requirement("RSK-003")
def test_nongated_ingestor_raises_dataset_structure_error_on_missing_xlsx(tmp_path):
    with pytest.raises(DatasetStructureError):
        COCANongatedIngestor(tmp_path)


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
    assert ood_warnings, "Expected OOD WARN for slices with extreme HU"
    report.auto_save("rsk003_ood_guard", evidence_output_dir)


# ---------------------------------------------------------------------------
# RSK-004: log1p target is always finite (no NaN from negative inputs)
# ---------------------------------------------------------------------------

@pytest.mark.requirement("RSK-004")
def test_log1p_target_is_always_finite():
    for scores in [
        {"lca": 0.0, "lad": 0.0, "lcx": 0.0, "rca": 0.0},
        {"lca": 500.0, "lad": 1200.0, "lcx": 0.0, "rca": 300.0},
        {"lca": -10.0, "lad": -50.0, "lcx": -1.0, "rca": -100.0},
    ]:
        sample = _make_sample(metadata=scores)
        task = NongatedCalciumScoreTask()
        for s in task.generate_training_samples(sample):
            assert torch.all(torch.isfinite(s["target"])), (
                f"Non-finite target for scores {scores}: {s['target']}"
            )
