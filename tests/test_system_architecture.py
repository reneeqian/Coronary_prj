import numpy as np
import pytest
import torch
from medical_image_ai_toolkit.dataobjects.annotation_bundle import AnnotationBundle, VectorROI
from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample
from medical_image_ai_toolkit.training.task_definition import TrainingTaskDefinition
from regulatory_tools.evidence.evidence_report import EvidenceReport

import Coronary_prj
from Coronary_prj.ingestors.base_ingestor import BaseIngestor
from Coronary_prj.ingestors.coca_gated_ingestor import COCAGatedIngestor
from Coronary_prj.task_definitions.coronary_calcium_task import CoronaryCalciumTask


def _make_patient_sample() -> PatientSample:
    rng = np.random.default_rng(0)
    volume = rng.uniform(-160, 240, size=(3, 32, 32)).astype(np.float32)
    contour = np.array([[4, 4], [16, 4], [16, 16], [4, 16]], dtype=np.float32)
    roi = VectorROI(slice_index=1, contour_px=contour, label="lesion")
    return PatientSample(
        image_volume=volume,
        spacing=(1.5, 0.7, 0.7),
        annotations=AnnotationBundle(
            vector_rois={1: [roi]}, segmentation_masks=None, label_map={"lesion": 1}
        ),
        patient_id="P0",
        metadata={},
    )


@pytest.mark.requirement("SYS-006")
def test_ingestor_and_task_are_in_project_not_toolkit(evidence_output_dir):
    report = EvidenceReport(subject="Dataset task encapsulation — project-level ownership")

    if not CoronaryCalciumTask.__module__.startswith("Coronary_prj"):
        report.error(
            f"CoronaryCalciumTask lives in '{CoronaryCalciumTask.__module__}', "
            "expected 'Coronary_prj.*'",
            "SYS-006",
        )
    if not COCAGatedIngestor.__module__.startswith("Coronary_prj"):
        report.error(
            f"COCAGatedIngestor lives in '{COCAGatedIngestor.__module__}', "
            "expected 'Coronary_prj.*'",
            "SYS-006",
        )

    report.info(
        f"CoronaryCalciumTask module={CoronaryCalciumTask.__module__}; "
        f"COCAGatedIngestor module={COCAGatedIngestor.__module__}",
        "SYS-006",
    )
    report.auto_save("SYS006_task_encapsulation", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TSK-001")
def test_coronary_task_implements_toolkit_interface(evidence_output_dir):
    report = EvidenceReport(subject="Task definition interface — toolkit contract")

    if not issubclass(CoronaryCalciumTask, TrainingTaskDefinition):
        report.error("CoronaryCalciumTask does not subclass TrainingTaskDefinition", "TSK-001")

    task = CoronaryCalciumTask()
    for method in ("generate_training_samples", "compute_loss"):
        if not callable(getattr(task, method, None)):
            report.error(f"CoronaryCalciumTask missing required method '{method}'", "TSK-001")

    report.info(
        "CoronaryCalciumTask subclasses TrainingTaskDefinition and exposes generate_training_samples + compute_loss",
        "TSK-001",
    )
    report.auto_save("TSK001_task_definition_interface", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TSK-003")
def test_task_output_is_deterministic_for_same_input(evidence_output_dir):
    report = EvidenceReport(subject="Task determinism — identical inputs produce identical outputs")

    task = CoronaryCalciumTask()
    sample = _make_patient_sample()

    run_a = list(task.generate_training_samples(sample))
    run_b = list(task.generate_training_samples(sample))

    if len(run_a) != len(run_b):
        report.error(
            f"Different number of samples produced: {len(run_a)} vs {len(run_b)}", "TSK-003"
        )
    else:
        for i, (sa, sb) in enumerate(zip(run_a, run_b, strict=True)):
            if not torch.equal(sa["input"], sb["input"]):
                report.error(f"Sample {i} 'input' differs between runs", "TSK-003")
                break
            if not torch.equal(sa["target"], sb["target"]):
                report.error(f"Sample {i} 'target' differs between runs", "TSK-003")
                break

    report.info(
        f"Two generate_training_samples() runs produced {len(run_a)} identical samples", "TSK-003"
    )
    report.auto_save("TSK003_task_determinism", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("SYS-004")
def test_coronary_calcium_task_is_instantiable(evidence_output_dir):
    report = EvidenceReport(
        subject="SYS-004: CoronaryCalciumTask can be instantiated without error"
    )

    task = CoronaryCalciumTask()

    report.info(f"CoronaryCalciumTask() instantiated: {type(task).__name__}", "SYS-004")
    report.auto_save("SYS004_coronary_calcium_task_is_instantiable", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("SYS-005")
def test_coca_gated_ingestor_subclasses_base_ingestor(evidence_output_dir):
    report = EvidenceReport(subject="SYS-005: COCAGatedIngestor subclasses BaseIngestor")

    if not issubclass(COCAGatedIngestor, BaseIngestor):
        report.error("COCAGatedIngestor does not subclass BaseIngestor", "SYS-005")

    report.info("COCAGatedIngestor is a subclass of BaseIngestor", "SYS-005")
    report.auto_save("SYS005_coca_gated_ingestor_subclasses_base_ingestor", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("SYS-007")
def test_intended_use_statement_is_advisory(evidence_output_dir):
    report = EvidenceReport(subject="Intended use — advisory, radiologist-facing")

    intended_use = getattr(Coronary_prj, "INTENDED_USE", None)
    if intended_use is None:
        report.error("Coronary_prj.INTENDED_USE is not defined", "SYS-007")
    else:
        text = intended_use.lower()
        if "advisory" not in text:
            report.error("INTENDED_USE does not contain 'advisory'", "SYS-007")
        if "radiologist" not in text:
            report.error("INTENDED_USE does not contain 'radiologist'", "SYS-007")

    report.info(
        "INTENDED_USE contains 'advisory' and 'radiologist' — output is non-diagnostic", "SYS-007"
    )
    report.auto_save("SYS007_intended_use", evidence_output_dir)
    assert not report.has_errors, report.summary()
