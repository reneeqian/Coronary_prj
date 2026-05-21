import numpy as np
import pytest
import torch
from medical_image_ai_toolkit.dataobjects.annotation_bundle import AnnotationBundle, VectorROI
from medical_image_ai_toolkit.dataobjects.datasources.deterministic_split import (
    DeterministicHoldoutSplit,
)
from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import (
    MedicalImageDataSource,
)
from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample
from medical_image_ai_toolkit.training.medical_image_trainer import MedicalImageTrainer
from medical_image_ai_toolkit.training.training_config import TrainingConfig
from regulatory_tools.evidence.evidence_report import EvidenceReport

from Coronary_prj.models.unet2d import UNet2D
from Coronary_prj.task_definitions.coronary_calcium_task import CoronaryCalciumTask


class _SyntheticIngestor:
    def __init__(self, num_patients: int = 6) -> None:
        self.patient_ids = [f"P{i}" for i in range(num_patients)]

    def list_patient_ids(self):
        return self.patient_ids

    def load_patient_sample(self, patient_id: str) -> PatientSample:
        rng = np.random.default_rng(abs(hash(patient_id)) % (2**32))
        volume = rng.uniform(-160, 240, size=(3, 32, 32)).astype(np.float32)
        contour = np.array([[4, 4], [12, 4], [12, 12], [4, 12]], dtype=np.float32)
        roi = VectorROI(slice_index=1, contour_px=contour, label="lesion")
        annotations = AnnotationBundle(
            vector_rois={1: [roi]}, segmentation_masks=None, label_map={"lesion": 1}
        )
        return PatientSample(
            image_volume=volume,
            spacing=(1.5, 0.7, 0.7),
            annotations=annotations,
            patient_id=patient_id,
            metadata={},
        )

    def get_sample(self, patient_id: str):
        sample = self.load_patient_sample(patient_id)
        return sample.image_volume[0][None, None], np.array([[1.0]])


def _make_datasource(tmp_path, num_patients: int = 6):
    ds = MedicalImageDataSource(
        dataset_root=tmp_path,
        ingestor=_SyntheticIngestor(num_patients=num_patients),
    )
    ds.create_partitions(DeterministicHoldoutSplit(train=0.6, val=0.2, seed=0))
    return ds


def _make_config() -> TrainingConfig:
    return TrainingConfig(
        task=CoronaryCalciumTask(),
        epochs=1,
        learning_rate=1e-3,
        device="cpu",
        early_stop=False,
    )


def _tiny_model() -> UNet2D:
    return UNet2D(base_channels=4, depth=2)


@pytest.mark.requirement("TRN-001")
def test_training_initialization_is_deterministic(evidence_output_dir):
    report = EvidenceReport(subject="Controlled model initialization — seed determinism")

    torch.manual_seed(42)
    model_a = _tiny_model()

    torch.manual_seed(42)
    model_b = _tiny_model()

    for name, pa in model_a.named_parameters():
        pb = dict(model_b.named_parameters())[name]
        if not torch.equal(pa.data, pb.data):
            report.error(
                f"Parameter '{name}' differs between models initialized with same seed",
                "TRN-001",
            )
            break

    report.info("All parameters identical between two models initialized with the same seed", "TRN-001")
    report.auto_save("TRN001_initialization_determinism", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TRN-002")
def test_training_artifacts_generated_after_training(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Training artifact generation")

    ds = _make_datasource(tmp_path)
    model = _tiny_model()
    config = _make_config()

    trainer = MedicalImageTrainer(ds, model, config, output_dir=tmp_path / "runs")
    results = trainer.train()

    if not (results.run_dir / "metrics.json").exists():
        report.error("metrics.json not found in run directory", "TRN-002")
    if "final_loss" not in results.metrics:
        report.error("metrics dict missing 'final_loss' key", "TRN-002")
    if not results.history:
        report.error("training history is empty", "TRN-002")

    report.info(f"metrics.json written; final_loss={results.metrics.get('final_loss'):.4f}; history present", "TRN-002")
    report.auto_save("TRN002_training_artifact_generation", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TRN-004")
def test_model_can_be_retrained_with_updated_config(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Coronary model retraining capability")

    ds = _make_datasource(tmp_path)
    model = _tiny_model()

    config_v1 = _make_config()
    trainer_v1 = MedicalImageTrainer(ds, model, config_v1, output_dir=tmp_path / "run1")
    trainer_v1.train()

    config_v2 = TrainingConfig(
        task=CoronaryCalciumTask(),
        epochs=1,
        learning_rate=5e-4,
        device="cpu",
        early_stop=False,
    )
    trainer_v2 = MedicalImageTrainer(ds, model, config_v2, output_dir=tmp_path / "run2")
    try:
        trainer_v2.train()
    except Exception as e:
        report.error(f"Retraining with updated config raised an exception: {e}", "TRN-004")

    report.info("Second training run with updated learning_rate completed without error", "TRN-004")
    report.auto_save("TRN004_model_retraining", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("MOD-002")
def test_model_state_dict_can_be_saved_and_loaded(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Model artifact persistence — save and reload")

    model = _tiny_model()
    model.eval()

    model_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), model_path)

    if not model_path.exists():
        report.error("model.pt not written to disk", "MOD-002")
    else:
        loaded_model = _tiny_model()
        loaded_model.load_state_dict(torch.load(model_path, weights_only=True))
        loaded_model.eval()

        x = torch.randn(1, 1, 32, 32)
        with torch.no_grad():
            out_orig = model(x)
            out_loaded = loaded_model(x)

        if not torch.allclose(out_orig, out_loaded, atol=1e-6):
            report.error("Loaded model produces different output than saved model", "MOD-002")

    report.info("Saved and loaded model produce identical outputs on same input", "MOD-002")
    report.auto_save("MOD002_model_persistence", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("MOD-003")
def test_model_evaluation_on_held_out_partition(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Model evaluation capability on test partition")

    from medical_image_ai_toolkit.pipeline.model_testing_pipeline import ModelTestingPipeline
    from medical_image_ai_toolkit.validation.segmentation_evaluator import SegmentationEvaluator

    ds = _make_datasource(tmp_path)
    model = _tiny_model()
    config = _make_config()

    trainer = MedicalImageTrainer(ds, model, config, output_dir=tmp_path / "runs")
    training_results = trainer.train()
    training_results.export_model(training_results.run_dir / "model.pt")
    training_results.artifacts["partitions"] = ds.save_partitions(training_results.run_dir)

    evaluator = SegmentationEvaluator()
    pipeline = ModelTestingPipeline(
        datasource=ds,
        model=model,
        config=config,
        evaluator=evaluator,
        output_dir=tmp_path / "testing",
    )
    testing_results = pipeline.run()

    for key in ("dice", "iou"):
        if key not in testing_results.metrics:
            report.error(f"Metric '{key}' missing from model testing results", "MOD-003")

    report.info(f"ModelTestingPipeline produced metrics: {sorted(testing_results.metrics.keys())}", "MOD-003")
    report.auto_save("MOD003_model_evaluation", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("INF-001")
def test_inference_produces_output_on_new_data(evidence_output_dir):
    report = EvidenceReport(subject="Inference capability on new data")

    model = _tiny_model()
    model.eval()

    x = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        out = model(x)

    if out.shape != x.shape:
        report.error(f"Expected output shape {x.shape}, got {out.shape}", "INF-001")
    if not torch.isfinite(out).all():
        report.error("Model output contains non-finite values", "INF-001")

    report.info(f"Inference output shape={out.shape}, all finite={torch.isfinite(out).all().item()}", "INF-001")
    report.auto_save("INF001_inference_capability", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("INF-002")
def test_inference_is_deterministic_for_same_input(evidence_output_dir):
    report = EvidenceReport(subject="Inference determinism — identical inputs produce identical outputs")

    model = _tiny_model()
    model.eval()

    x = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        out_a = model(x)
        out_b = model(x)

    if not torch.equal(out_a, out_b):
        report.error("Two inference runs on the same input produced different outputs", "INF-002")

    report.info("Two inference runs on identical input produced bit-identical outputs", "INF-002")
    report.auto_save("INF002_inference_determinism", evidence_output_dir)
    assert not report.has_errors, report.summary()
