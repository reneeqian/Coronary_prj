import sys
from pathlib import Path
import torch

from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import MedicalImageDataSource
from medical_image_ai_toolkit.training.training_config import TrainingConfig
from medical_image_ai_toolkit.pipeline.validation_pipeline import ValidationPipeline

from Coronary_prj.ingestors.coca_gated_ingestor import COCAGatedIngestor
from Coronary_prj.task_definitions.coronary_calcium_task import CoronaryCalciumTask
from Coronary_prj.models.small_segmentation_cnn import SmallSegmentationCNN
from Coronary_prj.models.unet2d import UNet2D


PROJECT_ROOT  = Path(__file__).resolve().parents[1]
DATASET_PATH  = PROJECT_ROOT / "data" / "raw" / "coca" / "cocacoronarycalciumandchestcts-2" / "Gated_release_final"
TRAINING_RUNS = PROJECT_ROOT / "artifacts" / "training_runs"


def _find_latest_run(training_runs_root: Path) -> tuple[Path | None, Path | None]:
    """
    Scan training_runs_root for the most recently created run directory
    that contains both model.pt and partitions.json.  Returns
    (model_path, partitions_path) or (None, None) if none exist.
    """
    candidates = sorted(training_runs_root.glob("*/model.pt"))
    for model_pt in reversed(candidates):
        partitions = model_pt.parent / "partitions.json"
        if partitions.exists():
            return model_pt, partitions
    # Fall back to most recent model even without partitions
    if candidates:
        return candidates[-1], None
    return None, None


def main():

    print("Creating ingestor...")
    ingestor = COCAGatedIngestor(DATASET_PATH)

    print("Creating datasource...")
    datasource = MedicalImageDataSource(
        dataset_root=DATASET_PATH,
        ingestor=ingestor,
    )

    print("Building model...")
    model = UNet2D()

    model_pt, partitions_path = _find_latest_run(TRAINING_RUNS)

    if model_pt is not None:
        print(f"Loading model weights from:  {model_pt}")
        model.load_state_dict(torch.load(model_pt, map_location="cpu"))
    else:
        print(
            "WARNING: No trained model found under artifacts/training_runs. "
            "Proceeding with randomly initialised weights."
        )

    if partitions_path is None:
        print(
            "ERROR: No partitions.json found in artifacts/training_runs. "
            "Run smoketesttraining.py first to produce a trained model and "
            "partition file before running validation."
        )
        sys.exit(1)

    # TrainingConfig is required by ValidationPipeline for device and task;
    # the split_strategy field is unused here because partitions are loaded
    # from file rather than re-computed.
    config = TrainingConfig(
        epochs=2,
        batch_size=2,
        task=CoronaryCalciumTask(),
    )

    pipeline = ValidationPipeline(
        datasource=datasource,
        model=model,
        config=config,
        partitions_path=partitions_path,
    )
    results = pipeline.run()


if __name__ == "__main__":
    main()