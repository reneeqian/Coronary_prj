from pathlib import Path

from medical_image_ai_toolkit.dataobjects.datasources.deterministic_split import (
    DeterministicHoldoutSplit,
)
from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import (
    MedicalImageDataSource,
)
from medical_image_ai_toolkit.pipeline.hyperparameter_tuning_pipeline import (
    HyperparameterTuningPipeline,
)
from medical_image_ai_toolkit.training.training_config import TrainingConfig
from medical_image_ai_toolkit.tuning.hyperparameter_space import HyperparameterSpace

from Coronary_prj.ingestors.coca_gated_ingestor import COCAGatedIngestor
from Coronary_prj.models.unet2d import UNet2D
from Coronary_prj.task_definitions.coronary_calcium_task import CoronaryCalciumTask

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "coca"
    / "cocacoronarycalciumandchestcts-2"
    / "Gated_release_final"
)


def main():
    print("Creating ingestor...")
    ingestor = COCAGatedIngestor(DATASET_PATH)

    print("Creating datasource...")
    datasource = MedicalImageDataSource(dataset_root=DATASET_PATH, ingestor=ingestor)

    # Search over learning rate and UNet2D architecture depth
    space = HyperparameterSpace(
        learning_rate=[1e-3, 1e-4],
        base_channels=[16, 32],
    )
    print(f"Search space: {len(space)} trial(s)")

    def trial_factory(params):
        model = UNet2D(base_channels=params["base_channels"])
        config = TrainingConfig(
            epochs=3,
            learning_rate=params["learning_rate"],
            task=CoronaryCalciumTask(),
            early_stop=True,
        )
        return model, config

    pipeline = HyperparameterTuningPipeline(
        datasource=datasource,
        trial_factory=trial_factory,
        space=space,
        split_strategy=DeterministicHoldoutSplit(
            train=0.7,
            val=0.15,
            seed=42,
            max_train=20,
            max_val=5,
            max_test=5,
        ),
        output_dir=PROJECT_ROOT / "artifacts" / "tuning_runs",
        search_strategy="grid",
    )

    results = pipeline.run()
    results.summary()

    if results.best_trial is not None:
        print(f"\nBest params: {results.best_trial.params}")


if __name__ == "__main__":
    main()
