from pathlib import Path

from Coronary_prj.ingestors.coca_nongated_ingestor import COCANongatedIngestor
from Coronary_prj.models.calcium_score_regressor import CalciumScoreRegressor
from Coronary_prj.task_definitions.nongated_calcium_score_task import NongatedCalciumScoreTask
from medical_image_ai_toolkit.dataobjects.datasources.deterministic_split import (
    DeterministicHoldoutSplit,
)
from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import (
    MedicalImageDataSource,
)
from medical_image_ai_toolkit.pipeline.training_pipeline import TrainingPipeline
from medical_image_ai_toolkit.training.training_config import TrainingConfig
from regulatory_tools.requirements.yaml_requirement_provider import YamlRequirementProvider

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "coca"
    / "cocacoronarycalciumandchestcts-2"
    / "deidentified_nongated"
)
REQUIREMENT_PATH = PROJECT_ROOT / "docs" / "requirements.yaml"


def main():
    learning_rate = 1e-3
    base_channels = 16

    print("Creating nongated ingestor...")
    ingestor = COCANongatedIngestor(DATASET_PATH)

    patient_ids = ingestor.list_patient_ids()
    print(f"  patients found: {len(patient_ids)}")

    print("Creating datasource...")
    datasource = MedicalImageDataSource(dataset_root=DATASET_PATH, ingestor=ingestor)

    provider = YamlRequirementProvider(REQUIREMENT_PATH)

    config = TrainingConfig(
        epochs=30,
        learning_rate=learning_rate,
        task=NongatedCalciumScoreTask(),
        split_strategy=DeterministicHoldoutSplit(
            train=0.7,
            val=0.15,
            seed=42,
            max_train=100,
            max_val=50,
            max_test=50,
        ),
        early_stop=True,
    )

    model = CalciumScoreRegressor(base_channels=base_channels)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTraining CalciumScoreRegressor(base_channels={base_channels}) | "
          f"params={total_params:,} | lr={learning_rate} | epochs={config.epochs}")

    pipeline = TrainingPipeline(
        datasource,
        model,
        config,
        req_provider=provider,
        output_dir=PROJECT_ROOT / "artifacts" / "training_runs",
    )
    outputs = pipeline.run()
    print("\nTraining complete:", outputs)


if __name__ == "__main__":
    main()
