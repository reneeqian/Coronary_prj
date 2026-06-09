import json
import sys
from pathlib import Path

from medical_image_ai_toolkit.dataobjects.datasources.deterministic_split import (
    DeterministicHoldoutSplit,
)
from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import (
    MedicalImageDataSource,
)
from medical_image_ai_toolkit.pipeline.training_pipeline import TrainingPipeline
from medical_image_ai_toolkit.training.training_config import TrainingConfig
from regulatory_tools.requirements.yaml_requirement_provider import YamlRequirementProvider

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
TUNING_RUNS_DIR = PROJECT_ROOT / "artifacts" / "tuning_runs"
REQUIREMENT_PATH = PROJECT_ROOT / "docs" / "requirements.yaml"


def load_best_params_from_latest_sweep() -> tuple[dict, Path]:
    """Return (best_params_dict, sweep_dir) from the most recent tuning run."""
    if not TUNING_RUNS_DIR.exists():
        print(f"No tuning runs directory found at {TUNING_RUNS_DIR}")
        print("Run smoketesthyperparametertuning.py first.")
        sys.exit(1)

    sweep_dirs = sorted(
        [d for d in TUNING_RUNS_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    if not sweep_dirs:
        print(f"No sweep runs found in {TUNING_RUNS_DIR}")
        print("Run smoketesthyperparametertuning.py first.")
        sys.exit(1)

    latest_dir = sweep_dirs[-1]
    report_path = latest_dir / "tuning_report.json"
    if not report_path.exists():
        print(f"No tuning_report.json found in {latest_dir}")
        sys.exit(1)

    report = json.loads(report_path.read_text())
    best_trial = report.get("best_trial")
    if best_trial is None:
        print(
            "tuning_report.json has no best_trial entry — sweep may have had no completed trials."
        )
        sys.exit(1)

    return best_trial["params"], latest_dir


def main():
    print("Loading best params from latest hyperparameter sweep...")
    best_params, sweep_dir = load_best_params_from_latest_sweep()
    print(f"  sweep: {sweep_dir.name}")
    print(f"  best params: {best_params}")

    learning_rate = best_params["learning_rate"]
    base_channels = best_params.get("base_channels", 32)

    print("\nCreating ingestor...")
    ingestor = COCAGatedIngestor(DATASET_PATH)

    print("Creating datasource...")
    datasource = MedicalImageDataSource(dataset_root=DATASET_PATH, ingestor=ingestor)

    provider = YamlRequirementProvider(REQUIREMENT_PATH)

    config = TrainingConfig(
        epochs=20,
        learning_rate=learning_rate,
        task=CoronaryCalciumTask(),
        split_strategy=DeterministicHoldoutSplit(
            train=0.7,
            val=0.15,
            seed=42,
            max_train=100,
            max_val=100,
            max_test=100,
        ),
        early_stop=True,
    )

    model = UNet2D(base_channels=base_channels)

    print(
        f"\nTraining UNet2D(base_channels={base_channels}) | lr={learning_rate} | epochs={config.epochs}"
    )

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
