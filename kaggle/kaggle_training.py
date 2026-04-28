"""
Coronary calcium segmentation — Kaggle training script.

Installs packages from GitHub, detects the Kaggle environment, picks the
best hyperparameters from the local sweep report (embedded below), and runs
a full training pipeline with per-epoch checkpointing so the session can be
resumed if interrupted.

Run on Kaggle:
    Add this script as a Kaggle dataset or paste into a notebook cell,
    then execute.  The COCA dataset must be attached as:
        reneeqian/coca-gated-release
    giving a data root of:
        /kaggle/input/coca-gated-release/Gated_release_final

Checkpoints are written to /kaggle/working/checkpoints/.
The final model is saved to /kaggle/working/artifacts/.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Best hyperparameters from local grid sweep (update after each new sweep)
# ---------------------------------------------------------------------------
BEST_PARAMS = {
    "learning_rate": 0.001,
    "base_channels": 32,
}
TRAIN_EPOCHS = 50


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
ON_KAGGLE = Path("/kaggle/working").exists()

if ON_KAGGLE:
    WORKING_DIR   = Path("/kaggle/working")
    DATASET_PATH  = Path("/kaggle/input/coca-gated-release/Gated_release_final")
    CHECKPOINT_DIR = WORKING_DIR / "checkpoints"
    ARTIFACTS_DIR  = WORKING_DIR / "artifacts"
else:
    # Local fallback — mirrors the smoketesttraining.py paths
    _PROJECT_ROOT  = Path(__file__).resolve().parents[1]
    WORKING_DIR    = _PROJECT_ROOT
    DATASET_PATH   = (
        _PROJECT_ROOT
        / "data" / "raw" / "coca"
        / "cocacoronarycalciumandchestcts-2"
        / "Gated_release_final"
    )
    CHECKPOINT_DIR = _PROJECT_ROOT / "artifacts" / "checkpoints"
    ARTIFACTS_DIR  = _PROJECT_ROOT / "artifacts" / "training_runs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_LATEST = CHECKPOINT_DIR / "checkpoint_latest.pt"


# ---------------------------------------------------------------------------
# Install packages (Kaggle session — packages are not pre-installed)
# ---------------------------------------------------------------------------
def _pip_install(package: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])


if ON_KAGGLE:
    print("Installing packages from GitHub...")
    _pip_install("git+https://github.com/reneeqian/regulatory_tools.git@main")
    _pip_install("git+https://github.com/reneeqian/medical_image_ai_toolkit.git@main")
    _pip_install("git+https://github.com/reneeqian/Coronary_prj.git@main")
    print("Packages installed.\n")


# ---------------------------------------------------------------------------
# Imports (after install so Kaggle can resolve them)
# ---------------------------------------------------------------------------
from Coronary_prj.ingestors.coca_gated_ingestor import COCAGatedIngestor  # noqa: E402
from Coronary_prj.models.unet2d import UNet2D  # noqa: E402
from Coronary_prj.task_definitions.coronary_calcium_task import CoronaryCalciumTask  # noqa: E402
from medical_image_ai_toolkit.dataobjects.datasources.deterministic_split import (  # noqa: E402
    DeterministicHoldoutSplit,
)
from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import (  # noqa: E402
    MedicalImageDataSource,
)
from medical_image_ai_toolkit.pipeline.training_pipeline import TrainingPipeline  # noqa: E402
from medical_image_ai_toolkit.training.training_config import TrainingConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("  Coronary Calcium Segmentation — Kaggle Training")
    print("=" * 60)
    print(f"  ON_KAGGLE      : {ON_KAGGLE}")
    print(f"  DATASET_PATH   : {DATASET_PATH}")
    print(f"  CHECKPOINT_DIR : {CHECKPOINT_DIR}")
    print(f"  ARTIFACTS_DIR  : {ARTIFACTS_DIR}")
    print(f"  best_params    : {BEST_PARAMS}")
    print(f"  epochs         : {TRAIN_EPOCHS}")

    # Auto-resume if a checkpoint exists
    resume_from = CHECKPOINT_LATEST if CHECKPOINT_LATEST.exists() else None
    if resume_from:
        print(f"  resume_from    : {resume_from}")
    else:
        print("  resume_from    : (none — starting fresh)")
    print("=" * 60 + "\n")

    if not DATASET_PATH.exists():
        print(f"ERROR: dataset not found at {DATASET_PATH}")
        print("Make sure the 'reneeqian/coca-gated-release' dataset is attached to this notebook.")
        sys.exit(1)

    learning_rate = BEST_PARAMS["learning_rate"]
    base_channels = BEST_PARAMS["base_channels"]

    ingestor   = COCAGatedIngestor(DATASET_PATH)
    datasource = MedicalImageDataSource(dataset_root=DATASET_PATH, ingestor=ingestor)

    config = TrainingConfig(
        epochs=TRAIN_EPOCHS,
        learning_rate=learning_rate,
        device="cuda" if _cuda_available() else "cpu",
        task=CoronaryCalciumTask(),
        split_strategy=DeterministicHoldoutSplit(
            train=0.7,
            val=0.15,
            seed=42,
            max_train=600,
            max_val=100,
            max_test=87,
        ),
        early_stop=True,
        loss_threshold=0.01,
        plateau_patience=7,
        checkpoint_every=1,
    )

    model = UNet2D(base_channels=base_channels)

    # Redirect checkpoint_latest into CHECKPOINT_DIR by symlinking run_dir.
    # TrainingPipeline writes checkpoints to run_dir; we tell MedicalImageTrainer
    # to use CHECKPOINT_DIR as output_dir so checkpoints land there.
    pipeline = TrainingPipeline(
        datasource,
        model,
        config,
        output_dir=CHECKPOINT_DIR,
        resume_from=resume_from,
    )
    outputs = pipeline.run()

    results = outputs["results"]
    print(f"\nTraining complete.")
    print(f"  run_dir : {results.run_dir}")
    print(f"  model   : {results.artifacts.get('model', '—')}")

    # Copy final model to ARTIFACTS_DIR for easy download
    import shutil
    model_src = results.artifacts.get("model")
    if model_src and Path(model_src).exists():
        dest = ARTIFACTS_DIR / "model_final.pt"
        shutil.copy2(model_src, dest)
        print(f"  copied model → {dest}")

    partitions_src = results.artifacts.get("partitions")
    if partitions_src and Path(partitions_src).exists():
        dest = ARTIFACTS_DIR / "partitions.json"
        shutil.copy2(partitions_src, dest)
        print(f"  copied partitions → {dest}")


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
