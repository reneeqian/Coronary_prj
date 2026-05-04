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

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Force unbuffered output so Kaggle shows progress in real time.
os.environ["PYTHONUNBUFFERED"] = "1"


def _log(msg: str = "") -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Persistent results dataset (set to "" to skip the push)
#   One-time setup: create an empty private dataset at kaggle.com → Datasets
#   → New Dataset with slug "coronary-training-results", then set this to
#   "reneeqian/coronary-training-results".  The script pushes a new version
#   after every successful run so outputs survive session expiry.
# ---------------------------------------------------------------------------
KAGGLE_RESULTS_DATASET = "reneeqian/coronary-training-results"

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
    WORKING_DIR    = Path("/kaggle/working")
    DATASET_PATH   = Path("/kaggle/input/coca-gated-release/Gated_release_final")
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
    short = package.split("/")[-1].split(".git")[0]
    _log(f"  installing {short} ...")
    t0 = time.time()
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])
    _log(f"  {short} installed ({time.time() - t0:.1f}s)")


if ON_KAGGLE:
    _log("Installing packages from GitHub...")
    _pip_install("git+https://github.com/reneeqian/regulatory_tools.git@main")
    _pip_install("git+https://github.com/reneeqian/medical_image_ai_toolkit.git@main")
    _pip_install("git+https://github.com/reneeqian/Coronary_prj.git@main")
    _log("All packages installed.\n")


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
    _log("=" * 56)
    _log("  Coronary Calcium Segmentation — Kaggle Training")
    _log("=" * 56)
    _log(f"  ON_KAGGLE      : {ON_KAGGLE}")
    _log(f"  DATASET_PATH   : {DATASET_PATH}")
    _log(f"  CHECKPOINT_DIR : {CHECKPOINT_DIR}")
    _log(f"  ARTIFACTS_DIR  : {ARTIFACTS_DIR}")
    _log(f"  best_params    : {BEST_PARAMS}")
    _log(f"  epochs         : {TRAIN_EPOCHS}")

    # Auto-resume if a checkpoint exists
    resume_from = CHECKPOINT_LATEST if CHECKPOINT_LATEST.exists() else None
    if resume_from:
        _log(f"  resume_from    : {resume_from}")
    else:
        _log("  resume_from    : (none — starting fresh)")

    # --- GPU / environment diagnostics ---
    _log("-" * 56)
    _log("  Environment:")
    _log(f"    python  : {sys.version.split()[0]}")
    try:
        import torch
        _log(f"    torch   : {torch.__version__}")
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            _log(f"    GPU     : {gpu}  ({mem_gb:.1f} GB)")
            _log(f"    CUDA    : {torch.version.cuda}")
        else:
            _log("    GPU     : not available — running on CPU")
    except ImportError:
        _log("    torch   : not installed yet")
    _log("=" * 56)
    print(flush=True)

    if not DATASET_PATH.exists():
        _log(f"ERROR: dataset not found at {DATASET_PATH}")
        _log("Make sure the 'reneeqian/coca-gated-release' dataset is attached.")
        sys.exit(1)

    # --- Dataset info ---
    _log("Loading dataset...")
    learning_rate = BEST_PARAMS["learning_rate"]
    base_channels = BEST_PARAMS["base_channels"]

    ingestor   = COCAGatedIngestor(DATASET_PATH)
    datasource = MedicalImageDataSource(dataset_root=DATASET_PATH, ingestor=ingestor)

    patient_ids = ingestor.list_patient_ids()
    _log(f"  patients found : {len(patient_ids)}")
    print(flush=True)

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
    try:
        import torch
        total_params = sum(p.numel() for p in model.parameters())
        trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _log(f"  model          : UNet2D  base_channels={base_channels}")
        _log(f"  parameters     : {total_params:,}  ({trainable:,} trainable)")
    except Exception:
        pass
    print(flush=True)

    _log("Starting TrainingPipeline.run() ...")
    _log("  (per-epoch progress printed below by the trainer)")
    print(flush=True)

    # TrainingPipeline writes checkpoints to run_dir; output_dir routes them
    # to CHECKPOINT_DIR so checkpoint_latest.pt is easy to find on resume.
    pipeline = TrainingPipeline(
        datasource,
        model,
        config,
        output_dir=CHECKPOINT_DIR,
        resume_from=resume_from,
    )
    t_train = time.time()
    outputs = pipeline.run()
    elapsed_train = time.time() - t_train

    results = outputs["results"]
    _log("=" * 56)
    _log(f"  Training complete  ({elapsed_train / 60:.1f} min)")
    _log(f"  run_dir : {results.run_dir}")
    _log(f"  model   : {results.artifacts.get('model', '—')}")
    print(flush=True)

    # Copy final model to ARTIFACTS_DIR for easy download
    import shutil
    model_src = results.artifacts.get("model")
    if model_src and Path(model_src).exists():
        dest = ARTIFACTS_DIR / "model_final.pt"
        shutil.copy2(model_src, dest)
        _log(f"  copied model → {dest}")

    partitions_src = results.artifacts.get("partitions")
    if partitions_src and Path(partitions_src).exists():
        dest = ARTIFACTS_DIR / "partitions.json"
        shutil.copy2(partitions_src, dest)
        _log(f"  copied partitions → {dest}")

    if ON_KAGGLE:
        _push_artifacts(ARTIFACTS_DIR, KAGGLE_RESULTS_DATASET)

    _log("Done.")
    print(flush=True)


def _push_artifacts(artifacts_dir: Path, dataset_id: str) -> None:
    """Push artifacts_dir as a new version of a Kaggle dataset."""
    if not dataset_id:
        return
    _log(f"Pushing artifacts to Kaggle dataset: {dataset_id} ...")
    meta = {
        "title": "Coronary Training Results",
        "id": dataset_id,
        "licenses": [{"name": "other"}],
    }
    (artifacts_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    run_label = time.strftime("%Y-%m-%d %H:%M")
    result = subprocess.run(
        [
            "kaggle", "datasets", "version",
            "-p", str(artifacts_dir),
            "-m", f"training run {run_label}",
            "--dir-mode", "zip",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        _log(f"  pushed OK → kaggle.com/datasets/{dataset_id}")
    else:
        _log(f"  WARNING: push failed (return code {result.returncode})")
        if result.stderr.strip():
            _log(f"  stderr: {result.stderr.strip()}")
    print(flush=True)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
