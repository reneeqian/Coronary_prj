"""
Coronary calcium Kaggle training script.

Change TRAINING_MODE below to select the dataset/task combination:

  "gated"    — gated CT segmentation    (UNet2D,                COCA Gated)
  "nongated" — non-gated CT regression  (CalciumScoreRegressor, COCA Non-gated)

On Kaggle, attach the matching dataset for the chosen mode:
  gated    → reneeqian/coca-gated-release      (/kaggle/input/coca-gated-release/)
  nongated → reneeqian/coca-nongated-release   (/kaggle/input/coca-nongated-release/)

Checkpoints go to /kaggle/working/checkpoints/<mode>/ and survive resume.
Final artifacts go to /kaggle/working/artifacts/<mode>/.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

# ---------------------------------------------------------------------------
# *** CHANGE THIS LINE TO SWITCH MODES ***
# ---------------------------------------------------------------------------
TRAINING_MODE = "gated"   # "gated" | "nongated"

# ---------------------------------------------------------------------------
# Shared settings
# ---------------------------------------------------------------------------
# One-time setup: create an empty private Kaggle dataset at kaggle.com →
# Datasets → New Dataset with slug "coronary-training-results".
# Set to "" to skip the push.
KAGGLE_RESULTS_DATASET = "reneeqian/coronary-training-results"

ON_KAGGLE = Path("/kaggle/working").exists()


def _log(msg: str = "") -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


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
from Coronary_prj.ingestors.coca_gated_ingestor import COCAGatedIngestor                      # noqa: E402
from Coronary_prj.ingestors.coca_nongated_ingestor import COCANongatedIngestor                 # noqa: E402
from Coronary_prj.models.calcium_score_regressor import CalciumScoreRegressor                  # noqa: E402
from Coronary_prj.models.unet2d import UNet2D                                                  # noqa: E402
from Coronary_prj.task_definitions.coronary_calcium_task import CoronaryCalciumTask            # noqa: E402
from Coronary_prj.task_definitions.nongated_calcium_score_task import NongatedCalciumScoreTask # noqa: E402
from medical_image_ai_toolkit.dataobjects.datasources.deterministic_split import (             # noqa: E402
    DeterministicHoldoutSplit,
)
from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import (        # noqa: E402
    MedicalImageDataSource,
)
from medical_image_ai_toolkit.pipeline.training_pipeline import TrainingPipeline               # noqa: E402
from medical_image_ai_toolkit.training.training_config import TrainingConfig                   # noqa: E402


# ---------------------------------------------------------------------------
# Mode registry
#
# kaggle_slug     — folder name under /kaggle/input/ (matches the attached dataset)
# kaggle_subpath  — subdirectory within that folder that is the ingestor root
#                   (set to "" if the dataset root IS the ingestor root)
# local_subpath   — path relative to project root for local fallback
# best_params     — update these after each local hyperparameter sweep
# split_kwargs    — passed directly to DeterministicHoldoutSplit
# ---------------------------------------------------------------------------
_MODES: dict[str, dict] = {
    "gated": {
        "label":          "Gated CT Segmentation (UNet2D)",
        "kaggle_slug":    "coca-gated-release",
        "kaggle_subpath": "Gated_release_final",
        "local_subpath":  "data/raw/coca/cocacoronarycalciumandchestcts-2/Gated_release_final",
        "ingestor_cls":   COCAGatedIngestor,
        "task_cls":       CoronaryCalciumTask,
        "model_cls":      UNet2D,
        "best_params":    {"learning_rate": 0.001, "base_channels": 32},
        "train_epochs":   50,
        "split_kwargs":   {"train": 0.7, "val": 0.15, "seed": 42,
                           "max_train": 600, "max_val": 100, "max_test": 87},
    },
    "nongated": {
        "label":          "Non-gated CT Regression (CalciumScoreRegressor)",
        "kaggle_slug":    "coca-nongated-release",
        "kaggle_subpath": "deidentified_nongated",
        "local_subpath":  "data/raw/coca/cocacoronarycalciumandchestcts-2/deidentified_nongated",
        "ingestor_cls":   COCANongatedIngestor,
        "task_cls":       NongatedCalciumScoreTask,
        "model_cls":      CalciumScoreRegressor,
        "best_params":    {"learning_rate": 0.001, "base_channels": 16},
        "train_epochs":   50,
        "split_kwargs":   {"train": 0.7, "val": 0.15, "seed": 42,
                           "max_train": 150, "max_val": 30, "max_test": 30},
    },
}

if TRAINING_MODE not in _MODES:
    raise ValueError(
        f"Unknown TRAINING_MODE {TRAINING_MODE!r}. Choose from: {list(_MODES)}"
    )

_cfg = _MODES[TRAINING_MODE]

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
if ON_KAGGLE:
    WORKING_DIR  = Path("/kaggle/working")
    _input_root  = Path("/kaggle/input") / _cfg["kaggle_slug"]
    DATASET_PATH = (_input_root / _cfg["kaggle_subpath"]
                    if _cfg["kaggle_subpath"] else _input_root)
else:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    WORKING_DIR   = _PROJECT_ROOT
    DATASET_PATH  = _PROJECT_ROOT / _cfg["local_subpath"]

CHECKPOINT_DIR    = WORKING_DIR / "checkpoints" / TRAINING_MODE
ARTIFACTS_DIR     = (WORKING_DIR / "artifacts" / TRAINING_MODE
                     if ON_KAGGLE else WORKING_DIR / "artifacts" / "training_runs")
CHECKPOINT_LATEST = CHECKPOINT_DIR / "checkpoint_latest.pt"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    best_params  = _cfg["best_params"]
    train_epochs = _cfg["train_epochs"]

    _log("=" * 60)
    _log(f"  Coronary Training — {_cfg['label']}")
    _log("=" * 60)
    _log(f"  mode           : {TRAINING_MODE}")
    _log(f"  ON_KAGGLE      : {ON_KAGGLE}")
    _log(f"  DATASET_PATH   : {DATASET_PATH}")
    _log(f"  CHECKPOINT_DIR : {CHECKPOINT_DIR}")
    _log(f"  ARTIFACTS_DIR  : {ARTIFACTS_DIR}")
    _log(f"  best_params    : {best_params}")
    _log(f"  epochs         : {train_epochs}")

    resume_from = CHECKPOINT_LATEST if CHECKPOINT_LATEST.exists() else None
    _log(f"  resume_from    : {resume_from or '(none — starting fresh)'}")

    _log("-" * 60)
    _log("  Environment:")
    _log(f"    python  : {sys.version.split()[0]}")
    try:
        import torch
        _log(f"    torch   : {torch.__version__}")
        if torch.cuda.is_available():
            gpu    = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            _log(f"    GPU     : {gpu}  ({mem_gb:.1f} GB)")
            _log(f"    CUDA    : {torch.version.cuda}")
        else:
            _log("    GPU     : not available — running on CPU")
    except ImportError:
        _log("    torch   : not installed yet")
    _log("=" * 60)
    print(flush=True)

    if not DATASET_PATH.exists():
        _log(f"ERROR: dataset not found at {DATASET_PATH}")
        _log(f"Attach the '{_cfg['kaggle_slug']}' dataset to this notebook.")
        sys.exit(1)

    learning_rate = best_params["learning_rate"]
    base_channels = best_params["base_channels"]

    _log("Loading dataset...")
    ingestor    = _cfg["ingestor_cls"](DATASET_PATH)
    datasource  = MedicalImageDataSource(dataset_root=DATASET_PATH, ingestor=ingestor)
    patient_ids = ingestor.list_patient_ids()
    _log(f"  patients found : {len(patient_ids)}")
    print(flush=True)

    config = TrainingConfig(
        epochs=train_epochs,
        learning_rate=learning_rate,
        device="cuda" if _cuda_available() else "cpu",
        task=_cfg["task_cls"](),
        split_strategy=DeterministicHoldoutSplit(**_cfg["split_kwargs"]),
        early_stop=True,
        loss_threshold=0.01,
        plateau_patience=7,
        checkpoint_every=1,
    )

    model = _cfg["model_cls"](base_channels=base_channels)
    try:
        import torch
        total_params = sum(p.numel() for p in model.parameters())
        trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _log(f"  model          : {model.__class__.__name__}  base_channels={base_channels}")
        _log(f"  parameters     : {total_params:,}  ({trainable:,} trainable)")
    except Exception:
        pass
    print(flush=True)

    _log("Starting TrainingPipeline.run() ...")
    _log("  (per-epoch progress printed below by the trainer)")
    print(flush=True)

    pipeline = TrainingPipeline(
        datasource,
        model,
        config,
        output_dir=CHECKPOINT_DIR,
        resume_from=resume_from,
    )
    t_train = time.time()
    outputs = pipeline.run()
    elapsed = time.time() - t_train

    results = outputs["results"]
    _log("=" * 60)
    _log(f"  Training complete  ({elapsed / 60:.1f} min)")
    _log(f"  run_dir : {results.run_dir}")
    _log(f"  model   : {results.artifacts.get('model', '—')}")
    print(flush=True)

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
