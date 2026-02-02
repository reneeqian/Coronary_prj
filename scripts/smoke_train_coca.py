from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
import sys
from typing import List
import time


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------
# PATH SETUP (SCRIPT-ONLY GLUE)
# This disappears once medical_image_ai_toolkit is packaged
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------
# TOOLKIT IMPORTS
# ---------------------------------------------------------------------
from src.medical_image_ai_toolkit.datamodules.lazy_patient_datasource import LazyPatientDataSource
from src.medical_image_ai_toolkit.training.splits import DeterministicHoldoutSplitStrategy
from src.medical_image_ai_toolkit.evidence.evidence_report import EvidenceReport
from src.medical_image_ai_toolkit.adapters.patient_sample_to_tensor import PatientSampleTensorAdapter
from src.medical_image_ai_toolkit.training.medical_image_trainer import MedicalImageTrainer

# ---------------------------------------------------------------------
# PROJECT IMPORTS
# ---------------------------------------------------------------------
from src.ingestors.coca_gated_ingestor import COCAGatedIngestor


# =====================================================================
# CONFIGURATION (TRAINER INPUTS — STATIC & AUDITABLE)
# =====================================================================

# ---- Dataset / ingestion ----
DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "coca"
    / "cocacoronarycalciumandchestcts-2"
    / "Gated_release_final"
)

# ---- Run intent ----
RUN_INTENT = "smoke_training"

# ---- Hardware ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Optimization ----
TRAIN_FRACTION = 0.8
BATCH_SIZE = 16
NUM_EPOCHS = 2
LEARNING_RATE = 1e-3

# ---- Visualization ----
SHOW_TRAINING_PLOT = True   # Toggle training progress plot on/off
SAVE_TRAINING_PLOT = True  # Save plot to RUN_DIR

# ---- Progress reporting ----
PRINT_EVERY_N_BATCHES = 50   # set to 10 for fast feedback, 0 to disable

# ---- Reproducibility ----
RANDOM_SEED = 1337


# =====================================================================
# ARTIFACT ROOT (TRAINER-OWNED)
# =====================================================================
ARTIFACT_ROOT = Path("artifacts/training_runs")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = ARTIFACT_ROOT / TIMESTAMP
RUN_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# MODEL DEFINITION (TRAINER INPUT)
# =====================================================================

class SmallSliceCNN(nn.Module):
    """
    Minimal 2D CNN for slice-based binary classification.

    This model is intentionally small, transparent, and explainable.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(16, 1)

    def forward(self, x):
        # x: (B, 1, H, W)
        feats = self.features(x)
        feats = feats.view(x.size(0), -1)
        return self.classifier(feats)

# =====================================================================
# getting patient ids function (TRAINER INPUT)
# =====================================================================

def resolve_coca_patient_ids(dataset_root: Path) -> List[str]:
    """
    Resolve patient IDs for the COCA gated dataset.

    COCA structure (assumed):
        dataset_root/
            patient/
                <patient_id>/
                    <series_dir>/
            calcium_xml/
                <patient_id>.xml

    Patient identity is defined by directories under `patient/`.
    """

    dataset_root = Path(dataset_root)
    patient_root = dataset_root / "patient"

    if not patient_root.exists():
        raise FileNotFoundError(f"COCA patient root not found: {patient_root}")

    patient_ids: List[str] = []

    for entry in sorted(patient_root.iterdir()):
        if not entry.is_dir():
            continue

        if entry.name.startswith("."):
            continue

        patient_ids.append(entry.name)

    if not patient_ids:
        raise RuntimeError(
            f"No patient directories found under: {patient_root}"
        )

    return patient_ids

# =====================================================================
# SliceViewDataset Definition (TRAINER INPUT) - eventually to migrate 
#   to Task Definition
# =====================================================================
class DatasetAccessError(RuntimeError):
    pass


class SliceViewDataset(Dataset):
    """
    Minimal slice-level view over a patient-level datasource.

    Purpose:
    - Flatten (patient → volume → slice) into independent samples
    - Guarantee fixed tensor shapes for DataLoader
    - Enable smoke training only (no balancing, no weighting, no caching)

    Assumptions about upstream adapter output:
        sample["image"]: Tensor shaped (1, D, H, W)
        sample["target"]: scalar or Tensor broadcastable to slice-level
    """

    def __init__(self, patient_source):
        self.patient_source = patient_source
        self.index = []

        if not hasattr(patient_source, "get_num_slices"):
            raise TypeError(
                "SliceViewDataset requires patient_source.get_num_slices()"
            )

        for patient_idx in range(len(patient_source)):
            depth = patient_source.get_num_slices(patient_idx)

            if patient_idx < 3:
                print(f"  Patient {patient_idx}: {depth} slices")

            for slice_idx in range(depth):
                self.index.append((patient_idx, slice_idx))

        if not self.index:
            raise RuntimeError("SliceViewDataset built with zero slices")
        
        print(
            f"🧩 SliceViewDataset built: "
            f"{len(self.index)} slices from {len(patient_source)} patients"
        )


    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        patient_idx, slice_idx = self.index[idx]
        try: 
            sample = self.patient_source[patient_idx]

            image_4d = sample["image"]  # (1, D, H, W)
            image_slice = image_4d[:, slice_idx, :, :]
        except Exception as e:
            raise DatasetAccessError(
                f"Patient {patient_idx}, slice {slice_idx} failed"
            ) from e
        target = sample.get("target")

        # ----------------------------
        # Target handling
        # ----------------------------
        if torch.is_tensor(target):
            # Case 1: scalar target (patient-level)
            if target.ndim == 0:
                pass

            # Case 2: length-1 tensor (patient-level)
            elif target.ndim == 1 and target.numel() == 1:
                target = target.squeeze(0)

            # Case 3: explicit slice-level target
            elif target.ndim == 1:
                if slice_idx >= target.numel():
                    raise IndexError(
                        f"slice_idx {slice_idx} out of bounds for target of shape {target.shape}"
                    )
                target = target[slice_idx]

            else:
                raise ValueError(
                    f"Unsupported target shape {target.shape} in SliceViewDataset"
                )
        else:
            # Smoke-training default / unlabeled case
            target = torch.zeros((), dtype=torch.float32)

        return {
            "image": image_slice,    # (1, H, W)
            "target": target,
            "patient_idx": patient_idx,
            "slice_idx": slice_idx,
        }

def plot_training_history(history, run_dir: Path, show: bool, save: bool):
    """
    Plot training / validation loss curves.

    Safe to call in scripts or notebooks.
    """
    if not show and not save:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib not available — skipping training plot")
        return

    epochs = [h["epoch"] for h in history]
    train_losses = [h["train"]["loss"] for h in history]
    val_losses = [h["val"]["loss"] for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, val_losses, label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)

    if save:
        plot_path = run_dir / "training_curve.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"📈 Training curve saved to: {plot_path}")

    if show:
        plt.show()
    else:
        plt.close()

def preflight_dataset_check(
    patient_source,
    max_patients: int = 5,
):
    """
    Lightweight sanity scan to catch structural issues before training.
    Does NOT exhaustively scan the dataset.
    """
    print("🧪 Running dataset preflight checks...")

    for i in range(min(len(patient_source), max_patients)):
        try:
            sample = patient_source[i]
            image = sample["image"]

            assert torch.is_tensor(image), "image is not a tensor"
            assert image.ndim == 4, f"expected (1, D, H, W), got {image.shape}"
            assert image.shape[1] > 0, "zero slices"

            if "target" in sample:
                target = sample["target"]
                assert torch.is_tensor(target), "target not tensor"

        except Exception as e:
            raise RuntimeError(
                f"Preflight failed on patient index {i}"
            ) from e

    print("✅ Preflight checks passed")


# =====================================================================
# MAIN (SCRIPT-LEVEL ORCHESTRATION — WILL SHRINK OVER TIME)
# =====================================================================

def main():
    
    print("=" * 80)
    print("🚀 Starting smoke training run")
    print(f"Run intent: {RUN_INTENT}")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Artifacts dir: {RUN_DIR}")
    print(f"Device: {DEVICE}")
    print("=" * 80)
    
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    
    print(f"🔧 Random seed set to: {RANDOM_SEED}")

    # ------------------------------------------------------------
    # INGESTOR (TRAINER INPUT)
    # ------------------------------------------------------------
    ingestor = COCAGatedIngestor(DATASET_ROOT)

    # ------------------------------------------------------------
    # SPLIT STRATEGY (TRAINER INPUT)
    # ------------------------------------------------------------
    split_strategy = DeterministicHoldoutSplitStrategy(
        train_fraction=TRAIN_FRACTION,
        val_fraction=1.0 - TRAIN_FRACTION,
    )

    patient_ids = resolve_coca_patient_ids(DATASET_ROOT)
    
    print(f"📂 Total patients discovered: {len(patient_ids)}")
    print(f"First 5 patient IDs: {patient_ids[:5]}")

    splits = split_strategy.split(patient_ids)
    
    print("🔀 Dataset split summary:")
    for split_name, ids in splits.items():
        print(f"  {split_name}: {len(ids)} patients")

    # ------------------------------------------------------------
    # ADAPTER (TRAINER INPUT)
    # ------------------------------------------------------------
    adapter = PatientSampleTensorAdapter(
        normalize=True,
        require_annotations=False,
    )

    # ------------------------------------------------------------
    # PATIENT-LEVEL DATA SOURCES (TRAINER INPUTS)
    # ------------------------------------------------------------

    train_source = LazyPatientDataSource(
        patient_ids=splits["train"],
        ingestor=ingestor,
        adapter=adapter,
    )

    val_source = LazyPatientDataSource(
        patient_ids=splits["val"],
        ingestor=ingestor,
        adapter=adapter,
    )

    print(f"🧠 Train source length: {len(train_source)} patients")
    print(f"🧠 Val source length:   {len(val_source)} patients")
    
    print("🧪 Testing lazy patient load (train[0])...")
    sample0 = train_source[0]
    print(f"  image shape: {sample0['image'].shape}")
    print(f"  target type: {type(sample0.get('target'))}")

    preflight_dataset_check(train_source)
    preflight_dataset_check(val_source)
    
    with open(RUN_DIR / "preflight.json", "w") as f:
        json.dump(
            {
                "status": "passed",
                "checked_patients_per_split": 5,
            },
            f,
            indent=2,
        )


    # ============================================================
    # >>> ALL TRAINER INPUTS HAVE NOW BEEN CONSTRUCTED <<<
    #
    # From this point onward, logic belongs inside MedicalImageTrainer
    # ============================================================

    # ------------------------------------------------------------
    # DATALOADERS (TRAINER-OWNED)
    # NOTE: Iteration granularity (slice vs volume) will move
    #       into TaskDefinition in the next iteration.
    # ------------------------------------------------------------
    train_dataset = SliceViewDataset(train_source)
    val_dataset = SliceViewDataset(val_source)

    # ------------------------------------------------------------
    # DATA SPLIT ARTIFACT (FDA-CRITICAL)
    # ------------------------------------------------------------
    with open(RUN_DIR / "data_splits.json", "w") as f:
        json.dump(
            {
                "train": {
                    "patient_ids": splits["train"],
                    "num_patients": len(splits["train"]),
                    "num_slices": len(train_dataset),
                },
                "val": {
                    "patient_ids": splits["val"],
                    "num_patients": len(splits["val"]),
                    "num_slices": len(val_dataset),
                },
            },
            f,
            indent=2,
        )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # keep 0 for smoke test
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )


    # ------------------------------------------------------------
    # OPTIMIZATION SETUP (TRAINER-OWNED)
    # ------------------------------------------------------------
    model = SmallSliceCNN().to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # ------------------------------------------------------------
    # STATIC ARTIFACT CAPTURE (TRAINER-OWNED)
    # ------------------------------------------------------------
    static_metadata = {
        "run_intent": RUN_INTENT,
        "run_type": "training",
        "timestamp": TIMESTAMP,

        "reproducibility": {
            "random_seed": RANDOM_SEED,
        },

        "hardware": {
            "device": str(DEVICE),
            "cuda_available": torch.cuda.is_available(),
        },

        "dataset": {
            "name": "COCA",
            "variant": "Gated_release_final",
            "root": str(DATASET_ROOT),
            "num_patients_discovered": len(patient_ids),
        },

        "split_strategy": split_strategy.metadata(),

        "model": {
            "architecture": "SmallSliceCNN",
            "task": "slice_binary_classification",
            "input_shape": [1, "H", "W"],
            "output_shape": [1],
        },

        "optimization": {
            "optimizer": "Adam",
            "loss": "BCEWithLogitsLoss",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs_planned": NUM_EPOCHS,
        },

        "software": {
            "torch_version": torch.__version__,
        },
    }


    with open(RUN_DIR / "run_config.json", "w") as f:
        json.dump(static_metadata, f, indent=2)

    # ------------------------------------------------------------
    # TRAINING LOOP (TRAINER-OWNED)
    # ------------------------------------------------------------
    trainer = MedicalImageTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=DEVICE,
        run_dir=RUN_DIR,
        run_config=static_metadata,
        data_splits={
            "train": {
                "patient_ids": splits["train"],
                "num_patients": len(train_source),
                "num_slices": len(train_dataset),
            },
            "val": {
                "patient_ids": splits["val"],
                "num_patients": len(val_source),
                "num_slices": len(val_dataset),
            },
        },
        show_training_plot=SHOW_TRAINING_PLOT,
        save_training_plot=SAVE_TRAINING_PLOT,
        print_every_n_batches=PRINT_EVERY_N_BATCHES,
    )

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
    )
    print("✅ Smoke training run complete")



if __name__ == "__main__":
    main()
