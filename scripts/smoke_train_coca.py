from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
import sys
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---- Toolkit imports ----
from src.medical_image_ai_toolkit.datamodules.lazy_patient_datamodule import LazyPatientDataset
from src.medical_image_ai_toolkit.training.splits import DeterministicHoldoutSplitStrategy
from src.medical_image_ai_toolkit.evidence import EvidenceReport
from src.medical_image_ai_toolkit.adapters.patient_sample_to_tensor import PatientSampleTensorAdapter

# ---- Project imports ----
from src.ingestors.coca_gated_ingestor import CocaGatedIngestor
from medical_image_ai_toolkit.datamodules.slice_dataset import SliceDataset


# ---------------------------------------------------------------------
# Configuration (explicit + auditable)
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # adjust if needed
sys.path.insert(0, str(PROJECT_ROOT))
DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "coca"
    / "cocacoronarycalciumandchestcts-2"
    / "Gated_release_final"
)

RUN_INTENT = "smoke_training"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_FRACTION = 0.8
BATCH_SIZE = 16
NUM_EPOCHS = 2
LEARNING_RATE = 1e-3

ARTIFACT_ROOT = Path("artifacts/training_runs")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = ARTIFACT_ROOT / TIMESTAMP
RUN_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Model (intentionally small + transparent)
# ---------------------------------------------------------------------

class SmallSliceCNN(nn.Module):
    """
    Minimal 2D CNN for slice-based binary classification.
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


# ---------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------

def main():

    # ------------------------------------------------------------
    # Load patients (domain objects)
    # ------------------------------------------------------------
    ingestor = CocaGatedIngestor(DATASET_ROOT)
    all_patient_ids = ingestor.list_patient_ids()


    # ------------------------------------------------------------
    # Split strategy
    # ------------------------------------------------------------
    split_strategy = DeterministicHoldoutSplitStrategy(
        train_fraction=TRAIN_FRACTION,
        val_fraction=1.0 - TRAIN_FRACTION,
    )

    splits = split_strategy.split(all_patient_ids)

    # ------------------------------------------------------------
    # Adapter
    # Vector ROIs → binary label, allow default negatives
    # ------------------------------------------------------------
    adapter = PatientSampleTensorAdapter(
        require_annotations=False,
        default_target=0.0,
    )

    # ------------------------------------------------------------
    # Datasets (slice-based)
    # ------------------------------------------------------------
    adapter = PatientSampleTensorAdapter(
        normalize=True,
        require_annotations=True,
    )

    train_ds = LazyPatientDataset(
        patient_ids=splits["train"],
        ingestor=ingestor,
        adapter=adapter,
    )

    val_ds = LazyPatientDataset(
        patient_ids=splits["val"],
        ingestor=ingestor,
        adapter=adapter,
    )


    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ------------------------------------------------------------
    # Model / loss / optimizer
    # ------------------------------------------------------------
    model = SmallSliceCNN().to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ------------------------------------------------------------
    # Artifact capture (static)
    # ------------------------------------------------------------
    static_metadata = {
        "run_intent": RUN_INTENT,
        "device": str(DEVICE),
        "split_strategy": split_strategy.metadata(),
        "model": "SmallSliceCNN",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
    }

    with open(RUN_DIR / "run_config.json", "w") as f:
        json.dump(static_metadata, f, indent=2)

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    history = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(DEVICE)     # (B, 1, H, W)
            targets = batch["target"].to(DEVICE)   # (B, 1)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---------------- Validation ----------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                targets = batch["target"].to(DEVICE)

                logits = model(images)
                loss = loss_fn(logits, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        history.append(epoch_record)

        print(f"[Epoch {epoch}] train={train_loss:.4f}, val={val_loss:.4f}")

    # ------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------
    with open(RUN_DIR / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)

    # ------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------
    torch.save(model.state_dict(), RUN_DIR / "model.pt")

    # ------------------------------------------------------------
    # Evidence report (explicitly non-clinical)
    # ------------------------------------------------------------
    evidence = EvidenceReport()
    evidence.info("Run intent: smoke training")
    evidence.info("Dataset: COCA")
    evidence.info("Annotation type: vector ROIs")
    evidence.info("Task: slice-based binary classification")
    evidence.info("No clinical performance claims implied")

    evidence.save(RUN_DIR / "evidence.json")


if __name__ == "__main__":
    main()
