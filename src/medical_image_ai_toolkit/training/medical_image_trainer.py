from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..evidence.evidence_report import EvidenceReport


class MedicalImageTrainer:
    """
    Owns the full lifecycle of a single training run:
    - Training / validation loops
    - Progress reporting
    - Metrics collection
    - Artifact persistence

    Designed for FDA-auditable SaMD development.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        run_dir: Path,
        run_config: Dict[str, Any],
        data_splits: Dict[str, Any],
        show_training_plot: bool = False,
        save_training_plot: bool = True,
        print_every_n_batches: int = 50,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

        self.run_dir = run_dir
        self.run_config = run_config
        self.data_splits = data_splits

        self.show_training_plot = show_training_plot
        self.save_training_plot = save_training_plot
        self.print_every_n_batches = print_every_n_batches

        self.history: List[Dict[str, Any]] = []

        self.run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def train(
        self,
        *,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
    ) -> None:
        self._save_static_artifacts()

        for epoch in range(num_epochs):
            epoch_record = self._run_epoch(
                epoch=epoch,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
            )
            self.history.append(epoch_record)

        self._save_dynamic_artifacts()
        self._save_evidence_report()
        self._maybe_plot_training_history()

    # ------------------------------------------------------------------
    # EPOCH LOGIC
    # ------------------------------------------------------------------

    def _run_epoch(
        self,
        *,
        epoch: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
    ) -> Dict[str, Any]:
        epoch_start = time.time()

        print(f"\n🟢 Epoch {epoch+1}/{num_epochs} — training")
        train_loss = self._train_one_epoch(train_loader, epoch_start)

        print(f"🔵 Epoch {epoch+1}/{num_epochs} — validation")
        val_loss = self._validate(val_loader)

        epoch_time = time.time() - epoch_start

        print(
            f"✅ Epoch {epoch+1}/{num_epochs} complete | "
            f"train={train_loss:.4f} | "
            f"val={val_loss:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        return {
            "epoch": epoch,
            "train": {"loss": train_loss},
            "val": {"loss": val_loss},
            "epoch_time_sec": epoch_time,
        }

    def _train_one_epoch(self, loader: DataLoader, epoch_start: float) -> float:
        self.model.train()
        total_loss = 0.0
        last_print = epoch_start

        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(self.device)
            targets = batch["target"].to(self.device).view(-1, 1)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.loss_fn(logits, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if self.print_every_n_batches > 0 and batch_idx % self.print_every_n_batches == 0:
                now = time.time()
                print(
                    f"  ⏱ batch {batch_idx:5d}/{len(loader)} | "
                    f"loss={loss.item():.4f} | "
                    f"+{now - last_print:.1f}s"
                )
                last_print = now

        return total_loss / len(loader)

    def _validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                targets = batch["target"].to(self.device).view(-1, 1)
                logits = self.model(images)
                loss = self.loss_fn(logits, targets)
                total_loss += loss.item()

        return total_loss / len(loader)

    # ------------------------------------------------------------------
    # ARTIFACTS
    # ------------------------------------------------------------------

    def _save_static_artifacts(self) -> None:
        with open(self.run_dir / "run_config.json", "w") as f:
            json.dump(self.run_config, f, indent=2)

        with open(self.run_dir / "data_splits.json", "w") as f:
            json.dump(self.data_splits, f, indent=2)

    def _save_dynamic_artifacts(self) -> None:
        with open(self.run_dir / "metrics.json", "w") as f:
            json.dump(self.history, f, indent=2)

        torch.save(self.model.state_dict(), self.run_dir / "model.pt")

    def _save_evidence_report(self) -> None:
        evidence = EvidenceReport(
            subject=self.run_config.get("run_intent", "Training run")
        )
        evidence.info("Trainer: MedicalImageTrainer")
        evidence.info("Non-clinical development run")
        evidence.info("No performance claims implied")
        evidence.save(self.run_dir / "evidence.json")

    def _maybe_plot_training_history(self) -> None:
        if not (self.show_training_plot or self.save_training_plot):
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️ matplotlib not available — skipping plot")
            return

        epochs = [h["epoch"] + 1 for h in self.history]
        train_losses = [h["train"]["loss"] for h in self.history]
        val_losses = [h["val"]["loss"] for h in self.history]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label="Train")
        plt.plot(epochs, val_losses, label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True)

        if self.save_training_plot:
            path = self.run_dir / "training_curve.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"📈 Training curve saved to {path}")

        if self.show_training_plot:
            plt.show()
        else:
            plt.close()
