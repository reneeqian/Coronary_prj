from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable, Dict, Any

import json
import datetime
import torch

from medimg_training.evidence.evidence_report import EvidenceReport


class MedicalImageTrainer:
    """
    Generic medical imaging training orchestrator.

    This class owns:
    - dataset wiring (via factories)
    - training configuration
    - run metadata
    - artifact capture

    It does NOT:
    - implement dataset-specific ingestion
    - preload PatientSample objects
    - assume a specific model architecture
    """

    def __init__(
        self,
        *,
        run_root: Path,
        dataset_factory: Callable[[], torch.utils.data.Dataset],
        model: torch.nn.Module,
        optimizer_factory: Callable[[torch.nn.Module], torch.optim.Optimizer],
        loss_fn: Callable,
        device: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.run_root = run_root
        self.dataset_factory = dataset_factory
        self.model = model
        self.optimizer_factory = optimizer_factory
        self.loss_fn = loss_fn
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.metadata = metadata or {}

        self.run_id = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.run_root / f"run_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=False)

        self.evidence = EvidenceReport(subject=f"TrainingRun:{self.run_id}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, *, max_steps: int = 1) -> None:
        """
        Execute a training run.

        Default max_steps=1 supports smoke testing.
        """
        self._log_run_config()

        dataset = self.dataset_factory()
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,  # deterministic for now
        )

        self.model.to(self.device)
        optimizer = self.optimizer_factory(self.model)

        self._train_loop(
            dataloader=dataloader,
            optimizer=optimizer,
            max_steps=max_steps,
        )

        self._save_model()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_loop(
        self,
        *,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        max_steps: int,
    ) -> None:
        self.model.train()

        for step, batch in enumerate(dataloader):
            if step >= max_steps:
                break

            image = batch["image"].to(self.device)
            target = batch.get("target")

            optimizer.zero_grad()
            output = self.model(image)

            if target is not None:
                target = target.to(self.device)
                loss = self.loss_fn(output, target)
                loss.backward()
                optimizer.step()

            self.evidence.info(
                message="Training step completed",
                requirement_id="PROC-TR-01",
                context=f"step={step}",
            )

    def _save_model(self) -> None:
        model_path = self.run_dir / "model.pt"
        torch.save(self.model.state_dict(), model_path)

    def _log_run_config(self) -> None:
        config = {
            "run_id": self.run_id,
            "device": self.device,
            "metadata": self.metadata,
        }

        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
