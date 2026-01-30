from typing import Dict, Any
import torch
import torch.nn as nn

from medical_image_ai_toolkit.training.tasks import TaskDefinition


class SmokeBinaryClassificationTask(TaskDefinition):
    """
    Minimal binary classification task intended ONLY for smoke testing.

    This task validates:
    - adapter → model → loss → metrics plumbing
    - artifact capture
    - trainer execution

    It does NOT encode clinical or dataset semantics.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)

    def name(self) -> str:
        return "smoke_binary_classification"

    def build_loss(self) -> nn.Module:
        # Robust, standard choice for binary outputs
        return nn.BCEWithLogitsLoss()

    def compute_metrics(
        self,
        model_outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Computes extremely simple metrics to prove the loop runs.
        """
        with torch.no_grad():
            # Ensure compatible shapes
            logits = model_outputs.view(-1)
            targets = targets.float().view(-1)

            probs = torch.sigmoid(logits)
            preds = (probs >= self.threshold).float()

            accuracy = (preds == targets).float().mean().item()

        return {
            "accuracy": accuracy,
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "type": "SmokeBinaryClassificationTask",
            "threshold": self.threshold,
            "notes": "Smoke-test task; no clinical meaning",
        }
