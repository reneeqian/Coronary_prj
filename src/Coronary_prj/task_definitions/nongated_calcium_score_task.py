from __future__ import annotations

import math
from collections.abc import Generator
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from medical_image_ai_toolkit.training.task_definition import TrainingTaskDefinition

if TYPE_CHECKING:
    from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample
    from regulatory_tools.evidence.evidence_report import EvidenceReport

_OOD_HU_MIN = -200.0
_OOD_HU_MAX = 400.0


class NongatedCalciumScoreTask(TrainingTaskDefinition):
    """
    Regression task for predicting per-vessel Agatston calcium scores from
    non-gated CT slices.

    Each patient yields one training sample per slice. All slices from the
    same patient share the same log1p-transformed target vector:

        target = [log1p(lca), log1p(lad), log1p(lcx), log1p(rca)]  shape (4,)

    The log1p transform linearises the heavily right-skewed Agatston
    distribution (0 to 1000+) and prevents high-score patients from
    dominating the MSE loss.

    At inference, apply ``torch.expm1(prediction)`` to recover predicted
    Agatston scores in the original scale.

    Input preprocessing uses the same cardiac HU window as CoronaryCalciumTask
    (WL=40, WW=400 → clip [-160, 240], normalise to roughly [-1, 1]).
    """

    def __init__(self, report: EvidenceReport | None = None):
        self._report = report

    def generate_training_samples(
        self, patient_sample: PatientSample
    ) -> Generator[dict[str, torch.Tensor], None, None]:

        volume = patient_sample.image_volume
        meta = patient_sample.metadata

        lca = max(0.0, float(meta.get("lca", 0.0)))
        lad = max(0.0, float(meta.get("lad", 0.0)))
        lcx = max(0.0, float(meta.get("lcx", 0.0)))
        rca = max(0.0, float(meta.get("rca", 0.0)))

        target = torch.tensor(
            [math.log1p(lca), math.log1p(lad), math.log1p(lcx), math.log1p(rca)],
            dtype=torch.float32,
        )  # shape (4,)

        Z = volume.shape[0]
        for slice_idx in range(Z):
            hu_raw = volume[slice_idx].astype(np.float32)
            if self._report is not None:
                mean_hu = float(hu_raw.mean())
                if not (_OOD_HU_MIN <= mean_hu <= _OOD_HU_MAX):
                    self._report.warn(
                        f"Slice {slice_idx} mean HU={mean_hu:.1f} outside expected range"
                        " — possible OOD input",
                        "RSK-003",
                    )

            hu = np.clip(hu_raw, -160.0, 240.0)
            hu = (hu - 40.0) / 200.0
            img = torch.tensor(hu).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

            yield {"input": img, "target": target}

    def compute_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(prediction.squeeze(), target.squeeze())
