from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING

import numpy as np
import torch
from medical_image_ai_toolkit.training.task_definition import TrainingTaskDefinition
from skimage.draw import polygon

if TYPE_CHECKING:
    from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample
    from regulatory_tools.evidence.evidence_report import EvidenceReport

_OOD_HU_MIN = -200.0
_OOD_HU_MAX = 400.0


class CoronaryCalciumTask(TrainingTaskDefinition):

    def __init__(self, report: EvidenceReport | None = None):
        self._report = report

    def generate_training_samples(
        self, patient_sample: PatientSample
    ) -> Generator[dict[str, torch.Tensor], None, None]:

        volume = patient_sample.image_volume
        annotations = patient_sample.annotations

        Z, H, W = volume.shape

        vector_rois = None
        if annotations is not None:
            vector_rois = annotations.vector_rois

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

            hu = np.clip(hu_raw, -160.0, 240.0)   # cardiac soft-tissue window (WL=40, WW=400)
            hu = (hu - 40.0) / 200.0              # centre on WL, scale to roughly [-1, 1]
            img = torch.tensor(hu).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)

            mask = np.zeros((H, W), dtype=np.float32)

            if vector_rois and slice_idx in vector_rois:

                for roi in vector_rois[slice_idx]:

                    contour = roi.contour_px
                    if contour is None or len(contour) < 3:
                        continue

                    rr, cc = self._polygon_to_mask(contour, H, W)
                    mask[rr, cc] = 1.0

            mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

            yield {
                "input": img,
                "target": mask_tensor
            }

    def compute_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction, target
        )
        prob = torch.sigmoid(prediction)
        intersection = (prob * target).sum()
        dice = 1.0 - (2.0 * intersection + 1.0) / (prob.sum() + target.sum() + 1.0)
        return 0.5 * bce + 0.5 * dice

    def _polygon_to_mask(self, contour: np.ndarray, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
        x = contour[:, 0]
        y = contour[:, 1]

        rr, cc = polygon(y, x, shape=(H, W))
        return rr, cc
