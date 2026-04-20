import torch
import torch.nn as nn
import numpy as np

from medical_image_ai_toolkit.training.task_definition import TrainingTaskDefinition

class CoronaryCalciumTask(TrainingTaskDefinition):

    def generate_training_samples(self, patient_sample):

        volume = patient_sample.image_volume
        annotations = patient_sample.annotations

        Z, H, W = volume.shape

        vector_rois = None
        if annotations is not None:
            vector_rois = annotations.vector_rois

        for slice_idx in range(Z):

            hu = volume[slice_idx].astype(np.float32)
            hu = np.clip(hu, -160.0, 240.0)   # cardiac soft-tissue window (WL=40, WW=400)
            hu = (hu - 40.0) / 200.0          # centre on WL, scale to roughly [-1, 1]
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

    def compute_loss(self, prediction, target):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction, target
        )
        prob = torch.sigmoid(prediction)
        intersection = (prob * target).sum()
        dice = 1.0 - (2.0 * intersection + 1.0) / (prob.sum() + target.sum() + 1.0)
        return 0.5 * bce + 0.5 * dice

    def _polygon_to_mask(self, contour, H, W):
        from skimage.draw import polygon

        x = contour[:, 0]
        y = contour[:, 1]

        rr, cc = polygon(y, x, shape=(H, W))
        return rr, cc