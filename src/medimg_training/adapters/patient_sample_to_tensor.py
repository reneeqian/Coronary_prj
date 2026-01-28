from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from src.dataobjects.patient_sample import PatientSample
from src.validators.patient_sample_contract import validate_patient_sample


class PatientSampleTensorAdapter:
    """
    Converts a validated PatientSample into PyTorch tensors.

    This class is intentionally stateless and deterministic.
    """

    def __init__(
        self,
        *,
        normalize: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        require_annotations: bool = False,
    ):
        self.normalize = normalize
        self.dtype = dtype
        self.device = device
        self.require_annotations = require_annotations

    def __call__(self, sample: PatientSample) -> Dict[str, object]:
        """
        Convert a PatientSample into model-ready tensors.

        Returns
        -------
        dict with keys:
            image: torch.Tensor (1, Z, Y, X)
            target: torch.Tensor or None
            metadata: dict
        """
        # --- Validate ---
        report = validate_patient_sample(
            sample,
            require_annotations=self.require_annotations,
        )

        if report.has_errors:
            raise ValueError(
                "PatientSample failed validation:\n"
                + report.to_string()
            )

        # --- Image tensor ---
        image = self._image_to_tensor(sample.image_volume)

        # --- Target tensor (task-dependent; placeholder for now) ---
        target = self._build_target(sample)

        # --- Metadata ---
        metadata = {
            "patient_id": sample.patient_id,
            "spacing": sample.spacing,
            **(sample.metadata or {}),
        }

        return {
            "image": image,
            "target": target,
            "metadata": metadata,
        }

    # -------------------------
    # Internal helpers
    # -------------------------

    def _image_to_tensor(self, volume: np.ndarray) -> torch.Tensor:
        """
        Convert image volume to torch tensor.

        Input:  (Z, Y, X)
        Output: (1, Z, Y, X)
        """
        tensor = torch.from_numpy(volume).to(self.dtype)

        if self.normalize:
            tensor = self._normalize_ct(tensor)

        # Add channel dimension
        tensor = tensor.unsqueeze(0)

        if self.device is not None:
            tensor = tensor.to(self.device)

        return tensor

    def _normalize_ct(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Simple CT normalization.

        This is intentionally conservative and explainable.
        """
        # Typical CT window for cardiac structures
        min_hu = -1000.0
        max_hu = 1000.0

        tensor = torch.clamp(tensor, min_hu, max_hu)
        tensor = (tensor - min_hu) / (max_hu - min_hu)

        return tensor

    def _build_target(
        self, sample: PatientSample
    ) -> Optional[torch.Tensor]:
        """
        Placeholder target builder.

        For now:
        - If annotations exist → binary presence label
        - Else → None
        """
        ann = sample.annotations

        if ann is None or not ann.vector_rois:
            return None

        # Example: CAC present / not present
        target = torch.tensor(1, dtype=torch.long)

        if self.device is not None:
            target = target.to(self.device)

        return target
