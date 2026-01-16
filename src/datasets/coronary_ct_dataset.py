"""
coronary_ct_dataset.py

PyTorch-compatible dataset for coronary CT angiography segmentation.

Responsibilities:
- Load NIfTI image + label pairs
- Enforce data contract via validators
- Normalize orientation, spacing, and intensity
- Return tensors ready for modeling

This dataset is intentionally strict to prevent silent data bugs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

from .validators import (
    enforce_ras_orientation,
    resample_to_spacing,
    enforce_data_contract,
)


class CoronaryCTDataset(Dataset):
    """
    Coronary CT angiography dataset for artery segmentation.

    Expected directory structure:

    root_dir/
      images/
        case_001.nii.gz
        case_002.nii.gz
      labels/
        case_001.nii.gz
        case_002.nii.gz
        
    Implements:
    - FR-001 Dataset Ingestion
    - FR-003 Dataset Access API
    """

    def __init__(
        self,
        root_dir: str | Path,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        hu_clip: Tuple[int, int] = (-1000, 1000),
        normalize: bool = True,
        transform: Optional[callable] = None,
    ):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "images"
        self.label_dir = self.root_dir / "labels"

        self.target_spacing = target_spacing
        self.hu_clip = hu_clip
        self.normalize = normalize
        self.transform = transform

        self.samples = self._collect_samples()

    # -------------------------
    # Dataset protocol
    # -------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label_path = self.samples[idx]

        # Load NIfTI files
        image_nifti = nib.load(str(image_path))
        label_nifti = nib.load(str(label_path))

        # Enforce canonical orientation (RAS)
        image_nifti = enforce_ras_orientation(image_nifti)
        label_nifti = enforce_ras_orientation(label_nifti)

        # Convert to numpy arrays
        image = image_nifti.get_fdata(dtype=np.float32)
        label = label_nifti.get_fdata(dtype=np.int16)

        # Extract original voxel spacing (z, y, x)
        original_spacing = image_nifti.header.get_zooms()[:3]

        # Resample to target spacing if needed
        if not np.allclose(original_spacing, self.target_spacing):
            image, label = resample_to_spacing(
                image=image,
                label=label,
                original_spacing=original_spacing,
                target_spacing=self.target_spacing,
            )

        # Intensity preprocessing
        image = self._process_intensity(image)

        # Enforce data contract (fail fast)
        enforce_data_contract(image, label)

        # Convert to torch tensors
        image = torch.from_numpy(image).unsqueeze(0)  # (1, D, H, W)
        label = torch.from_numpy(label).long()

        # Optional MONAI / torchvision transforms
        if self.transform is not None:
            image, label = self.transform(image, label)

        return {
            "image": image,
            "label": label,
            "id": image_path.stem,
        }

    # -------------------------
    # Internal helpers
    # -------------------------

    def _collect_samples(self) -> List[Tuple[Path, Path]]:
        """
        Collect matching image/label pairs and verify correspondence.
        """
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        image_files = sorted(self.image_dir.glob("*.nii*"))
        label_files = sorted(self.label_dir.glob("*.nii*"))

        label_map = {f.name: f for f in label_files}

        samples = []
        for img_path in image_files:
            if img_path.name not in label_map:
                raise ValueError(f"Missing label for image: {img_path.name}")
            samples.append((img_path, label_map[img_path.name]))

        if not samples:
            raise RuntimeError("No valid image/label pairs found")

        return samples

    def _process_intensity(self, image: np.ndarray) -> np.ndarray:
        """
        Apply HU clipping and optional normalization.
        """
        # Clip HU range
        image = np.clip(image, self.hu_clip[0], self.hu_clip[1])

        if self.normalize:
            mean = image.mean()
            std = image.std()
            if std > 0:
                image = (image - mean) / std
            else:
                raise ValueError("Image has zero standard deviation")

        return image
