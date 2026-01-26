# medimg_training/datamodules/tensor_datamodule.py

from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset


class TensorDataset(Dataset):
    """
    Torch Dataset wrapping pre-converted tensor samples.

    Assumes:
    - All validation has already occurred
    - All tensors are correctly shaped and typed
    """

    def __init__(
        self,
        tensor_samples: List[Dict[str, Any]],
        deterministic: bool = True,
    ):
        if deterministic:
            tensor_samples = sorted(
                tensor_samples,
                key=lambda x: str(x.get("patient_id", ""))
            )

        self.samples = tensor_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
