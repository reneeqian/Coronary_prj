from abc import ABC, abstractmethod
from typing import Iterable, Dict, List
import hashlib


class SplitStrategy(ABC):
    @abstractmethod
    def split(self, patient_ids: Iterable[str]) -> Dict[str, List[str]]:
        """
        Returns:
            {
              "train": [...],
              "val": [...]
            }
        """
        ...


class HashPatientIDSplit(SplitStrategy):
    def __init__(self, val_fraction: float = 0.2):
        if not (0.0 < val_fraction < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        self.val_fraction = val_fraction

    def split(self, patient_ids: Iterable[str]) -> Dict[str, List[str]]:
        train, val = [], []

        for pid in patient_ids:
            h = hashlib.sha256(pid.encode("utf-8")).hexdigest()
            bucket = int(h[:8], 16) / 0xFFFFFFFF

            if bucket < self.val_fraction:
                val.append(pid)
            else:
                train.append(pid)

        return {
            "train": sorted(train),
            "val": sorted(val),
        }
