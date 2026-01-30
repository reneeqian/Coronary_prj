from typing import Sequence, Dict
from torch.utils.data import Dataset

from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample


class LazyPatientDataset(Dataset):
    """
    Lazily loads PatientSamples via an ingestor and adapts them to tensors.

    Assumptions:
    - Dataset may be very large
    - Only one patient is loaded into memory at a time
    """

    def __init__(
        self,
        *,
        patient_ids: Sequence[str],
        ingestor,
        adapter,
    ):
        self.patient_ids = list(patient_ids)
        self.ingestor = ingestor
        self.adapter = adapter

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        patient_id = self.patient_ids[idx]

        # --- Lazy load ---
        sample: PatientSample = self.ingestor.load_patient(patient_id)

        # --- Adapt to tensors ---
        return self.adapter(sample)
