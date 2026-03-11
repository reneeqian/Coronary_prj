# src/ingestors/base_ingestor.py

from abc import ABC, abstractmethod
from typing import Iterable
from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample


class BaseIngestor(ABC):
    """Abstract base class for dataset ingestors."""
    
    @abstractmethod
    def list_patient_ids(self) -> Iterable[str]:
        raise NotImplementedError

    @abstractmethod
    def load_patient(self, patient_id: str) -> PatientSample:
        raise NotImplementedError

    @abstractmethod
    def get_sample(self, patient_id: str):
        raise NotImplementedError 