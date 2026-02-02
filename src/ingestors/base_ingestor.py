# src/ingestors/base_ingestor.py

from abc import ABC, abstractmethod
from src.medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample


class BaseIngestor(ABC):
    """Abstract base class for dataset ingestors."""

    @abstractmethod
    def ingest_patient(self, patient_path):
        """Ingest a single patient directory into a PatientSample."""
        raise NotImplementedError

    @abstractmethod
    def ingest_dataset(self, dataset_root):
        """Iterate over dataset and yield PatientSamples."""
        raise NotImplementedError

    @abstractmethod
    def load_patient(self, patient_id: str) -> PatientSample:
        """Load a single patient lazily."""
        raise NotImplementedError
