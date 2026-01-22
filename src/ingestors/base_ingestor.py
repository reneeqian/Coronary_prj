# src/ingestors/base_ingestor.py

from abc import ABC, abstractmethod

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
