from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class CACAnnotation:
    slice_index: int
    contours_px: List[np.ndarray]
    artery_name: str | None


@dataclass
class PatientSample:
    image_volume: np.ndarray
    annotations: Dict[int, List[CACAnnotation]]
    spacing: Tuple[float, float, float]
    metadata: dict
    patient_id: str | int


class COCAGatedIngestor:
    """
    Ingestor for the gated portion of the COCA dataset.
    """

    def __init__(self, coca_root: Path):
        self.coca_root = coca_root
        self.gated_root = (
            coca_root
            / "cocacoronarycalciumandchestcts-2"
            / "Gated_release_final"
        )

        if not self.gated_root.exists():
            raise FileNotFoundError(
                f"Gated COCA root not found:\n{self.gated_root}"
            )

    def list_patients(self) -> List[int]:
        """Return available patient IDs."""
        raise NotImplementedError

    def load_patient(self, patient_id: int) -> PatientSample:
        """
        Load a single gated COCA patient and return a standardized PatientSample.
        """
        raise NotImplementedError

    # --- internal helpers (dataset-specific) ---

    def _load_dicom_series(self, series_dir: Path):
        raise NotImplementedError

    def _parse_annotation_xml(self, xml_path: Path):
        raise NotImplementedError
