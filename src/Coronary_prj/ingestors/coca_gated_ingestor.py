from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.dataset import Dataset

from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample




class DatasetStructureError(RuntimeError):
    """Raised when the dataset structure or required contents are invalid."""
    pass


class COCAGatedIngestor:
    """
    Ingests COCA gated CT dataset into PatientSample objects.

    Public API guarantees:
        - All dataset structure or integrity issues raise DatasetStructureError.
        - No raw FileNotFoundError or RuntimeError escapes the boundary.
    """

    def __init__(self, dataset_root: Path):
        self.dataset_root = Path(dataset_root)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def list_patient_ids(self) -> List[str]:
        try:
            if not self.dataset_root.exists():
                raise DatasetStructureError(
                    f"Dataset root does not exist: {self.dataset_root}"
                )

            patient_dirs = [
                p.name for p in self.dataset_root.iterdir() if p.is_dir()
            ]

            if not patient_dirs:
                raise DatasetStructureError(
                    f"No patient directories found in {self.dataset_root}"
                )

            return sorted(patient_dirs)

        except OSError as e:
            raise DatasetStructureError(str(e)) from e

    def ingest_patient(self, patient_id: str) -> PatientSample:
        try:
            patient_dir = self.dataset_root / patient_id
            if not patient_dir.exists():
                raise DatasetStructureError(
                    f"Patient directory not found: {patient_dir}"
                )

            series_dir = self._resolve_gated_series_dir(patient_dir)
            volume, spacing, metadata = self._load_image_volume(series_dir)
            annotations = self._load_annotations(patient_dir, volume.shape[0])

            return PatientSample(
                patient_id=patient_id,
                image_volume=volume,
                spacing=spacing,
                annotations=annotations,
                metadata=metadata,
            )

        except DatasetStructureError:
            # Preserve intentional domain errors
            raise
        except Exception as e:
            # Convert all unexpected failures to domain-safe error
            raise DatasetStructureError(str(e)) from e

    def ingest_dataset(self) -> List[PatientSample]:
        try:
            patient_ids = self.list_patient_ids()
            return [self.ingest_patient(pid) for pid in patient_ids]
        except DatasetStructureError:
            raise
        except Exception as e:
            raise DatasetStructureError(str(e)) from e

    # ------------------------------------------------------------------
    # INTERNAL HELPERS (may raise FileNotFoundError/RuntimeError)
    # These MUST NOT leak outside public API.
    # ------------------------------------------------------------------

    def _resolve_gated_series_dir(self, patient_dir: Path) -> Path:
        """
        Locate the gated CT DICOM series directory.
        """

        series_dirs = [p for p in patient_dir.iterdir() if p.is_dir()]
        if not series_dirs:
            raise DatasetStructureError(
                f"No series directories found in {patient_dir}"
            )

        # For COCA, assume first directory is gated series
        return series_dirs[0]

    def _load_image_volume(
        self, series_dir: Path
    ) -> Tuple[np.ndarray, Tuple[float, float, float], Dict]:

        dicom_files = sorted(series_dir.glob("*.dcm"))
        if not dicom_files:
            raise DatasetStructureError(
                f"No DICOM files found in {series_dir}"
            )

        slices = []
        z_positions = []

        for f in dicom_files:
            try:
                ds = pydicom.dcmread(f)

                if not hasattr(ds, "ImagePositionPatient"):
                    raise DatasetStructureError(
                        f"Missing ImagePositionPatient in {f}"
                    )

                z = float(ds.ImagePositionPatient[2])
                z_positions.append(z)

                image = ds.pixel_array.astype(np.float32)

                # Apply rescale if available
                slope = float(getattr(ds, "RescaleSlope", 1.0))
                intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                image = image * slope + intercept

                slices.append(image)

            except (InvalidDicomError, AttributeError, KeyError, ValueError) as e:
                raise DatasetStructureError(
                    f"Invalid or corrupt DICOM file: {f}"
                ) from e

        if not slices:
            raise DatasetStructureError(
                f"No valid DICOM slices loaded from {series_dir}"
            )

        # Sort slices by z position
        sorted_indices = np.argsort(z_positions)
        volume = np.stack([slices[i] for i in sorted_indices], axis=0)

        # Extract spacing from first slice
        ds0 = pydicom.dcmread(dicom_files[0])
        try:
            pixel_spacing = tuple(map(float, ds0.PixelSpacing))
            slice_thickness = float(ds0.SliceThickness)
        except Exception as e:
            raise DatasetStructureError(
                f"Missing spacing metadata in {series_dir}"
            ) from e

        spacing = (slice_thickness, pixel_spacing[0], pixel_spacing[1])

        metadata = {
            "series_dir": str(series_dir),
            "num_slices": volume.shape[0],
        }

        return volume, spacing, metadata

    def _load_annotations(
        self, patient_dir: Path, num_slices: int
    ) -> Dict:

        annotations_file = patient_dir / "annotations.txt"

        if not annotations_file.exists():
            # Gracefully allow no annotations
            return {}

        annotations: Dict[int, str] = {}

        try:
            with open(annotations_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split(",")
                    if len(parts) != 2:
                        raise DatasetStructureError(
                            f"Malformed annotation line: {line}"
                        )

                    slice_idx = int(parts[0]) - 1  # COCA is 1-based
                    label = parts[1].strip()

                    if not (0 <= slice_idx < num_slices):
                        raise DatasetStructureError(
                            f"Annotation slice index out of bounds: {slice_idx}"
                        )

                    annotations[slice_idx] = label

        except Exception as e:
            raise DatasetStructureError(
                f"Failed to parse annotations in {annotations_file}"
            ) from e

        return annotations
