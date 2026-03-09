from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
import pydicom
import plistlib
from pydicom.errors import InvalidDicomError
from pydicom.dataset import Dataset

from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample
from medical_image_ai_toolkit.dataobjects.annotation_bundle import AnnotationBundle, VectorROI


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
        patient_root = self.dataset_root / "patient"
        try:
            if not patient_root.exists():
                raise DatasetStructureError(
                    f"Patient root does not exist: {patient_root}"
                )

            patient_dirs = [
                p.name for p in patient_root.iterdir() if p.is_dir()
            ]

            if not patient_dirs:
                raise DatasetStructureError(
                    f"No patient directories found in {patient_root}"
                )

            return sorted(patient_dirs)

        except OSError as e:
            raise DatasetStructureError(str(e)) from e

    def ingest_patient(self, patient_id: str) -> PatientSample:
        try:
            patient_dir = self.dataset_root / "patient" / patient_id
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
    # DATA ACCESS API (Dataset-Specific)
    # ------------------------------------------------------------------

    def get_patient(self, patient_id: str) -> PatientSample:
        """
        Load and return a single patient sample.
        """
        return self.ingest_patient(patient_id)

    def get_volume(self, patient_id: str) -> np.ndarray:
        """
        Return 3D volume for a single patient without loading entire dataset.
        """
        sample = self.ingest_patient(patient_id)
        return sample.image_volume

    def get_slice(self, patient_id: str, slice_index: int) -> np.ndarray:
        try:
            patient_dir = self.dataset_root / "patient" / patient_id
            if not patient_dir.exists():
                raise DatasetStructureError(
                    f"Patient directory not found: {patient_dir}"
                )

            series_dir = self._resolve_gated_series_dir(patient_dir)
            sorted_files = self._get_sorted_dicom_files(series_dir)

            if not (0 <= slice_index < len(sorted_files)):
                raise DatasetStructureError(
                    f"Slice index {slice_index} out of bounds "
                    f"(0–{len(sorted_files)-1})"
                )

            ds = pydicom.dcmread(sorted_files[slice_index])
            image = ds.pixel_array.astype(np.float32)

            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))

            return image * slope + intercept

        except DatasetStructureError:
            raise
        except Exception as e:
            raise DatasetStructureError(
                f"Failed to load slice {slice_index} "
                f"for patient {patient_id}"
            ) from e

    def get_sample(self, patient_id):
        patient = self.get_patient(patient_id)

        volume = patient.image_volume

        slice_idx = volume.shape[0] // 2

        return volume[slice_idx]

    # ------------------------------------------------------------------
    # INTERNAL HELPERS (may raise FileNotFoundError/RuntimeError)
    # These MUST NOT leak outside public API.
    # ------------------------------------------------------------------

    def _resolve_gated_series_dir(self, patient_dir: Path) -> Path:
        """
        Locate the gated CT DICOM series directory.
        """

        series_dirs = sorted([p for p in patient_dir.iterdir() if p.is_dir()])
        if not series_dirs:
            raise DatasetStructureError(
                f"No series directories found in {patient_dir}"
            )

        # For COCA, assume first directory is gated series
        return series_dirs[0]

    def _load_image_volume(
        self, series_dir: Path
    ) -> Tuple[np.ndarray, Tuple[float, float, float], Dict]:

        # Deterministic, validated, Z-sorted files
        sorted_files = self._get_sorted_dicom_files(series_dir)

        slices = []

        for f in sorted_files:
            try:
                ds = pydicom.dcmread(f)

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

        # Stack already-sorted slices
        volume = np.stack(slices, axis=0)

        # Extract spacing from first slice in sorted order
        try:
            ds0 = pydicom.dcmread(sorted_files[0], stop_before_pixels=True)

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
    ) -> AnnotationBundle:

        patient_id = patient_dir.name
        xml_file = self.dataset_root / "calcium_xml" / f"{patient_id}.xml"

        if not xml_file.exists():
            return AnnotationBundle(vector_rois=None)

        try:
            with open(xml_file, "rb") as f:
                plist_data = plistlib.load(f)

            images = plist_data.get("Images", [])
            vectors: Dict[int, List[VectorROI]] = {}

            for image_entry in images:
                image_index = image_entry.get("ImageIndex")
                rois = image_entry.get("ROIs", [])

                if image_index is None:
                    continue

                slice_idx = int(image_index) - 1

                if not (0 <= slice_idx < num_slices):
                    raise DatasetStructureError(
                        f"Annotation slice index out of bounds: {slice_idx}"
                    )

                for roi in rois:
                    num_points = roi.get("NumberOfPoints", 0)
                    if num_points == 0:
                        continue

                    artery_name = roi.get("Name", "calcium")
                    point_px_list = roi.get("Point_px", [])
                    if not point_px_list:
                        continue

                    contour_points = []

                    for point_str in point_px_list:
                        point_str = point_str.strip("()")
                        x_str, y_str = point_str.split(",")
                        x = float(x_str)
                        y = float(y_str)
                        contour_points.append([x, y])
                    
                    if len(contour_points) < 3:
                        continue

                    contour_array = np.array(contour_points, dtype=np.float32)
                    
                    # close contour if needed
                    if not np.allclose(contour_array[0], contour_array[-1]):
                        contour_array = np.vstack([contour_array, contour_array[0]])
                        
                    roi_obj = VectorROI(
                        slice_index=slice_idx,
                        contour_px=contour_array,
                        label="calcium",
                        metadata={
                            "artery_name": artery_name,
                            "mean_hu": roi.get("Mean"),
                            "max_hu": roi.get("Max"),
                            "min_hu": roi.get("Min"),
                            "area": roi.get("Area"),
                        }
                    )

                    vectors.setdefault(slice_idx, []).append(roi_obj)

            if not vectors:
                return AnnotationBundle(vector_rois=None)

            return AnnotationBundle(vector_rois=vectors)

        except Exception as e:
            raise DatasetStructureError(
                f"Failed to parse COCA annotation XML: {xml_file}"
            ) from e

    def _get_sorted_dicom_files(self, series_dir: Path) -> List[Path]:
        dicom_files = sorted(series_dir.glob("*.dcm"))
        if not dicom_files:
            raise DatasetStructureError(
                f"No DICOM files found in {series_dir}"
            )

        z_positions = []
        for f in dicom_files:
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True)
            except Exception as e:
                raise DatasetStructureError(
                    f"Invalid DICOM file: {f}"
                ) from e

            if not hasattr(ds, "ImagePositionPatient"):
                raise DatasetStructureError(
                    f"Missing ImagePositionPatient in {f}"
                )

            z_positions.append(float(ds.ImagePositionPatient[2]))

        sorted_indices = np.argsort(z_positions)
        return [dicom_files[i] for i in sorted_indices]