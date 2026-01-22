# src/ingestors/coca_gated_ingestor.py

from pathlib import Path
from typing import Tuple, Dict, List, Iterator
import numpy as np
from lxml import etree
import pydicom

from src.ingestors.base_ingestor import BaseIngestor
from src.datasets.coronary_ct_dataset import PatientSample
from src.annotations.annotation_bundle import AnnotationBundle  # if/when created


class COCAGatedIngestor(BaseIngestor):
    """
    Ingestor for COCA gated cardiac CT dataset.

    Responsibilities:
    - Load gated CT DICOM series into a 3D volume
    - Load slice-wise CAC polygon annotations
    - Produce one PatientSample per patient
    """

    def __init__(self, *,
                 enforce_sorted_slices: bool = True,
                 load_metadata: bool = True):
        self.enforce_sorted_slices = enforce_sorted_slices
        self.load_metadata = load_metadata

    # -------------------------
    # Public API
    # -------------------------

    def ingest_dataset(self, dataset_root: Path) -> Iterator[PatientSample]:
        """
        Iterate over all patients in the dataset directory.
        """
        self.dataset_root = Path(dataset_root)

        for patient_dir in self._iter_patient_dirs(self.dataset_root):
            yield self.ingest_patient(patient_dir)

    def ingest_patient(self, patient_dir: Path) -> PatientSample:
        """
        Ingest a single COCA gated patient.
        """
        patient_dir = Path(patient_dir)
        patient_id = patient_dir.name

        series_dir = self._resolve_gated_series_dir(patient_dir)

        volume, spacing, metadata = self._load_image_volume(series_dir)

        annotations = self._load_annotations(
            dataset_root=self.dataset_root,
            patient_id=patient_id,
        )

        return PatientSample(
            image_volume=volume,
            annotations=annotations,
            spacing=spacing,
            metadata=metadata,
            patient_id=patient_id,
        )

    # -------------------------
    # Internal helpers
    # -------------------------

    def _iter_patient_dirs(self, dataset_root: Path) -> List[Path]:
        """
        Return patient root directories (e.g. dataset_root/patient/0).
        """
        patient_root = dataset_root / "patient"

        if not patient_root.exists():
            raise FileNotFoundError(f"Patient root not found: {patient_root}")

        return sorted(
            p for p in patient_root.iterdir()
            if p.is_dir()
        )

    def _resolve_gated_series_dir(self, patient_dir: Path) -> Path:
        """
        Resolve the gated CT series directory for a patient.

        Assumes exactly one gated series directory exists.
        """
        series_dirs = [p for p in patient_dir.iterdir() if p.is_dir()]

        if len(series_dirs) == 0:
            raise FileNotFoundError(f"No series directory found in {patient_dir}")

        if len(series_dirs) > 1:
            raise RuntimeError(
                f"Multiple series directories found in {patient_dir}: {series_dirs}"
            )

        return series_dirs[0]

    def _load_image_volume(
        self,
        series_dir: Path,
    ) -> Tuple[np.ndarray, Tuple[float, float, float], dict]:
        dicom_files = sorted(series_dir.glob("*.dcm"))
        if not dicom_files:
            raise FileNotFoundError(f"No DICOM files found in {series_dir}")

        slices = [pydicom.dcmread(f) for f in dicom_files]

        try:
            slices.sort(key=lambda ds: int(ds.InstanceNumber))
        except Exception as e:
            raise RuntimeError("Failed to sort DICOMs by InstanceNumber") from e

        volume = np.stack([ds.pixel_array for ds in slices]).astype(np.float32)

        slope = float(getattr(slices[0], "RescaleSlope", 1.0))
        intercept = float(getattr(slices[0], "RescaleIntercept", 0.0))
        volume = volume * slope + intercept

        dy, dx = map(float, slices[0].PixelSpacing)
        dz = float(slices[0].SliceThickness)

        spacing = (dz, dy, dx)

        metadata = {
            "manufacturer": getattr(slices[0], "Manufacturer", None),
            "series_description": getattr(slices[0], "SeriesDescription", None),
            "slice_count": len(slices),
        }

        return volume, spacing, metadata

    def _load_annotations(
        self,
        dataset_root: Path,
        patient_id: str,
    ) -> AnnotationBundle:
        xml_path = dataset_root / "calcium_xml" / f"{patient_id}.xml"

        if not xml_path.exists():
            return AnnotationBundle(vector_rois=None)

        tree = etree.parse(str(xml_path))
        root = tree.getroot()

        vector_rois: Dict[int, list] = {}

        for image_dict in root.findall(".//dict"):
            keys = [k.text for k in image_dict.findall("key")]
            if "ImageIndex" not in keys:
                continue

            image_index = int(image_dict.findall("integer")[0].text)

            for roi_dict in image_dict.findall("dict"):
                strings = roi_dict.findall("string")
                if not strings:
                    continue

                label = strings[0].text

                contour = []
                for elem in roi_dict.iter():
                    if elem.tag == "string" and elem.text.startswith("("):
                        x, y = elem.text.strip("()").split(",")
                        contour.append([float(x), float(y)])

                if contour:
                    roi = VectorROI(
                        slice_index=image_index,
                        contour_px=np.array(contour),
                        label="CAC",
                        metadata={"artery": label},
                    )
                    vector_rois.setdefault(image_index, []).append(roi)

        return AnnotationBundle(vector_rois=vector_rois)

