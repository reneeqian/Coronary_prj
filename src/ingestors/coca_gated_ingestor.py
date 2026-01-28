# src/ingestors/coca_gated_ingestor.py

from pathlib import Path
from typing import Tuple, Dict, List, Iterator
import numpy as np
from lxml import etree
import pydicom

from src.ingestors.base_ingestor import BaseIngestor
from src.medimg_training.src.dataobjects.patient_sample import PatientSample
from src.medimg_training.src.dataobjects.annotation_bundle import AnnotationBundle, VectorROI  # if/when created

class COCAGatedIngestor(BaseIngestor):

    def __init__(
        self,
        dataset_root: Path,
        *,
        enforce_sorted_slices: bool = True,
        load_metadata: bool = True,
    ):
        self.dataset_root = Path(dataset_root)
        self.enforce_sorted_slices = enforce_sorted_slices
        self.load_metadata = load_metadata


    # -------------------------
    # Public API
    # -------------------------

    def ingest_dataset(self) -> Iterator[PatientSample]:
        patient_root = self.dataset_root / "patient"

        for p in sorted(patient_root.iterdir()):
            if p.is_dir():
                yield self.ingest_patient(p.name)


    def ingest_patient(self, patient_id: str) -> PatientSample:
        """
        Ingest a single COCA gated patient.
        """
        patient_dir = self.dataset_root / "patient" / patient_id
        if not patient_dir.exists():
            raise FileNotFoundError(f"Patient directory not found: {patient_dir}")
        
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

        # Root plist dict
        plist_dict = root.find("dict")
        if plist_dict is None:
            return AnnotationBundle(vector_rois=None)

        children = list(plist_dict)

        # Find Images array
        images_array = None
        for i, elem in enumerate(children):
            if elem.tag == "key" and elem.text == "Images":
                images_array = children[i + 1]
                break

        if images_array is None:
            return AnnotationBundle(vector_rois=None)

        # Iterate image dicts
        for image_dict in images_array.findall("dict"):
            elems = list(image_dict)

            # --- ImageIndex ---
            image_index = None
            for i, e in enumerate(elems):
                if e.tag == "key" and e.text == "ImageIndex":
                    image_index = int(elems[i + 1].text)
                    break

            if image_index is None:
                continue

            # --- ROIs array ---
            rois_array = None
            for i, e in enumerate(elems):
                if e.tag == "key" and e.text == "ROIs":
                    rois_array = elems[i + 1]
                    break

            if rois_array is None:
                continue

            for roi_dict in rois_array.findall("dict"):
                roi_elems = list(roi_dict)

                name = None
                points_px = None

                for i, e in enumerate(roi_elems):
                    if e.tag == "key" and e.text == "Name":
                        name = roi_elems[i + 1].text
                    if e.tag == "key" and e.text == "Point_px":
                        points_px = roi_elems[i + 1]

                if points_px is None or len(points_px) == 0:
                    continue

                contour = []
                for pt in points_px.findall("string"):
                    x, y = pt.text.strip("()").split(",")
                    contour.append([float(x), float(y)])

                if not contour:
                    continue

                roi = VectorROI(
                    slice_index=image_index,
                    contour_px=np.asarray(contour),
                    label="CAC",
                    metadata={"artery": name},
                )

                vector_rois.setdefault(image_index, []).append(roi)

        return AnnotationBundle(vector_rois=vector_rois)
