from pathlib import Path
from typing import Tuple, Dict, List, Iterator
import numpy as np
from lxml import etree
import pydicom
from pydicom.dataset import Dataset

from Coronary_prj.ingestors.base_ingestor import BaseIngestor
from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample
from medical_image_ai_toolkit.dataobjects.annotation_bundle import (
    AnnotationBundle,
    VectorROI,
)


class COCAGatedIngestor(BaseIngestor):
    """
    COCA gated cardiac CT ingestor.

    Dataset assumptions (verified against public COCA releases):
    - One gated CT series per patient
    - Slice ordering defined by ImagePositionPatient (z)
    - Annotations reference annotation-tool slice indices (NOT DICOM order)
    - Annotation indices must be remapped via physical Z
    """

    def __init__(
        self,
        dataset_root: Path,
        *,
        load_metadata: bool = True,
    ):
        self.dataset_root = Path(dataset_root)
        self.load_metadata = load_metadata

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_dataset(self) -> Iterator[PatientSample]:
        patient_root = self.dataset_root / "patient"
        if not patient_root.exists():
            raise FileNotFoundError(f"Missing patient directory: {patient_root}")

        for p in sorted(patient_root.iterdir()):
            if p.is_dir():
                yield self.ingest_patient(p.name)

    def load_patient(self, patient_id: str) -> PatientSample:
        """Toolkit-facing lazy loader"""
        return self.ingest_patient(patient_id)
    
    def get_num_slices(self, patient_id: str) -> int:
        """
        Return number of slices for a patient without loading pixel data.

        This is metadata-only and safe to call during dataset construction.
        """
        patient_dir = self.dataset_root / "patient" / patient_id
        if not patient_dir.exists():
            raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

        series_dir = self._resolve_gated_series_dir(patient_dir)

        return self._count_slices_in_series(series_dir)

    def ingest_patient(self, patient_id: str) -> PatientSample:
        patient_dir = self.dataset_root / "patient" / patient_id
        if not patient_dir.exists():
            raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

        series_dir = self._resolve_gated_series_dir(patient_dir)

        volume, spacing, metadata = self._load_image_volume(series_dir)

        annotations = self._load_annotations(
            dataset_root=self.dataset_root,
            patient_id=patient_id,
        )

        # --- COCA CRITICAL STEP ---
        # Remap annotation slice indices via physical Z
        if annotations.vector_rois:
            self._remap_annotation_slices(
                annotations,
                metadata["slice_z_positions"],
            )

        return PatientSample(
            image_volume=volume,
            annotations=annotations,
            spacing=spacing,
            metadata=metadata,
            patient_id=patient_id,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_gated_series_dir(self, patient_dir: Path) -> Path:
        """
        Resolve the correct gated CT series for a COCA patient.

        Strategy:
        1. Prefer series containing 'BestDiast'
        2. If multiple BestDiast series exist, choose highest % phase
        3. Otherwise, fallback to single available series
        """

        series_dirs = [p for p in patient_dir.iterdir() if p.is_dir()]

        if not series_dirs:
            raise FileNotFoundError(f"No series directory found in {patient_dir}")

        # --- Prefer BestDiast series ---
        best_diast = [
            p for p in series_dirs
            if "BestDiast" in p.name
        ]

        candidates = best_diast if best_diast else series_dirs

        if len(candidates) == 1:
            return candidates[0]

        # --- Extract % phase and pick highest ---
        def extract_phase_percent(path: Path) -> float:
            # Matches "..._73_%" or "..._72_%"
            for token in path.name.split("_"):
                if token.endswith("%"):
                    try:
                        return float(token.strip("%"))
                    except ValueError:
                        pass
            return -1.0

        candidates_sorted = sorted(
            candidates,
            key=extract_phase_percent,
            reverse=True,
        )

        chosen = candidates_sorted[0]

        return chosen

    def _count_slices_in_series(self, series_dir: Path) -> int:
        """
        Count slices in a DICOM series using metadata only.

        Rules:
        - Uses ImagePositionPatient[2] for slice identity
        - Ignores files without spatial metadata
        - Does NOT load pixel data
        """

        dicom_files = list(series_dir.glob("*.dcm"))
        if not dicom_files:
            raise RuntimeError(f"No DICOM files found in {series_dir}")

        z_positions = []

        for path in dicom_files:
            try:
                ds = pydicom.dcmread(
                    path,
                    stop_before_pixels=True,
                    specific_tags=["ImagePositionPatient"],
                )

                if not hasattr(ds, "ImagePositionPatient"):
                    continue

                z = float(ds.ImagePositionPatient[2])
                z_positions.append(z)

            except Exception:
                continue

        if not z_positions:
            raise RuntimeError(
                f"No valid ImagePositionPatient metadata in {series_dir}"
            )

        # Unique Zs protect against duplicate instances
        unique_z = set(z_positions)

        return len(unique_z)

    # ------------------------------------------------------------------
    # Image loading (COCA-correct)
    # ------------------------------------------------------------------

    def _load_image_volume(
        self,
        series_dir: Path,
    ) -> Tuple[np.ndarray, Tuple[float, float, float], dict]:
        """
        Load gated CT volume using COCA-safe rules:
        - ImagePositionPatient defines ordering
        - Z spacing derived from physical positions
        """

        dicom_files = sorted(series_dir.glob("*.dcm"))
        if not dicom_files:
            raise RuntimeError(f"No DICOM files found in {series_dir}")

        datasets: List[Dataset] = []
        pixel_arrays: List[np.ndarray] = []
        z_positions: List[float] = []

        for path in dicom_files:
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=False)

                if not hasattr(ds, "ImagePositionPatient"):
                    continue

                arr = ds.pixel_array.astype(np.float32)

                z = float(ds.ImagePositionPatient[2])

                datasets.append(ds)
                pixel_arrays.append(arr)
                z_positions.append(z)

            except Exception:
                continue

        if not datasets:
            raise RuntimeError(f"No valid image slices in {series_dir}")

        # --- Enforce consistent in-plane shape ---
        shapes = {arr.shape for arr in pixel_arrays}
        if len(shapes) != 1:
            raise RuntimeError(f"Inconsistent slice shapes: {shapes}")

        # --- Sort by physical Z ---
        order = np.argsort(z_positions)
        datasets = [datasets[i] for i in order]
        pixel_arrays = [pixel_arrays[i] for i in order]
        z_positions = [z_positions[i] for i in order]

        # --- Stack volume ---
        volume = np.stack(pixel_arrays, axis=0)

        # --- Rescale to Hounsfield Units ---
        slope = float(getattr(datasets[0], "RescaleSlope", 1.0))
        intercept = float(getattr(datasets[0], "RescaleIntercept", 0.0))
        volume = volume * slope + intercept

        # --- Extract spacing ---
        dy, dx = map(float, datasets[0].PixelSpacing)

        if len(z_positions) > 1:
            dz = float(np.mean(np.diff(z_positions)))
        else:
            dz = float(getattr(datasets[0], "SliceThickness", 1.0))

        spacing = (dz, dy, dx)

        metadata = {
            "slice_count": volume.shape[0],
            "slice_z_positions": z_positions,
            "series_description": getattr(datasets[0], "SeriesDescription", None),
            "manufacturer": getattr(datasets[0], "Manufacturer", None),
        }

        return volume, spacing, metadata

    # ------------------------------------------------------------------
    # Annotation loading
    # ------------------------------------------------------------------

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

        plist_dict = root.find("dict")
        if plist_dict is None:
            return AnnotationBundle(vector_rois=None)

        children = list(plist_dict)

        images_array = None
        for i, elem in enumerate(children):
            if elem.tag == "key" and elem.text == "Images":
                images_array = children[i + 1]
                break

        if images_array is None:
            return AnnotationBundle(vector_rois=None)

        vector_rois: Dict[int, List[VectorROI]] = {}

        for image_dict in images_array.findall("dict"):
            elems = list(image_dict)

            image_index = None
            for i, e in enumerate(elems):
                if e.tag == "key" and e.text == "ImageIndex":
                    image_index = int(elems[i + 1].text)
                    break

            if image_index is None:
                continue

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
                    elif e.tag == "key" and e.text == "Point_px":
                        points_px = roi_elems[i + 1]

                if points_px is None:
                    continue

                contour = []
                for pt in points_px.findall("string"):
                    x, y = pt.text.strip("()").split(",")
                    contour.append([float(x), float(y)])

                if not contour:
                    continue

                roi = VectorROI(
                    slice_index=image_index,  # TEMP: remapped later
                    contour_px=np.asarray(contour),
                    label="CAC",
                    metadata={"artery": name},
                )

                vector_rois.setdefault(image_index, []).append(roi)

        return AnnotationBundle(vector_rois=vector_rois or None)

    # ------------------------------------------------------------------
    # Annotation remapping (COCA critical logic)
    # ------------------------------------------------------------------

    def _remap_annotation_slices(
        self,
        annotations: AnnotationBundle,
        slice_z_positions: List[float],
    ):
        """
        Remap COCA annotation ImageIndex -> actual volume slice index
        using physical Z alignment.
        """

        z_positions = np.asarray(slice_z_positions)
        remapped: Dict[int, List[VectorROI]] = {}

        for ann_idx, rois in annotations.vector_rois.items():
            # COCA ImageIndex is 1-based
            ann_idx0 = ann_idx - 1

            if ann_idx0 < 0 or ann_idx0 >= len(z_positions):
                continue

            target_z = z_positions[ann_idx0]
            slice_idx = int(np.argmin(np.abs(z_positions - target_z)))

            for roi in rois:
                roi.slice_index = slice_idx

            remapped.setdefault(slice_idx, []).extend(rois)

        annotations.vector_rois = remapped
