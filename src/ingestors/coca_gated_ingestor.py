"""
COCA Gated CT Ingestor

Transforms raw COCA gated CT studies into a CanonicalCACVolume.

This module is intentionally explicit and stage-based to support:
- Traceability
- Validation
- Dataset evolution
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os
import numpy as np
import xml.etree.ElementTree as ET
import pydicom


# ---------------------------------------------------------------------
# Base Interfaces
# ---------------------------------------------------------------------

class BaseIngestor(ABC):
    @abstractmethod
    def ingest(self, study_path: str):
        pass


# ---------------------------------------------------------------------
# Canonical Data Structures
# ---------------------------------------------------------------------

@dataclass
class CACAnnotation:
    """
    Slice-level CAC annotation.
    """
    slice_index: int
    contours_px: List[np.ndarray]   # each contour is shape (N, 2)
    artery_name: Optional[str] = None


@dataclass
class CanonicalCACVolume:
    """
    Canonical internal representation of a CAC CT volume.
    """
    image_volume: np.ndarray                  # shape (Z, Y, X)
    spacing_mm: Tuple[float, float, float]    # (dz, dy, dx)
    orientation: str
    annotations: Dict[int, List[CACAnnotation]]
    patient_id: str
    dataset_id: str
    acquisition_type: str
    metadata: dict


# ---------------------------------------------------------------------
# COCA Gated Ingestor
# ---------------------------------------------------------------------

class COCAGatedIngestor(BaseIngestor):
    """
    Ingestor for COCA gated CT datasets.
    """

    DATASET_ID = "COCA"
    ACQUISITION_TYPE = "gated"

    # ----------------------------
    # Public API
    # ----------------------------

    def ingest(self, study_path: str) -> CanonicalCACVolume:
        """
        Main ingestion entry point.
        """
        dicom_files, annotation_files = self._discover_files(study_path)

        volume, dicom_meta = self._load_dicom_volume(dicom_files)
        spacing = self._extract_spacing(dicom_meta)
        orientation = self._canonicalize_orientation(volume, dicom_meta)

        annotations = self._parse_annotations(annotation_files)

        self._validate(volume, spacing, annotations)

        patient_id = dicom_meta.get("PatientID", "unknown")

        return CanonicalCACVolume(
            image_volume=volume,
            spacing_mm=spacing,
            orientation=orientation,
            annotations=annotations,
            patient_id=patient_id,
            dataset_id=self.DATASET_ID,
            acquisition_type=self.ACQUISITION_TYPE,
            metadata=dicom_meta,
        )

    # ----------------------------
    # Pipeline Stages
    # ----------------------------

    def _discover_files(self, study_path: str):
        """
        Locate DICOM and annotation files within a study directory.
        """
        dicom_files = []
        annotation_files = []

        for root, _, files in os.walk(study_path):
            for fname in files:
                path = os.path.join(root, fname)
                if fname.lower().endswith(".dcm"):
                    dicom_files.append(path)
                elif fname.lower().endswith(".xml"):
                    annotation_files.append(path)

        if not dicom_files:
            raise ValueError("No DICOM files found in study")

        return dicom_files, annotation_files

    def _load_dicom_volume(self, dicom_files):
        """
        Load and stack DICOM slices into a 3D volume.
        """
        slices = []
        metas = []

        for path in dicom_files:
            ds = pydicom.dcmread(path)
            slices.append(ds.pixel_array.astype(np.float32))
            metas.append(ds)

        # Sort by InstanceNumber
        order = np.argsort([int(m.InstanceNumber) for m in metas])
        volume = np.stack([slices[i] for i in order], axis=0)

        meta = {
            "PatientID": getattr(metas[0], "PatientID", "unknown"),
            "SeriesInstanceUID": getattr(metas[0], "SeriesInstanceUID", None),
            "SliceThickness": float(metas[0].SliceThickness),
            "PixelSpacing": tuple(map(float, metas[0].PixelSpacing)),
            "NumSlices": volume.shape[0],
        }

        return volume, meta

    def _extract_spacing(self, meta) -> Tuple[float, float, float]:
        """
        Extract voxel spacing in mm.
        """
        dz = meta["SliceThickness"]
        dy, dx = meta["PixelSpacing"]
        return (dz, dy, dx)

    def _canonicalize_orientation(self, volume, meta) -> str:
        """
        Canonicalize orientation.

        NOTE:
        For COCA gated CT, orientation is assumed axial.
        This is a deliberate stub for future affine-based handling.
        """
        return "RAS"

    def _parse_annotations(self, annotation_files) -> Dict[int, List[CACAnnotation]]:
        """
        Parse CAC annotations from XML files.
        """
        annotations: Dict[int, List[CACAnnotation]] = {}

        for path in annotation_files:
            tree = ET.parse(path)
            root = tree.getroot()

            for roi in root.findall(".//ROI"):
                slice_idx = int(roi.findtext("SliceIndex"))
                artery = roi.findtext("Artery")

                contour_points = []
                for pt in roi.findall(".//Point"):
                    x = float(pt.get("x"))
                    y = float(pt.get("y"))
                    contour_points.append([x, y])

                contour = np.array(contour_points)

                ann = CACAnnotation(
                    slice_index=slice_idx,
                    contours_px=[contour],
                    artery_name=artery,
                )

                annotations.setdefault(slice_idx, []).append(ann)

        return annotations

    # ----------------------------
    # Validation
    # ----------------------------

    def _validate(self, volume, spacing, annotations):
        """
        Lightweight internal validation.
        """
        if volume.ndim != 3:
            raise ValueError("Image volume must be 3D")

        if not np.isfinite(volume).all():
            raise ValueError("Image contains non-finite values")

        if len(spacing) != 3:
            raise ValueError("Spacing must be (dz, dy, dx)")

        num_slices = volume.shape[0]
        for slice_idx in annotations.keys():
            if slice_idx < 0 or slice_idx >= num_slices:
                raise ValueError(
                    f"Annotation slice index out of bounds: {slice_idx}"
                )
