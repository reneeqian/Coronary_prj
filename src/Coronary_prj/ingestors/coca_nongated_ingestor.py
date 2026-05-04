from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pydicom
from medical_image_ai_toolkit.dataobjects.annotation_bundle import AnnotationBundle
from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample
from pydicom.errors import InvalidDicomError

from Coronary_prj.ingestors.base_ingestor import BaseIngestor
from Coronary_prj.ingestors.coca_gated_ingestor import DatasetStructureError

if TYPE_CHECKING:
    from regulatory_tools.evidence.evidence_report import EvidenceReport


class COCANongatedIngestor(BaseIngestor):
    """
    Ingests COCA deidentified non-gated CT dataset into PatientSample objects.

    Dataset layout::

        {dataset_root}/
            {patient_id}/       # e.g. "1", "2", ..., "213"
                IM-*.dcm        # DICOM slices directly in patient dir (no series subdir)
            scores.xlsx         # per-vessel Agatston scores; filename column is "{n}A"

    Scores are loaded once at construction and cached. Patient metadata
    includes ``total_score``, ``lca``, ``lad``, ``lcx``, and ``rca`` keys.

    Public API guarantees:
        - All dataset structure or integrity issues raise DatasetStructureError.
        - No raw FileNotFoundError or RuntimeError escapes the boundary.
    """

    def __init__(self, dataset_root: Path, report: EvidenceReport | None = None) -> None:
        self.dataset_root = Path(dataset_root)
        self.report = report
        self._scores = self._load_scores_cache()

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def list_patient_ids(self) -> list[str]:
        try:
            if not self.dataset_root.exists():
                raise DatasetStructureError(
                    f"Dataset root does not exist: {self.dataset_root}"
                )

            patient_dirs = [
                p.name
                for p in self.dataset_root.iterdir()
                if p.is_dir()
            ]

            if not patient_dirs:
                raise DatasetStructureError(
                    f"No patient directories found in {self.dataset_root}"
                )

            return sorted(patient_dirs, key=lambda x: int(x) if x.isdigit() else x)

        except OSError as e:
            raise DatasetStructureError(str(e)) from e

    def load_patient_sample(self, patient_id: str) -> PatientSample:
        try:
            patient_dir = self.dataset_root / patient_id
            if not patient_dir.exists():
                raise DatasetStructureError(
                    f"Patient directory not found: {patient_dir}"
                )

            volume, spacing, metadata = self._load_image_volume(patient_dir)

            score_key = f"{patient_id}A"
            scores_row = self._scores.get(score_key)
            if scores_row is None:
                msg = f"No score entry for patient {patient_id} (key '{score_key}') — using zero scores"
                print(f"Warning: {msg}")
                if self.report:
                    self.report.warn(
                        message=msg,
                        requirement_tag="DAT-016",
                        context=f"patient_id={patient_id}",
                    )
                scores_row = {"total": 0.0, "lca": 0.0, "lad": 0.0, "lcx": 0.0, "rca": 0.0}

            metadata.update(
                total_score=scores_row["total"],
                lca=scores_row["lca"],
                lad=scores_row["lad"],
                lcx=scores_row["lcx"],
                rca=scores_row["rca"],
            )

            if self.report:
                self.report.info(
                    message=f"Loaded patient {patient_id}: {volume.shape[0]} slices, total_score={scores_row['total']}",
                    requirement_tag="DAT-015",
                    context=f"patient_id={patient_id} | num_slices={volume.shape[0]}",
                )

            return PatientSample(
                patient_id=patient_id,
                image_volume=volume,
                spacing=spacing,
                annotations=AnnotationBundle(vector_rois=None),
                metadata=metadata,
            )

        except DatasetStructureError:
            raise
        except Exception as e:
            raise DatasetStructureError(str(e)) from e

    def get_sample(self, patient_id: str) -> tuple[np.ndarray, np.ndarray]:
        sample = self.load_patient_sample(patient_id)
        score_key = f"{patient_id}A"
        scores_row = self._scores.get(score_key, {"lca": 0.0, "lad": 0.0, "lcx": 0.0, "rca": 0.0})
        scores_arr = np.array(
            [scores_row["lca"], scores_row["lad"], scores_row["lcx"], scores_row["rca"]],
            dtype=np.float32,
        )
        return sample.image_volume.astype(np.float32), scores_arr

    def ingest_patient(self, patient_id: str) -> PatientSample:
        return self.load_patient_sample(patient_id)

    def ingest_dataset(self) -> list[PatientSample]:
        return [self.load_patient_sample(pid) for pid in self.list_patient_ids()]

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    def _load_scores_cache(self) -> dict[str, dict[str, float]]:
        import openpyxl

        scores_path = self.dataset_root / "scores.xlsx"
        if not scores_path.exists():
            raise DatasetStructureError(
                f"Scores file not found: {scores_path}"
            )

        try:
            wb = openpyxl.load_workbook(scores_path, read_only=True, data_only=True)
            ws = wb.active

            rows = list(ws.iter_rows(values_only=True))
            wb.close()

            if not rows:
                raise DatasetStructureError(
                    f"Scores file is empty: {scores_path}"
                )

            header = [str(c).strip().lower() if c is not None else "" for c in rows[0]]
            expected = {"filename", "lca", "lad", "lcx", "rca", "total"}
            missing = expected - set(header)
            if missing:
                raise DatasetStructureError(
                    f"scores.xlsx missing columns: {missing}. Found: {header}"
                )

            col = {name: idx for idx, name in enumerate(header)}

            scores: dict[str, dict[str, float]] = {}
            for row in rows[1:]:
                if not row or row[col["filename"]] is None:
                    continue
                key = str(row[col["filename"]]).strip()

                def _f(v: Any) -> float:
                    return float(v) if v is not None else 0.0

                scores[key] = {
                    "total": _f(row[col["total"]]),
                    "lca":   _f(row[col["lca"]]),
                    "lad":   _f(row[col["lad"]]),
                    "lcx":   _f(row[col["lcx"]]),
                    "rca":   _f(row[col["rca"]]),
                }

            return scores

        except DatasetStructureError:
            raise
        except Exception as e:
            raise DatasetStructureError(
                f"Failed to parse scores file: {scores_path}"
            ) from e

    def _load_image_volume(
        self, patient_dir: Path
    ) -> tuple[np.ndarray, tuple[float, float, float], dict[str, Any]]:
        sorted_files = self._get_sorted_dicom_files(patient_dir)

        slices = []
        for f in sorted_files:
            try:
                ds = pydicom.dcmread(f)
                image = ds.pixel_array.astype(np.float32)
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
                f"No valid DICOM slices loaded from {patient_dir}"
            )

        volume = np.stack(slices, axis=0)

        try:
            ds0 = pydicom.dcmread(sorted_files[0], stop_before_pixels=True)
            pixel_spacing = tuple(map(float, ds0.PixelSpacing))
            slice_thickness = float(ds0.SliceThickness)
            spacing: tuple[float, float, float] = (
                slice_thickness, pixel_spacing[0], pixel_spacing[1]
            )
        except Exception:
            msg = f"Missing spacing metadata in {patient_dir} — using (1.0, 1.0, 1.0)"
            print(f"Warning: {msg}")
            if self.report:
                self.report.warn(
                    message=msg,
                    requirement_tag="DAT-015",
                    context=f"patient_dir={patient_dir}",
                )
            spacing = (1.0, 1.0, 1.0)

        metadata: dict[str, Any] = {
            "series_dir": str(patient_dir),
            "num_slices": volume.shape[0],
        }

        return volume, spacing, metadata

    def _get_sorted_dicom_files(self, patient_dir: Path) -> list[Path]:
        dicom_files = sorted(patient_dir.glob("*.dcm"))
        if not dicom_files:
            raise DatasetStructureError(
                f"No DICOM files found in {patient_dir}"
            )

        z_positions = []
        valid_files = []
        skipped = []

        for f in dicom_files:
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True)
            except Exception as e:
                raise DatasetStructureError(f"Invalid DICOM file: {f}") from e

            if not hasattr(ds, "ImagePositionPatient"):
                skipped.append(f.name)
                continue

            z_positions.append(float(ds.ImagePositionPatient[2]))
            valid_files.append(f)

        if skipped:
            msg = f"Skipped {len(skipped)} DICOM file(s) missing ImagePositionPatient in {patient_dir.name}"
            print(f"Warning: {msg}: {skipped}")
            if self.report:
                self.report.warn(
                    message=msg,
                    requirement_tag="DAT-015",
                    context=f"patient_dir={patient_dir} | skipped={skipped}",
                )

        if not valid_files:
            raise DatasetStructureError(
                f"No DICOM files with valid ImagePositionPatient found in {patient_dir}"
            )

        sorted_indices = np.argsort(z_positions)
        return [valid_files[i] for i in sorted_indices]
