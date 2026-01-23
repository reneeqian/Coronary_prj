from typing import Optional
import numpy as np

from src.validators.validation_report import ValidationReport
from src.datasets.patient_sample import PatientSample
from src.annotations.annotation_bundle import VectorROI


class PatientSampleValidationError(ValueError):
    """Raised when a PatientSample violates required invariants."""
    pass


def validate_patient_sample(
    sample: PatientSample,
    *,
    require_annotations: bool = False,
    report: ValidationReport | None = None,
) -> ValidationReport:
    """
    Validate structural and semantic correctness of a PatientSample.

    Raises
    ------
    PatientSampleValidationError
        If any invariant is violated.
    """
    report = report or ValidationReport(
        subject=f"PatientSample:{sample.patient_id}"
    )

    print(f"[Validator] Validating PatientSample {sample.patient_id}...")

    _validate_volume(sample, report)
    _validate_spacing(sample, report)
    _validate_patient_id(sample, report)
    _validate_annotations(sample, report, require_annotations=require_annotations)
    
    print("[Validator] Validation complete")

    return report

def _validate_volume(sample: PatientSample, report: ValidationReport) -> None:
    vol = sample.image_volume

    if not isinstance(vol, np.ndarray):
        report.error("image_volume is not a numpy array")
        return

    if vol.ndim != 3:
        report.error(
            "image_volume is not 3D",
            context=f"shape={vol.shape}",
        )
    else:
        report.info(
            "volume shape OK",
            context=str(vol.shape),
        )

    if vol.shape[1] <= 0 or vol.shape[2] <= 0:
        report.error(
            "Invalid spatial dimensions",
            context=f"shape={vol.shape}",
        )


def _validate_spacing(sample: PatientSample, report: ValidationReport) -> None:
    spacing = sample.spacing

    if spacing is None:
        report.error("spacing must not be None")
        return

    if len(spacing) != 3:
        report.error(
            "spacing must be (z, y, x)",
            context=f"got {spacing}",
        )
        return

    if any(s <= 0 for s in spacing):
        report.error(
            "spacing values must be > 0",
            context=str(spacing),
        )
    else:
        report.info(
            "spacing OK",
            context=str(spacing),
        )

def _validate_patient_id(sample: PatientSample, report: ValidationReport) -> None:
    if not sample.patient_id:
        report.error("patient_id must be set")
    else:
        report.info(
            "patient_id OK",
            context=sample.patient_id,
        )
        
def _validate_annotations(
    sample: PatientSample,
    report: ValidationReport,
    *,
    require_annotations: bool,
) -> None:
    ann = sample.annotations
    vol = sample.image_volume

    if ann is None or ann.vector_rois is None:
        if require_annotations:
            report.error("Annotations required but none found")
        else:
            report.warn("No annotations present")
        return

    report.info(
        "annotations present",
        context=f"slices={sorted(ann.vector_rois.keys())}",
    )

    for slice_idx, rois in ann.vector_rois.items():
        if not isinstance(slice_idx, int):
            report.error(
                "Slice index must be int",
                context=f"type={type(slice_idx)}",
            )
            continue

        if slice_idx < 0 or slice_idx >= vol.shape[0]:
            report.error(
                "ROI slice out of bounds",
                context=f"slice={slice_idx}, depth={vol.shape[0]}",
            )
            continue

        for roi in rois:
            _validate_vector_roi(
                roi=roi,
                volume=vol,
                report=report,
                slice_idx=slice_idx,
            )

def _validate_vector_roi(
    roi: VectorROI,
    volume: np.ndarray,
    report: ValidationReport,
    slice_idx: int,
) -> None:
    contour = roi.contour_px

    if contour.ndim != 2 or contour.shape[1] != 2:
        report.error(
            "ROI contour must be (N, 2)",
            context=f"slice={slice_idx}, shape={contour.shape}",
        )
        return

    h, w = volume.shape[1:]

    if (contour[:, 0] < 0).any() or (contour[:, 0] >= w).any():
        report.error(
            "ROI x-coordinates out of bounds",
            context=f"slice={slice_idx}, width={w}",
        )

    if (contour[:, 1] < 0).any() or (contour[:, 1] >= h).any():
        report.error(
            "ROI y-coordinates out of bounds",
            context=f"slice={slice_idx}, height={h}",
        )