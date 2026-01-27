# Data Requirements (CAC)
# CAC-DR-01: image_volume must be a 3D numpy array
# CAC-DR-02: image_volume spatial dimensions must be > 0
# CAC-DR-03: spacing must be present and valid (z, y, x) > 0
# CAC-DR-04: patient_id must be non-empty
# CAC-DR-05: annotations must exist if required
# CAC-DR-06: ROI slice indices must be valid
# CAC-DR-07: ROI contours must be in-bounds


from typing import Optional
import numpy as np

from src.medimg_training.validators.validation_report import ValidationReport
from src.datasets.patient_sample import PatientSample
from src.annotations.annotation_bundle import VectorROI

def validate_patient_sample(
    sample: PatientSample,
    *,
    require_annotations: bool = False,
    report: ValidationReport | None = None,
) -> ValidationReport:
    """
    Validate structural and semantic correctness of a PatientSample.
    
        Implements data requirements CAC-DR-01 through CAC-DR-08.
    """
    if report is None:
        report = ValidationReport(subject=f"PatientSample:{sample.patient_id}")


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
        report.error(
            message="image_volume is not a numpy array",
            requirement_id="CAC-DR-01",
        )
        return

    if vol.ndim != 3:
        report.error(
            message="image_volume is not 3D",
            requirement_id="CAC-DR-01",
            context=f"shape={vol.shape}",
        )
    else:
        report.info(
            message="volume shape OK",
            requirement_id="CAC-DR-01",
            context=str(vol.shape),
        )

    if vol.shape[1] <= 0 or vol.shape[2] <= 0:
        report.error(
            message="Invalid spatial dimensions",
            requirement_id="CAC-DR-02",
            context=f"shape={vol.shape}",
        )


def _validate_spacing(sample: PatientSample, report: ValidationReport) -> None:
    spacing = sample.spacing

    if spacing is None:
        report.error(
            message="spacing must not be None",
            requirement_id="CAC-DR-03",)
        return

    if len(spacing) != 3:
        report.error(
            message="spacing must be (z, y, x)",
            requirement_id="CAC-DR-03",
            context=f"got {spacing}",
        )
        return

    if any(s <= 0 for s in spacing):
        report.error(
            message="spacing values must be > 0",
            requirement_id="CAC-DR-03",
            context=str(spacing),
        )
    else:
        report.info(
            message="spacing OK",
            requirement_id="CAC-DR-03",
            context=str(spacing),
        )

def _validate_patient_id(sample: PatientSample, report: ValidationReport) -> None:
    if not sample.patient_id:
        report.error(
            message="patient_id must be set",
            requirement_id="CAC-DR-04",
        )
    else:
        report.info(
            message="patient_id OK",
            requirement_id="CAC-DR-04",
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
            report.error(
                message="Annotations required but none found",
                requirement_id="CAC-DR-05",)
        else:
            report.warn(
                message="No annotations present",
                requirement_id="CAC-DR-05",
            )
        return

    report.info(
        message="annotations present",
        requirement_id="CAC-DR-05",
        context=f"slices={sorted(ann.vector_rois.keys())}",
    )

    for slice_idx, rois in ann.vector_rois.items():
        if not isinstance(slice_idx, int):
            report.error(
                message="Slice index must be int",
                requirement_id="CAC-DR-06",
                context=f"type={type(slice_idx)}",
            )
            continue

        if slice_idx < 0 or slice_idx >= vol.shape[0]:
            report.error(
                message="ROI slice out of bounds",
                requirement_id="CAC-DR-06",
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
            message="ROI contour must be (N, 2)",
            requirement_id="CAC-DR-07",
            context=f"slice={slice_idx}, shape={contour.shape}",
        )
        return

    h, w = volume.shape[1:]

    if (contour[:, 0] < 0).any() or (contour[:, 0] >= w).any():
        report.error(
            message="ROI x-coordinates out of bounds",
            requirement_id="CAC-DR-07",
            context=f"slice={slice_idx}, width={w}",
        )

    if (contour[:, 1] < 0).any() or (contour[:, 1] >= h).any():
        report.error(
            message="ROI y-coordinates out of bounds",
            requirement_id="CAC-DR-07",
            context=f"slice={slice_idx}, height={h}",
        )