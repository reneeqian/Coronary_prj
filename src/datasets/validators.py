"""
validators.py

Data contract enforcement utilities for coronary CT datasets.

This module defines validation, correction, and assertion functions
that enforce assumptions about:
- dimensionality
- spatial alignment
- intensity values
- label semantics

All dataset samples MUST pass through these checks before reaching a model.
"""

from __future__ import annotations
from cProfile import label

from matplotlib import image
import numpy as np
import nibabel as nib
import SimpleITK as sitk


# =========================
# Validation (fail fast)
# =========================

def validate_ndim(image: np.ndarray, label: np.ndarray) -> None:
    """
    VR-01: The system shall reject image and label volumes that are not 3D. 
    Derived from DR-01.
    """
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image, got {image.ndim}D")
    if label.ndim != 3:
        raise ValueError(f"Expected 3D label, got {label.ndim}D")


def validate_shape_match(image: np.ndarray, label: np.ndarray) -> None:
    """
    VR-02: The system shall reject samples where image and label shapes differ.
    Derived from DR-02.
    """
    if image.shape != label.shape:
        raise ValueError(
            f"Image/label shape mismatch: image={image.shape}, label={label.shape}"
        )


def validate_finite(image: np.ndarray) -> None:
    """
    VR-03: The system shall reject images containing NaN or infinite values.
    Derived from DR-03.
    """
    if not np.isfinite(image).all():
        raise ValueError("Image contains NaN or Inf values")


def validate_label_values(label: np.ndarray, allowed_values=(0, 1)) -> None:
    """
    VR-04: The system shall reject labels containing values outside the allowed class set.
    Derived from DR-04.
    """
    unique_vals = np.unique(label)
    invalid = set(unique_vals) - set(allowed_values)
    if invalid:
        raise ValueError(f"Label contains invalid values: {invalid}")


# =========================
# Correction (safe fixes)
# =========================

def enforce_ras_orientation(nifti_img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Convert NIfTI image to closest canonical (RAS) orientation.
    """
    return nib.as_closest_canonical(nifti_img)


def resample_to_spacing(
    image: np.ndarray,
    label: np.ndarray,
    original_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample image and label volumes to target spacing.

    - Image: linear interpolation
    - Label: nearest-neighbor interpolation
    """

    def _resample(volume, spacing, is_label=False):
        sitk_img = sitk.GetImageFromArray(volume)
        sitk_img.SetSpacing(spacing[::-1])  # SITK uses (x, y, z)

        original_size = np.array(sitk_img.GetSize())
        original_spacing = np.array(sitk_img.GetSpacing())
        target_spacing_arr = np.array(target_spacing[::-1])

        new_size = np.round(
            original_size * (original_spacing / target_spacing_arr)
        ).astype(int)

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize([int(s) for s in new_size])
        resampler.SetOutputSpacing(target_spacing_arr.tolist())
        resampler.SetOutputDirection(sitk_img.GetDirection())
        resampler.SetOutputOrigin(sitk_img.GetOrigin())

        if is_label:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(sitk.sitkLinear)

        resampled = resampler.Execute(sitk_img)
        return sitk.GetArrayFromImage(resampled)

    image_resampled = _resample(image, original_spacing, is_label=False)
    label_resampled = _resample(label, original_spacing, is_label=True)

    return image_resampled, label_resampled


# =========================
# Post-condition assertions
# =========================

def assert_postconditions(image: np.ndarray, label: np.ndarray) -> None:
    """
    VR-05: The system shall guarantee dataset postconditions prior to model input.
    """
    assert image.shape == label.shape, "Postcondition failed: shape mismatch"
    assert image.ndim == 3, "Postcondition failed: image not 3D"
    assert image.dtype in (np.float32, np.float64), "Unexpected image dtype"
    assert label.dtype in (np.int16, np.int32, np.int64, np.uint8), "Unexpected label dtype"


# =========================
# Unified enforcement entrypoint
# =========================

def enforce_data_contract(
    image: np.ndarray,
    label: np.ndarray,
) -> None:
    """
    Applies all Validation Requirements (VR-01 through VR-05).

    This function represents the enforcement boundary of the dataset data contract.
    """
    validate_ndim(image, label)
    validate_shape_match(image, label)
    validate_finite(image)
    validate_label_values(label)
    assert_postconditions(image, label)
