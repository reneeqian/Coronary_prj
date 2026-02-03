#%%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # adjust if needed
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestors.coca_gated_ingestor import COCAGatedIngestor
from src.annotations.annotation_bundle import AnnotationBundle
from src.datasets.patient_sample import PatientSample
from src.contracts.patient_sample_contract import check_patient_sample_contract

# -------------------------
# Configuration
# -------------------------

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "coca"
    / "cocacoronarycalciumandchestcts-2"
    / "Gated_release_final"
)
PATIENT_ID = "0"                               # <-- CHANGE IF NEEDED
MAX_PATIENTS_TO_SHOW = 1


# -------------------------
# Visualization helpers
# -------------------------

def visualize_slice_with_rois(
    volume: np.ndarray,
    annotation_bundle: AnnotationBundle,
    slice_index: int,
    title: str = "",
):
    """
    Visualize a single CT slice with vector ROIs overlaid.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(volume[slice_index], cmap="gray")
    ax.set_title(title)
    ax.axis("off")

    if annotation_bundle.vector_rois is None:
        plt.show()
        return

    rois = annotation_bundle.vector_rois.get(slice_index, [])
    for roi in rois:
        contour = roi.contour_px
        ax.plot(contour[:, 0], contour[:, 1], linewidth=2)

    plt.show()


def visualize_patient_sample(sample: PatientSample):
    """
    Pick a representative slice and visualize.
    """
    volume = sample.image_volume
    annotations = sample.annotations

    # Choose middle slice by default
    z = volume.shape[0] // 2

    # Prefer a slice that actually has annotations
    if annotations.vector_rois:
        z = sorted(annotations.vector_rois.keys())[0]

    print("Annotated slices:", sorted(annotations.vector_rois.keys())[:10])
    print("Volume depth:", volume.shape[0])

    visualize_slice_with_rois(
        volume=volume,
        annotation_bundle=annotations,
        slice_index=z,
        title=f"Patient {sample.patient_id} | Slice {z}",
    )


# -------------------------
# Main debug routine
# -------------------------
# %%
print(DATASET_ROOT)
ingestor = COCAGatedIngestor(dataset_root=DATASET_ROOT)

print("=== Ingesting single patient ===")
patient_dir = DATASET_ROOT / "patient" / PATIENT_ID
sample = ingestor.ingest_patient(PATIENT_ID)

assert isinstance(sample, PatientSample)
assert sample.image_volume.ndim == 3

print(
    f"Patient {sample.patient_id}: "
    f"volume shape={sample.image_volume.shape}, "
    f"spacing={sample.spacing}"
)


visualize_patient_sample(sample)

# %%
print("\n=== Ingesting dataset (limited) ===")
for i, sample in enumerate(ingestor.ingest_dataset()):
    print(f"Ingested patient {sample.patient_id}")
    
    visualize_patient_sample(sample)

    if i + 1 >= MAX_PATIENTS_TO_SHOW:
        break


# %%
sample = ingestor.ingest_patient("0")
check_patient_sample_contract(sample)
# %%
