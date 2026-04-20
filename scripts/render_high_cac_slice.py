"""
Finds the patient with the highest total CAC burden in the COCA dataset,
computes per-ROI Agatston scores, and renders the highest-scoring slice
to a PNG suitable for presentations.
"""

from pathlib import Path
import plistlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.draw import polygon as sk_polygon

from Coronary_prj.ingestors.coca_gated_ingestor import COCAGatedIngestor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "coca" / "cocacoronarycalciumandchestcts-2" / "Gated_release_final"
XML_DIR      = DATASET_PATH / "calcium_xml"
OUTPUT       = PROJECT_ROOT / "artifacts" / "cac_highest_burden.png"

ORANGE   = "#FF7A00"
CT_WL    = 40    # window level (HU)
CT_WW    = 400   # window width (HU)


def density_factor(peak_hu: float) -> int:
    if peak_hu >= 400: return 4
    if peak_hu >= 300: return 3
    if peak_hu >= 200: return 2
    if peak_hu >= 130: return 1
    return 0


def find_highest_burden_patient() -> str:
    scores = {}
    for xml_file in XML_DIR.glob("*.xml"):
        try:
            with open(xml_file, "rb") as f:
                data = plistlib.load(f)
            total_area = sum(
                roi.get("Area", 0.0)
                for img in data.get("Images", [])
                for roi in img.get("ROIs", [])
            )
            scores[xml_file.stem] = total_area
        except Exception:
            pass
    return max(scores, key=scores.__getitem__)


def compute_roi_scores(volume, rois_by_slice, pixel_area_mm2):
    total_agatston = 0.0
    roi_scores = {}

    for slice_idx, rois in rois_by_slice.items():
        Z = volume.shape[0]
        if slice_idx < 0 or slice_idx >= Z:
            continue
        hu_slice = volume[slice_idx]
        slice_scores = []
        for roi in rois:
            pts = roi.contour_px
            if pts is None or len(pts) < 3:
                continue
            H, W = hu_slice.shape
            rr, cc = sk_polygon(pts[:, 1], pts[:, 0], shape=(H, W))
            if len(rr) == 0:
                continue
            hu_vals = hu_slice[rr, cc]
            n_calcium = (hu_vals >= 130).sum()
            if n_calcium == 0:
                continue
            peak_hu  = hu_vals.max()
            score    = n_calcium * pixel_area_mm2 * density_factor(peak_hu)
            total_agatston += score
            bbox = (rr.min(), cc.min(), rr.max(), cc.max())
            slice_scores.append((roi, score, rr, cc, bbox))
        if slice_scores:
            roi_scores[slice_idx] = slice_scores

    return total_agatston, roi_scores


def render_clean(patient_id, volume, best_slice, output_path):
    H, W = volume.shape[1], volume.shape[2]

    fig, ax = plt.subplots(figsize=(14, 14), facecolor="black")
    ax.imshow(
        volume[best_slice].astype(float), cmap="gray",
        vmin=CT_WL - CT_WW / 2, vmax=CT_WL + CT_WW / 2,
        interpolation="bilinear", extent=[0, W, H, 0],
    )
    ax.set_title(
        f"Patient {patient_id}  |  Slice {best_slice}",
        color="white", fontsize=24, fontweight="bold", pad=14,
    )
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches=None, facecolor="black", pad_inches=0.15)
    plt.close()
    print(f"Saved → {output_path}")


def render(patient_id, volume, spacing, total_agatston, roi_scores, output_path):
    H, W = volume.shape[1], volume.shape[2]

    best_slice     = max(roi_scores, key=lambda s: sum(x[1] for x in roi_scores[s]))
    slice_agatston = sum(x[1] for x in roi_scores[best_slice])

    fig, ax = plt.subplots(figsize=(14, 14), facecolor="black")

    hu_slice = volume[best_slice].astype(float)
    ax.imshow(
        hu_slice, cmap="gray",
        vmin=CT_WL - CT_WW / 2, vmax=CT_WL + CT_WW / 2,
        interpolation="bilinear", extent=[0, W, H, 0],
    )

    for roi, score, rr, cc, bbox in roi_scores[best_slice]:
        # Pixel mask fill
        overlay = np.zeros((H, W, 4), dtype=float)
        rgba = matplotlib.colors.to_rgba(ORANGE)
        overlay[rr, cc] = (*rgba[:3], 0.50)
        ax.imshow(overlay, interpolation="none", extent=[0, W, H, 0])

        # Contour outline
        pts    = roi.contour_px
        closed = np.vstack([pts, pts[0]])
        ax.plot(closed[:, 0], closed[:, 1], color=ORANGE, linewidth=2.0)

        # Bounding box
        r0, c0, r1, c1 = bbox
        rect = mpatches.FancyBboxPatch(
            (c0, r0), c1 - c0, r1 - r0,
            boxstyle="square,pad=2",
            linewidth=2.0, edgecolor=ORANGE, facecolor="none",
        )
        ax.add_patch(rect)

        # Per-ROI Agatston label
        ax.text(
            c0 + 3, r0 - 6,
            f"Agatston: {score:.1f}",
            color=ORANGE, fontsize=16, fontweight="bold",
            verticalalignment="bottom",
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2),
        )

    # Summary box
    ax.text(
        0.015, 0.985,
        f"Total Agatston:  {total_agatston:.0f}\nSlice Agatston:  {slice_agatston:.1f}",
        transform=ax.transAxes,
        color="white", fontsize=20, fontweight="bold",
        verticalalignment="top",
        bbox=dict(facecolor="black", alpha=0.65, edgecolor="#555", boxstyle="round,pad=0.5"),
    )

    ax.set_title(
        f"Patient {patient_id}  |  Slice {best_slice}",
        color="white", fontsize=24, fontweight="bold", pad=14,
    )
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches=None, facecolor="black", pad_inches=0.15)
    plt.close()
    print(f"Saved → {output_path}")


def main():
    print("Scanning annotation files for highest CAC burden...")
    patient_id = find_highest_burden_patient()
    print(f"Highest burden patient: {patient_id}")

    print("Loading patient data...")
    ingestor = COCAGatedIngestor(DATASET_PATH)
    sample   = ingestor.load_patient_sample(patient_id)

    dy, dx = sample.spacing[1], sample.spacing[2]
    total_agatston, roi_scores = compute_roi_scores(
        sample.image_volume,
        sample.annotations.vector_rois,
        pixel_area_mm2=dy * dx,
    )
    print(f"Total Agatston: {total_agatston:.0f}")

    best_slice = max(roi_scores, key=lambda s: sum(x[1] for x in roi_scores[s]))

    render_clean(patient_id, sample.image_volume, best_slice,
                 OUTPUT.with_name(OUTPUT.stem + "_clean" + OUTPUT.suffix))
    render(patient_id, sample.image_volume, sample.spacing,
           total_agatston, roi_scores, OUTPUT)


if __name__ == "__main__":
    main()
