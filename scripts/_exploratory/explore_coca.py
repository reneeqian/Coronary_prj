# %%
import pydicom
import numpy as np
from pathlib import Path
from lxml import etree
import matplotlib.pyplot as plt
import random
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_dicom_series(series_dir: Path):
    dicoms = []
    for f in series_dir.glob("*.dcm"):
        ds = pydicom.dcmread(f)
        dicoms.append(ds)

    # Sort by InstanceNumber (critical)
    dicoms.sort(key=lambda x: int(x.InstanceNumber))

    volume = np.stack([ds.pixel_array for ds in dicoms], axis=0)

    return volume, dicoms

def parse_coca_xml(xml_path):
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    annotations = {}

    for image_dict in root.findall(".//dict"):
        keys = [k.text for k in image_dict.findall("key")]
        if "ImageIndex" not in keys:
            continue

        idx = keys.index("ImageIndex")
        image_index = int(image_dict.findall("integer")[0].text)

        roi_list = image_dict.findall(".//dict")
        rois = []

        for roi in roi_list:
            name = roi.findtext("string")
            points_px = roi.xpath(".//key[.='Point_px']/following-sibling::array[1]/string")

            coords = []
            for p in points_px:
                x, y = p.text.strip("()").split(",")
                coords.append((float(x), float(y)))

            if coords:
                rois.append({
                    "name": name,
                    "points_px": coords
                })

        if rois:
            annotations[image_index] = rois

    return annotations

def show_slice_with_rois(volume, annotations, slice_idx, ax=None):
    img = volume[slice_idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        standalone = True
    else:
        standalone = False
        
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Slice {slice_idx}")
    
    if slice_idx in annotations:
        for roi in annotations[slice_idx]:
            pts = np.array(roi["points_px"])
            ax.plot(pts[:, 0], pts[:, 1], "-r", linewidth=2)
            ax.scatter(pts[:, 0], pts[:, 1], s=5)
            
    ax.axis("off")

    if standalone:
        plt.show()
        
def visualize_volume_slice_with_rois(
    ax,
    volume,
    annotations,
    slice_idx,
    title=None
):
    img = volume[slice_idx]

    ax.imshow(img, cmap="gray")

    if slice_idx in annotations:
        for roi in annotations[slice_idx]:
            pts = np.array(roi["points_px"])
            ax.plot(pts[:, 0], pts[:, 1], "-r", linewidth=1)
            ax.scatter(pts[:, 0], pts[:, 1], s=5)

    ax.set_title(title if title else f"Slice {slice_idx}")
    ax.axis("off")

def visualize_random_slices(volume, annotations, n=4):
    import random
    slice_indices = random.sample(range(volume.shape[0]), n)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for ax, idx in zip(axes, slice_indices):
        has_cac = idx in annotations
        title = f"Slice {idx} | CAC: {'YES' if has_cac else 'NO'}"

        visualize_volume_slice_with_rois(
            ax=ax,
            volume=volume,
            annotations=annotations,
            slice_idx=idx,
            title=title
        )

    plt.tight_layout()
    plt.show()

def visualize_coca_patient_cac(patient_idx: int):
    """
    Visualize all slices containing CAC ROIs for a given COCA patient.
    """
    COCA_ROOT = PROJECT_ROOT / "data" / "raw" / "coca" / "cocacoronarycalciumandchestcts-2" / "Gated_release_final"
    if not COCA_ROOT.exists():
        raise RuntimeError(
            f"COCA root not found.\n"
            f"Expected: {COCA_ROOT}\n"
            f"CWD: {Path.cwd()}"
        )
    patient_dir = COCA_ROOT / "patient" / str(patient_idx)
    xml_path = COCA_ROOT / "calcium_xml" / f"{patient_idx}.xml"

    print("[INFO] Resolving paths:")
    print(f"  Patient directory: {patient_dir.resolve()}")
    print(f"  Annotation XML:    {xml_path.resolve()}")

    if not patient_dir.exists():
        raise FileNotFoundError(f"Patient directory does not exist: {patient_dir}")

    if not xml_path.exists():
        raise FileNotFoundError(f"Annotation XML does not exist: {xml_path}")

    # --- load annotations ---
    annotations = parse_coca_xml(xml_path)

    if len(annotations) == 0:
        print(f"[INFO] No CAC annotations found for patient {patient_idx}")
        return

    # --- pick first gated series (assumption, explicit by design) ---
    series_dirs = [d for d in patient_dir.iterdir() if d.is_dir()]

    if len(series_dirs) == 0:
        raise RuntimeError(f"No DICOM series found in {patient_dir}")

    series_dir = series_dirs[0]
    print(f"[INFO] Using DICOM series:\n  {series_dir.resolve()}")

    # --- load volume ---
    volume, _ = load_dicom_series(series_dir)

    # --- slices with CAC ---
    cac_slices = sorted(annotations.keys())
    num_slices = len(cac_slices)

    print(f"[INFO] Found {num_slices} slices with CAC")

    # --- subplot layout ---
    cols = 4
    rows = math.ceil(num_slices / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for ax, slice_idx in zip(axes, cac_slices):
        title = f"Patient {patient_idx} | Slice {slice_idx} | CAC: YES"

        visualize_volume_slice_with_rois(
            ax=ax,
            volume=volume,
            annotations=annotations,
            slice_idx=slice_idx,
            title=title
        )

    # --- turn off unused axes ---
    for ax in axes[len(cac_slices):]:
        ax.axis("off")

    fig.suptitle(
        f"COCA Patient {patient_idx} – Slices with Coronary Calcium",
        fontsize=16,
        y=0.98
    )

    plt.tight_layout()
    plt.show()

# %%
series_dir = PROJECT_ROOT / "data/raw/coca/cocacoronarycalciumandchestcts-2/Gated_release_final/patient/0"
volume, dicoms = load_dicom_series(series_dir)

# %%
volume.shape
# %%
xml_path = PROJECT_ROOT / "data/raw/coca/cocacoronarycalciumandchestcts-2/Gated_release_final/calcium_xml/0.xml"
annotations = parse_coca_xml(xml_path)
print(len(annotations))
print(list(annotations.keys())[:5])

# %%
visualize_coca_patient_cac(2)

# %%
