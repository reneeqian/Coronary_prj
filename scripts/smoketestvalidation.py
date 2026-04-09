from pathlib import Path
import random
import torch
import torch.nn as nn
import numpy as np

from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import MedicalImageDataSource
from medical_image_ai_toolkit.training.training_config import TrainingConfig
from medical_image_ai_toolkit.training.task_definition import TrainingTaskDefinition
from medical_image_ai_toolkit.pipeline.validation_pipeline import ValidationPipeline
from regulatory_tools.requirements.yaml_requirement_provider import YamlRequirementProvider

from Coronary_prj.ingestors.coca_gated_ingestor import COCAGatedIngestor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "coca" / "cocacoronarycalciumandchestcts-2" / "Gated_release_final"
REQUIREMENT_PATH = PROJECT_ROOT / "docs" / "requirements.yaml"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "training_runs"   # adjust to point at a specific run if needed


class DeterministicHoldoutSplitStrategy:

    def __init__(self, train=0.7, val=0.15, seed=42):

        self.train = train
        self.val = val
        self.seed = seed

    def split(self, patient_ids):

        rng = random.Random(self.seed)

        ids = list(patient_ids)

        rng.shuffle(ids)

        n = len(ids)

        train_end = int(self.train * n)
        val_end = train_end + int(self.val * n)

        train_ids = ids[:train_end]
        val_ids = ids[train_end:val_end]
        test_ids = ids[val_end:]

        return train_ids, val_ids, test_ids


class SmallSegmentationCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 1, 1)  # output mask
        )

    def forward(self, x):
        return self.net(x)  # (B,1,H,W)


class CoronaryCalciumTask(TrainingTaskDefinition):

    def generate_training_samples(self, patient_sample):

        volume = patient_sample.image_volume
        annotations = patient_sample.annotations

        Z, H, W = volume.shape

        vector_rois = None
        if annotations is not None:
            vector_rois = annotations.vector_rois

        for slice_idx in range(Z):

            img = torch.tensor(
                volume[slice_idx],
                dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)

            mask = np.zeros((H, W), dtype=np.float32)

            if vector_rois and slice_idx in vector_rois:

                for roi in vector_rois[slice_idx]:

                    contour = roi.contour_px
                    if contour is None or len(contour) < 3:
                        continue

                    rr, cc = self._polygon_to_mask(contour, H, W)
                    mask[rr, cc] = 1.0

            mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

            yield {
                "input": img,
                "target": mask_tensor
            }

    def compute_loss(self, prediction, target):
        return torch.nn.functional.binary_cross_entropy_with_logits(
            prediction, target
        )

    def _polygon_to_mask(self, contour, H, W):
        from skimage.draw import polygon

        x = contour[:, 0]
        y = contour[:, 1]

        rr, cc = polygon(y, x, shape=(H, W))
        return rr, cc


def _find_latest_model(training_runs_root: Path) -> Path | None:
    """
    Scans training_runs_root for the most recently created model.pt
    and returns its path, or None if none exist.
    """
    candidates = sorted(training_runs_root.glob("*/model.pt"))
    return candidates[-1] if candidates else None


def main():

    print("Creating ingestor...")
    ingestor = COCAGatedIngestor(DATASET_PATH)

    print("Creating datasource...")
    datasource = MedicalImageDataSource(
        dataset_root=DATASET_PATH,
        ingestor=ingestor,
    )

    print("Partitioning datasource...")
    split_strategy = DeterministicHoldoutSplitStrategy(train=0.7, val=0.15, seed=42)
    datasource.create_partitions(split_strategy)
    datasource.partition_summary()

    print("Building model...")
    model = SmallSegmentationCNN()

    # Attempt to load weights from the most recent training run so that
    # the smoke test exercises a real trained model.  If no saved model
    # is found, the test proceeds with randomly initialised weights and
    # prints a clear warning — this is intentional; it lets the pipeline
    # plumbing be exercised even before a full training run exists.
    model_pt = _find_latest_model(MODEL_PATH)
    if model_pt is not None:
        print(f"Loading model weights from: {model_pt}")
        model.load_state_dict(torch.load(model_pt, map_location="cpu"))
    else:
        print(
            "WARNING: No trained model found under artifacts/training_runs. "
            "Proceeding with randomly initialised weights."
        )

    config = TrainingConfig(
        epochs=5,
        batch_size=2,
        task=CoronaryCalciumTask(),
        split_strategy=split_strategy,
    )

    pipeline = ValidationPipeline(datasource, model, config)
    results = pipeline.run()


if __name__ == "__main__":
    main()