from pathlib import Path
import random
import torch
import torch.nn as nn
import numpy as np

from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import MedicalImageDataSource
from medical_image_ai_toolkit.training.training_config import TrainingConfig
from medical_image_ai_toolkit.training.medical_image_trainer import MedicalImageTrainer
from medical_image_ai_toolkit.results.medical_image_training_results import MedicalImageTrainingResults
from medical_image_ai_toolkit.training.task_definition import TrainingTaskDefinition

# import your existing ingestor
from Coronary_prj.ingestors.coca_gated_ingestor import COCAGatedIngestor


DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "coca" / "cocacoronarycalciumandchestcts-2" / "Gated_release_final"


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

def main():

    print("Creating ingestor...")
    ingestor = COCAGatedIngestor(DATASET_PATH)

    print("Creating datasource...")
    datasource = MedicalImageDataSource(
        dataset_root=DATASET_PATH,
        ingestor=ingestor,
    )

    n_patients = datasource.get_num_patients()
    print(f"Patients discovered: {n_patients}")

    patient_ids = datasource.patient_ids

    print("Generating partitions...")
    datasource.create_partitions(
        DeterministicHoldoutSplitStrategy()
    )
    
    datasource.partition_summary()

    # Test loading a sample
    print("Loading first sample...")
    
    pnum = 50
    train_ids = datasource.get_train_ids()
    sample = datasource.get_patient(train_ids[pnum])

    print("Loaded sample type:", type(sample))
    print("Patient ID:", sample.patient_id)
    print("Image shape:", sample.image_volume.shape)
    print("Spacing:", sample.spacing)
    print("Has annotations:", sample.annotations is not None)

    if sample.annotations:
        print("Annotation slices:", len(sample.annotations.vector_rois or {}))
        
    print("Loading slice...")

    slice_img = datasource.load_slice(train_ids[pnum], 10)

    print("Slice shape:", slice_img.shape)
    
    config = TrainingConfig(
        epochs=5,
        batch_size=2,
        task=CoronaryCalciumTask()
    )

    model = SmallSegmentationCNN()
    
    trainer = MedicalImageTrainer(
        datasource,
        model,
        training_config = config
    )
    
    trainer.sanity_check()
    
    img = torch.randn(1, 1, 512, 512)
    out = model(img)
    print("Output shape:", out.shape)
    
    results = trainer.train()

    
if __name__ == "__main__":
    main()