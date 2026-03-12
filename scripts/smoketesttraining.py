from pathlib import Path
import random
import torch
import torch.nn as nn

from Coronary_prj.tasks.coca_classification_task import CocaClassificationTask
from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import MedicalImageDataSource
from medical_image_ai_toolkit.training.training_config import TrainingConfig
from medical_image_ai_toolkit.training.medical_image_trainer import MedicalImageTrainer
from medical_image_ai_toolkit.results.medical_image_training_results import MedicalImageTrainingResults

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

class SmallSliceCNN(nn.Module):
    """
    Minimal CNN for testing trainer wiring.
    Accepts single-channel medical image slices.
    """

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),

            nn.Linear(16, 1)
        )

    def forward(self, x):

        return self.net(x)

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
    train_ids, val_ids, test_ids = datasource.create_partitions(
        DeterministicHoldoutSplitStrategy()
    )
    
    datasource.partition_summary()

    # Test loading a sample
    print("Loading first sample...")
    
    pnum = 50

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
        batch_size=2
    )

    model = SmallSliceCNN()
    
    trainer = MedicalImageTrainer(
        datasource,
        model,
        task=CocaClassificationTask(),
        training_config = config
    )
    
    trainer.sanity_check()
    
    results = trainer.train()

    
if __name__ == "__main__":
    main()