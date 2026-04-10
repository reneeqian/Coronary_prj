from pathlib import Path
import random
import torch.nn as nn

from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import MedicalImageDataSource
from medical_image_ai_toolkit.training.training_config import TrainingConfig
from medical_image_ai_toolkit.pipeline.training_pipeline import TrainingPipeline
from medical_image_ai_toolkit.pipeline.validation_pipeline import ValidationPipeline
from regulatory_tools.requirements.yaml_requirement_provider import YamlRequirementProvider

from Coronary_prj.ingestors.coca_gated_ingestor import COCAGatedIngestor
from Coronary_prj.task_definitions.coronary_calcium_task import CoronaryCalciumTask

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "coca" / "cocacoronarycalciumandchestcts-2" / "Gated_release_final"
REQUIREMENT_PATH = PROJECT_ROOT / "docs" / "requirements.yaml"


class DeterministicHoldoutSplitStrategy:

    def __init__(self, train=0.7, val=0.15, seed=42, max_train=None, max_val=None, max_test=None):

        self.train = train
        self.val = val
        self.seed = seed
        self.max_train = max_train
        self.max_val = max_val
        self.max_test = max_test


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

        if self.max_train is not None:
            train_ids = train_ids[:self.max_train]
        if self.max_val is not None:
            val_ids = val_ids[:self.max_val]
        if self.max_test is not None:
            test_ids = test_ids[:self.max_test]

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
    

def main():

    print("Creating ingestor...")
    ingestor = COCAGatedIngestor(DATASET_PATH)

    print("Creating datasource...")
    datasource = MedicalImageDataSource(
        dataset_root=DATASET_PATH,
        ingestor=ingestor,
    )
    
    provider = YamlRequirementProvider(REQUIREMENT_PATH)

    config = TrainingConfig(
        epochs=5,     
        batch_size=2,
        task=CoronaryCalciumTask(),
        split_strategy = DeterministicHoldoutSplitStrategy(
            train=0.7, val=0.15, seed=42,
            max_train=100, max_val=100, max_test=100
        )
    )

    model = SmallSegmentationCNN()
    
    train_pipeline = TrainingPipeline(datasource, model, config, req_provider=provider)
    outputs = train_pipeline.run()
    
    val_pipeline = ValidationPipeline(datasource, model, config)
    results = val_pipeline.run()

    
if __name__ == "__main__":
    main()