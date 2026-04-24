from pathlib import Path

from tests import config

from Coronary_prj.models.unet2d import UNet2D
from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import MedicalImageDataSource
from medical_image_ai_toolkit.dataobjects.datasources.deterministic_split import DeterministicHoldoutSplit
from medical_image_ai_toolkit.training.training_config import TrainingConfig
from medical_image_ai_toolkit.pipeline.training_pipeline import TrainingPipeline
from medical_image_ai_toolkit.pipeline.model_testing_pipeline import ModelTestingPipeline
from regulatory_tools.requirements.yaml_requirement_provider import YamlRequirementProvider

from Coronary_prj.ingestors.coca_gated_ingestor import COCAGatedIngestor
from Coronary_prj.task_definitions.coronary_calcium_task import CoronaryCalciumTask
from Coronary_prj.models.small_segmentation_cnn import SmallSegmentationCNN
from Coronary_prj.models.unet2d import UNet2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "coca" / "cocacoronarycalciumandchestcts-2" / "Gated_release_final"
REQUIREMENT_PATH = PROJECT_ROOT / "docs" / "requirements.yaml"



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
        split_strategy = DeterministicHoldoutSplit(
            train=0.7, val=0.15, seed=42,
            max_train=100, max_val=100, max_test=100
        )
    )

    #model = SmallSegmentationCNN()
    
    model = UNet2D()
    
    train_pipeline = TrainingPipeline(
        datasource, model, config,
        req_provider=provider,
        output_dir=PROJECT_ROOT / "artifacts" / "training_runs",
    )
    outputs = train_pipeline.run()
    print("Training outputs:", outputs)

    
if __name__ == "__main__":
    main()