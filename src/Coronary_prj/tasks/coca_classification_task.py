import torch
import torch.nn.functional as F

from medical_image_ai_toolkit.training.task_definition import TrainingTaskDefinition


class CocaClassificationTask(TrainingTaskDefinition):

    def prepare_training_sample(self, sample):

        # example: take middle slice
        volume = sample.image_volume
        z = volume.shape[0] // 2

        image = volume[z]

        label = sample.annotations.global_labels["coca"]

        x = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)

        return x, y

    def compute_loss(self, pred, y):

        return F.cross_entropy(pred, y.unsqueeze(0))