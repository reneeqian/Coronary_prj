import torch.nn as nn


class SmallSegmentationCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x):
        return self.net(x)  # (B, 1, H, W)
