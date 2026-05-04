import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CalciumScoreRegressor(nn.Module):
    """
    Lightweight CNN regression model for per-vessel Agatston score prediction.

    Takes a single normalised CT slice and outputs log1p-scaled Agatston
    score estimates for four coronary vessels (LCA, LAD, LCX, RCA).

    Three conv-pool blocks progressively halve spatial resolution;
    AdaptiveAvgPool2d removes spatial dimensions entirely so the model
    accepts any CT slice size (typically 512×512).

    At inference, apply ``torch.expm1(model(x))`` to recover Agatston
    scores in the original scale.

    Parameters
    ----------
    base_channels : int
        Feature maps in the first block; doubles each block.
        Default 16 gives ~47 K parameters — fast on CPU for smoke tests.
    num_outputs : int
        Number of regression outputs (default 4: LCA, LAD, LCX, RCA).

    Input / output
    --------------
    x : (B, 1, H, W)  — single-channel normalised CT slice
    returns : (B, num_outputs) — log1p(Agatston) per vessel
    """

    def __init__(self, base_channels: int = 16, num_outputs: int = 4) -> None:
        super().__init__()
        c = base_channels
        self.block1 = _conv_block(1, c)
        self.block2 = _conv_block(c, c * 2)
        self.block3 = _conv_block(c * 2, c * 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c * 4, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)       # (B, c*4, 1, 1)
        x = x.flatten(1)       # (B, c*4)
        return self.fc(x)      # (B, num_outputs)
