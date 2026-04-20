import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet2D(nn.Module):
    """
    2-D U-Net for single-channel medical image segmentation.

    Designed for sparse foreground (e.g. coronary calcium) where the
    lesion occupies a tiny fraction of each slice:

    - Skip connections from encoder to decoder preserve the fine spatial
      detail that a flat conv stack would discard after pooling.
    - BatchNorm after every conv stabilises training on the wide
      intensity range of CT Hounsfield units.
    - Bilinear upsampling + conv in the decoder avoids checkerboard
      artefacts from transposed convolutions.

    Parameters
    ----------
    base_channels : int
        Number of feature maps in the first encoder block.  Doubles at
        each downsampling level.  Default 32 gives a parameter count
        around 1.9 M, which trains comfortably on CPU for smoke tests.
    depth : int
        Number of encoder levels (each level halves spatial resolution).
        Default 4 produces a receptive field large enough to capture
        context around small calcified foci.

    Input / output
    --------------
    x : (B, 1, H, W)  — single-channel 2-D slice (any size ≥ 2^depth)
    returns : (B, 1, H, W) — raw logits; apply sigmoid for probabilities
    """

    def __init__(self, base_channels: int = 32, depth: int = 4):
        super().__init__()

        self.depth = depth
        ch = [base_channels * (2 ** i) for i in range(depth + 1)]

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = 1
        for out_ch in ch[:-1]:
            self.encoders.append(_conv_block(in_ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = _conv_block(ch[-2], ch[-1])

        # Decoder
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.upsamples.append(nn.Conv2d(ch[i + 1], ch[i], 1))
            self.decoders.append(_conv_block(ch[i] * 2, ch[i]))

        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = up(x)
            x = dec(torch.cat([skip, x], dim=1))

        return self.head(x)
