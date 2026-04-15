"""
PatchGAN Discriminator for pix2pix sCT synthesis.

Architecture: 70×70 receptive field (n_layers=3), InstanceNorm, LeakyReLU(0.2).
Anatomy conditioning: learned embedding added to feature map after block 0.

Input : cat(mr_1ch, ct_or_pred) → (B, in_channels, H, W)
Output: (B, 1, pH, pW)  — patch logits, no sigmoid
         Use with GANLoss (BCEWithLogitsLoss) for numerical stability.
"""

import torch
import torch.nn as nn


class NLayerDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator.

    Receptive field with n_layers=3, kernel=4, stride=2:
        RF = 1 + 3 * (4-1) * 2^{0..n-1} → 70px for n_layers=3

    Args:
        in_channels : channels of cat(mr, ct) input. Default 2.
        ndf         : base filter count. Default 64.
        n_layers    : number of strided Conv blocks. Default 3.
        use_anatomy : inject anatomy embedding after block 0. Default True.
        n_anatomy   : number of anatomy classes (HN/TH/AB). Default 3.
    """

    def __init__(
        self,
        in_channels: int = 2,
        ndf: int = 64,
        n_layers: int = 3,
        use_anatomy: bool = True,
        n_anatomy: int = 3,
    ):
        super().__init__()
        self.use_anatomy = use_anatomy

        # Block 0: no normalisation (standard pix2pix design)
        self.block0 = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Anatomy embedding injected into block0 output as a channel-wise bias
        if use_anatomy:
            self.anatomy_emb = nn.Embedding(n_anatomy, ndf)
            nn.init.normal_(self.anatomy_emb.weight, mean=0.0, std=0.02)

        # Blocks 1 … n_layers — each doubles channels up to 512
        blocks = []
        in_ch = ndf
        for i in range(1, n_layers + 1):
            out_ch = min(ndf * (2 ** i), 512)
            stride = 1 if i == n_layers else 2      # last block keeps spatial res
            blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride,
                          padding=1, bias=False),
                nn.InstanceNorm2d(out_ch, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            in_ch = out_ch

        self.blocks = nn.ModuleList(blocks)

        # Output: 1-channel patch map (logits, no sigmoid)
        self.out_conv = nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        anatomy_idx: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x           : (B, in_channels, H, W)  — cat(mr, ct/pred)
            anatomy_idx : (B,)  long tensor, values in {0, 1, 2}

        Returns:
            (B, 1, pH, pW) patch logits
        """
        x = self.block0(x)

        if self.use_anatomy and anatomy_idx is not None:
            emb = self.anatomy_emb(anatomy_idx)     # (B, ndf)
            x   = x + emb[:, :, None, None]         # broadcast over spatial dims

        for block in self.blocks:
            x = block(x)

        return self.out_conv(x)
