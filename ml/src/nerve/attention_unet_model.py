"""
attention_unet_model.py
Attention U-Net for mandibular canal segmentation.
Adds attention gates to standard U-Net – focuses on canal region
and suppresses irrelevant background activations.

Reference: Oktay et al. "Attention U-Net: Learning Where to Look
           for the Pancreas" (2018)

Input : (B, 1, H, W) grayscale panoramic X-ray (512×256)
Output: (B, 1, H, W) sigmoid probability map
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Conv → BN → ReLU → Dropout → Conv → BN → ReLU"""
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool → DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.pool_conv(x)


class AttentionGate(nn.Module):
    """
    Attention Gate – learns to highlight the canal region
    and suppress irrelevant features in skip connections.

    g  : gating signal from decoder (coarser, semantic)
    x  : skip connection from encoder (finer, spatial)
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # Upsample g to match x spatial size if needed
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear',
                              align_corners=False)
        g1  = self.W_g(g)
        x1  = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi          # attended skip connection


class Up(nn.Module):
    """Upsample → AttentionGate → Concat → DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up      = nn.ConvTranspose2d(in_ch, in_ch // 2,
                                          kernel_size=2, stride=2)
        self.att     = AttentionGate(F_g=in_ch // 2,
                                     F_l=in_ch // 2,
                                     F_int=in_ch // 4)
        self.conv    = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x    = self.up(x)

        # Handle odd spatial sizes
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh > 0 or dw > 0:
            x = F.pad(x, [0, dw, 0, dh])

        # Apply attention to skip connection
        skip = self.att(g=x, x=skip)

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# ATTENTION U-NET
# ─────────────────────────────────────────────────────────────────────────────

class AttentionUNet(nn.Module):
    """
    Attention U-Net with 4 encoder levels.
    features=(32, 64, 128, 256) – same as standard U-Net for fair comparison.
    The attention gates add ~+10% parameters but significantly improve
    thin structure detection (canals, vessels, nerves).
    """
    def __init__(self, in_channels=1, out_channels=1,
                 features=(32, 64, 128, 256)):
        super().__init__()
        f = features

        # Encoder
        self.enc1     = DoubleConv(in_channels, f[0])
        self.enc2     = Down(f[0], f[1])
        self.enc3     = Down(f[1], f[2])
        self.enc4     = Down(f[2], f[3])

        # Bottleneck
        self.bottleneck = Down(f[3], f[3] * 2)

        # Decoder (with attention gates built into Up)
        self.dec4     = Up(f[3] * 2, f[3])
        self.dec3     = Up(f[3],     f[2])
        self.dec2     = Up(f[2],     f[1])
        self.dec1     = Up(f[1],     f[0])

        # Output
        self.out_conv = nn.Conv2d(f[0], out_channels, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b  = self.bottleneck(s4)
        d4 = self.dec4(b,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return torch.sigmoid(self.out_conv(d1))   # sigmoid output [0,1]


# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    model  = AttentionUNet()
    dummy  = torch.randn(2, 1, 256, 512)
    output = model(dummy)
    print(f"Input  : {dummy.shape}")
    print(f"Output : {output.shape}")
    total  = sum(p.numel() for p in model.parameters())
    print(f"Params : {total:,}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")