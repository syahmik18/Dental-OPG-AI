"""
attention_unet_v2.py
Improved Attention U-Net for mandibular canal segmentation.

Key improvements over v1:
  1. Residual connections in every DoubleConv block
  2. ASPP (Atrous Spatial Pyramid Pooling) bottleneck for multi-scale context
  3. Larger feature maps (64, 128, 256, 512) for finer canal detail
  4. Deep supervision (auxiliary outputs at dec3 and dec2)
  5. Squeeze-and-Excitation channel attention in encoder blocks

Input : (B, 1, H, W) grayscale panoramic X-ray (512×256)
Output: (B, 1, H, W) sigmoid probability map
        + optional aux outputs during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """Squeeze-and-Excitation: recalibrates channel-wise features."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ResDoubleConv(nn.Module):
    """Conv→BN→ReLU→Dropout→Conv→BN + residual projection + SE."""
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        # 1×1 projection if channels differ
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if in_ch != out_ch else nn.Identity()

        self.se   = SEBlock(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.se(self.conv(x)) + self.proj(x))


class Down(nn.Module):
    """MaxPool → ResDoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResDoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.pool_conv(x)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling – captures multi-scale canal context.
    Uses dilations suited to 512×256 input (small to medium receptive fields).
    """
    def __init__(self, in_ch, out_ch, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            for d in dilations
        ])
        # Global average pooling branch
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(dilations) + 1), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        h, w    = x.shape[2:]
        feats   = [b(x) for b in self.branches]
        gap_out = F.interpolate(self.gap(x), size=(h, w),
                                mode='bilinear', align_corners=False)
        feats.append(gap_out)
        return self.project(torch.cat(feats, dim=1))


class AttentionGate(nn.Module):
    """Soft-attention gate for skip connections."""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g,   F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l,   F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1,     1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:],
                              mode='bilinear', align_corners=False)
        psi = self.relu(self.W_g(g) + self.W_x(x))
        return x * self.psi(psi)


class Up(nn.Module):
    """ConvTranspose → AttentionGate → Concat → ResDoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.att  = AttentionGate(F_g=in_ch // 2,
                                  F_l=in_ch // 2,
                                  F_int=in_ch // 4)
        self.conv = ResDoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x   = self.up(x)
        dh  = skip.size(2) - x.size(2)
        dw  = skip.size(3) - x.size(3)
        if dh > 0 or dw > 0:
            x = F.pad(x, [0, dw, 0, dh])
        skip = self.att(g=x, x=skip)
        return self.conv(torch.cat([skip, x], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# ATTENTION U-NET V2
# ─────────────────────────────────────────────────────────────────────────────

class AttentionUNetV2(nn.Module):
    """
    Attention U-Net V2 – larger capacity + ASPP + residuals + deep supervision.

    features=(64, 128, 256, 512): 2× the original capacity.
    deep_supervision=True adds auxiliary sigmoid heads at dec3 and dec2 levels.
    These aux outputs are only used during training (weighted loss),
    improving gradient flow to lower decoder layers.
    """
    def __init__(self, in_channels=1, out_channels=1,
                 features=(64, 128, 256, 512),
                 deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        f = features

        #  Encoder
        self.enc1 = ResDoubleConv(in_channels, f[0])
        self.enc2 = Down(f[0], f[1])
        self.enc3 = Down(f[1], f[2])
        self.enc4 = Down(f[2], f[3])

        # Bottleneck (ASPP)
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            ResDoubleConv(f[3], f[3] * 2),
            ASPP(f[3] * 2, f[3] * 2),
        )

        # Decoder
        self.dec4 = Up(f[3] * 2, f[3])
        self.dec3 = Up(f[3],     f[2])
        self.dec2 = Up(f[2],     f[1])
        self.dec1 = Up(f[1],     f[0])

        # Output head 
        self.out_conv = nn.Conv2d(f[0], out_channels, 1)

        # Auxiliary deep-supervision heads 
        if deep_supervision:
            self.aux3 = nn.Conv2d(f[2], out_channels, 1)   # after dec3
            self.aux2 = nn.Conv2d(f[1], out_channels, 1)   # after dec2

    def forward(self, x):
        H, W = x.shape[2:]

        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        # Bottleneck
        b  = self.bottleneck(s4)

        # Decoder
        d4 = self.dec4(b,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        out = torch.sigmoid(self.out_conv(d1))

        if self.deep_supervision and self.training:
            # Upsample aux outputs to input resolution
            aux3 = torch.sigmoid(
                F.interpolate(self.aux3(d3), size=(H, W),
                              mode='bilinear', align_corners=False))
            aux2 = torch.sigmoid(
                F.interpolate(self.aux2(d2), size=(H, W),
                              mode='bilinear', align_corners=False))
            return out, aux3, aux2

        return out


# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    model  = AttentionUNetV2()
    dummy  = torch.randn(2, 1, 256, 512)

    model.train()
    out, a3, a2 = model(dummy)
    print(f"Input      : {dummy.shape}")
    print(f"Main out   : {out.shape}")
    print(f"Aux3 out   : {a3.shape}")
    print(f"Aux2 out   : {a2.shape}")

    model.eval()
    out = model(dummy)
    print(f"Eval out   : {out.shape}")

    total = sum(p.numel() for p in model.parameters())
    print(f"Params     : {total:,}")
