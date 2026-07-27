# model.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 🐍 SS2D Mamba Block
# ==============================================================================
class SS2DMambaBlock(nn.Module):
    def __init__(self, dim, expand=2):
        super().__init__()
        self.dim = dim
        self.hidden_dim = int(expand * dim)
        self.in_proj = nn.Linear(dim, self.hidden_dim * 2)
        self.conv2d = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, groups=self.hidden_dim)
        self.act = nn.SiLU()
        self.scan_h = nn.GRU(self.hidden_dim, self.hidden_dim // 2, batch_first=True, bidirectional=True)
        self.scan_v = nn.GRU(self.hidden_dim, self.hidden_dim // 2, batch_first=True, bidirectional=True)
        self.out_proj = nn.Linear(self.hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def _forward_ssm(self, x, scan_layer, H, W, transpose=False):
        B, L, C = x.shape
        if transpose:
            x = x.view(B, H, W, C).transpose(1, 2).reshape(B, L, C)

        x = x.float()
        with torch.amp.autocast('cuda', enabled=False):
            x, _ = scan_layer(x)

        if transpose:
            x = x.view(B, W, H, -1).transpose(1, 2).reshape(B, L, -1)
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        res = x
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm(x_flat)
        combined = self.in_proj(x_norm)
        x_gate, x_ssm = combined.chunk(2, dim=-1)

        x_ssm = self.act(self.conv2d(x_ssm.transpose(1, 2).view(B, -1, H, W))).flatten(2).transpose(1, 2)
        y_ssm = self._forward_ssm(x_ssm, self.scan_h, H, W) + self._forward_ssm(x_ssm, self.scan_v, H, W, True)
        y = self.act(x_gate) * y_ssm

        out = self.out_proj(y).transpose(1, 2).view(B, -1, H, W)
        return res + out


# ==============================================================================
# 🧱 Basic Blocks
# ==============================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        return self.conv(x)


class UpSampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x1.shape != x2.shape:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
        return self.conv(torch.cat([x2, x1], dim=1))


# ==============================================================================
# 🚀 PolyMambaUNet
# ==============================================================================
class PolyMambaUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, width=64):
        super().__init__()

        self.inc = DoubleConv(in_ch, width)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(width, width * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(width * 2, width * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(width * 4, width * 8), SS2DMambaBlock(width * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(width * 8, width * 16), SS2DMambaBlock(width * 16))

        self.up1 = UpSampleLayer(width * 16 + width * 8, width * 8)
        self.up2 = UpSampleLayer(width * 8 + width * 4, width * 4)
        self.up3 = UpSampleLayer(width * 4 + width * 2, width * 2)
        self.up4 = UpSampleLayer(width * 2 + width, width)


        self.outc = nn.Conv2d(width, out_ch, kernel_size=1)

        print(f"🐍 PolyMambaUNet Initialized | Pure Denoising Mode (No Variance Output)")

    def forward(self, x, noise_map=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        out = self.outc(x)


        return out


def build_model(cfg):
    return PolyMambaUNet(
        in_ch=cfg['model'].get('in_ch', 1),
        out_ch=cfg['model'].get('out_ch', 1),
        width=cfg['model'].get('width', 32)
    )