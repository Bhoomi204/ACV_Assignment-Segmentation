import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class BaselineUNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 21):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.bottleneck = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward_features(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return e1, d2, d1

    def forward(self, x):
        _, _, d1 = self.forward_features(x)
        return self.head(d1), None

class BoundaryGatedUNet(BaselineUNet):
    def __init__(self, in_channels: int = 3, num_classes: int = 21):
        super().__init__(in_channels, num_classes)
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        self.shape_proj = nn.Sequential(
            nn.Conv2d(65, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.gate_conv = nn.Conv2d(32 + 128, 32, kernel_size=1, bias=True)
        self.shape_refine = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.bnd_head = nn.Conv2d(32, 1, kernel_size=1)

    def compute_sobel_cue(self, x):
        gray = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x.to(x.dtype), padding=1)
        gy = F.conv2d(gray, self.sobel_y.to(x.dtype), padding=1)
        return torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)

    def forward(self, x):
        e1, d2, d1 = self.forward_features(x)
        grad_cue = self.compute_sobel_cue(x)
        s = self.shape_proj(torch.cat([e1, grad_cue], dim=1))
        r = F.interpolate(d2, size=s.shape[-2:], mode="bilinear", align_corners=False)

        alpha = torch.sigmoid(self.gate_conv(torch.cat([s, r], dim=1)))
        s_gated = self.shape_refine((s * alpha) + s)

        bnd_logits = self.bnd_head(s_gated)
        bnd_prob = torch.sigmoid(bnd_logits)
        d1_sharp = d1 * (1.0 + bnd_prob)
        return self.head(d1_sharp), bnd_logits
