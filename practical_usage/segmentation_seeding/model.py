import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvGNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        groups = 32
        while out_ch % groups != 0 and groups > 1:
            groups //= 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SwinEncoder(nn.Module):
    """
    Returns 4 feature maps at strides 4, 8, 16, 32.
    """
    def __init__(self, backbone_name: str = 'swin_tiny', pretrained: bool = True):
        super().__init__()
        name = backbone_name.lower()

        if name == 'swin_tiny':
            weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.swin_t(weights=weights)
        elif name == 'swin_small':
            weights = models.Swin_S_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.swin_s(weights=weights)
        else:
            raise ValueError("backbone_name must be 'swin_tiny' or 'swin_small'")

        self.features = backbone.features
        self.norm = backbone.norm
        self.out_channels = [96, 192, 384, 768]

    @staticmethod
    def _to_nchw(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor):
        x = self.features[0](x)

        x = self.features[1](x)
        c1 = self._to_nchw(x)

        x = self.features[2](x)
        x = self.features[3](x)
        c2 = self._to_nchw(x)

        x = self.features[4](x)
        x = self.features[5](x)
        c3 = self._to_nchw(x)

        x = self.features[6](x)
        x = self.features[7](x)
        x = self.norm(x)
        c4 = self._to_nchw(x)

        return c1, c2, c3, c4


class FPNDecoder(nn.Module):
    def __init__(self, in_channels, decoder_dim: int = 96):
        super().__init__()
        self.lat1 = nn.Conv2d(in_channels[0], decoder_dim, kernel_size=1)
        self.lat2 = nn.Conv2d(in_channels[1], decoder_dim, kernel_size=1)
        self.lat3 = nn.Conv2d(in_channels[2], decoder_dim, kernel_size=1)
        self.lat4 = nn.Conv2d(in_channels[3], decoder_dim, kernel_size=1)

        self.smooth1 = ConvGNReLU(decoder_dim, decoder_dim)
        self.smooth2 = ConvGNReLU(decoder_dim, decoder_dim)
        self.smooth3 = ConvGNReLU(decoder_dim, decoder_dim)
        self.smooth4 = ConvGNReLU(decoder_dim, decoder_dim)

        self.fuse = nn.Sequential(
            ConvGNReLU(decoder_dim * 4, decoder_dim),
            ConvGNReLU(decoder_dim, decoder_dim),
        )
        self.head = nn.Conv2d(decoder_dim, 1, kernel_size=1)

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

    def forward(self, features, out_hw):
        c1, c2, c3, c4 = features

        p4 = self.smooth4(self.lat4(c4))
        p3 = self.smooth3(self.lat3(c3) + self._up(p4, c3))
        p2 = self.smooth2(self.lat2(c2) + self._up(p3, c2))
        p1 = self.smooth1(self.lat1(c1) + self._up(p2, c1))

        p2_up = self._up(p2, p1)
        p3_up = self._up(p3, p1)
        p4_up = self._up(p4, p1)

        fused = self.fuse(torch.cat([p1, p2_up, p3_up, p4_up], dim=1))
        logits = self.head(fused)
        logits = F.interpolate(logits, size=out_hw, mode='bilinear', align_corners=False)
        return logits


class PathwaySegmentationModel(nn.Module):
    """
    Standalone RGB segmentation model.
    Input:  (B, 3, H, W)
    Output: (B, 1, H, W) logits
    """
    def __init__(self, backbone_name: str = 'swin_tiny', pretrained: bool = True, decoder_dim: int = 96):
        super().__init__()
        self.encoder = SwinEncoder(backbone_name=backbone_name, pretrained=pretrained)
        self.decoder = FPNDecoder(self.encoder.out_channels, decoder_dim=decoder_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_hw = x.shape[-2:]
        features = self.encoder(x)
        logits = self.decoder(features, out_hw=out_hw)
        return logits
