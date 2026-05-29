import torch
import torch.nn as nn
from torchvision import models

def _replace_conv_in_channels(conv: nn.Conv2d, in_ch: int = 4, zero_init_extra: bool = True) -> nn.Conv2d:
    
    """
    Create a new Conv2d with the same hyperparams but different in_channels.
    Copies pretrained weights for the first 3 channels; initializes extra channel(s) to 0 (or kaiming if desired).
    """
    
    assert isinstance(conv, nn.Conv2d)
    new_conv = nn.Conv2d(
        in_channels=in_ch,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=(conv.bias is not None),
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        # copy existing channels
        c_old = conv.weight.shape[1]
        c_copy = min(c_old, in_ch)
        new_conv.weight[:, :c_copy].copy_(conv.weight[:, :c_copy])

        # init extra channels
        if in_ch > c_old:
            if zero_init_extra:
                new_conv.weight[:, c_old:in_ch].zero_()
            else:
                nn.init.kaiming_normal_(new_conv.weight[:, c_old:in_ch], mode="fan_out", nonlinearity="relu")

        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias)

    return new_conv


class PathwayModel(nn.Module):
    """
    4-channel input (RGB + canvas) -> backbone -> MLP head -> 361 classes

    Supported backbone_name:
      - "resnet18", "resnet50"
      - "convnext_tiny", "convnext_small"
      - "swin_tiny", "swin_small"
      - "vit_tiny", "vit_small"   (requires timm)
      - optional: "vit_b_16" (torchvision) as a stand-in if you don't want timm
    """

    def __init__(self, num_classes: int = 361, backbone_name: str = "resnet18", dropout: float = 0.5):
        super().__init__()
        self.backbone_name = backbone_name.lower()

        self.backbone, feat_dim, hidden_dim, self._backbone_kind = self._build_backbone(self.backbone_name)

        # Policy head (Outputs 361 classes)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def _build_backbone(self, name: str):
        # -------------------------
        # ResNet family (torchvision)
        # -------------------------
        if name == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            backbone.conv1 = _replace_conv_in_channels(backbone.conv1, in_ch=4, zero_init_extra=True)
            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            hidden_dim = 256
            kind = "vector_out"  # backbone(x) -> (B, feat_dim)
            return backbone, feat_dim, hidden_dim, kind

        if name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            backbone.conv1 = _replace_conv_in_channels(backbone.conv1, in_ch=4, zero_init_extra=True)
            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            hidden_dim = 512
            kind = "vector_out"
            return backbone, feat_dim, hidden_dim, kind

        # -------------------------
        # ConvNeXt (torchvision)
        # -------------------------
        if name == "convnext_tiny":
            backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            # first conv is patchify conv at features[0][0]
            backbone.features[0][0] = _replace_conv_in_channels(backbone.features[0][0], in_ch=4, zero_init_extra=True)
            # remove classifier; keep avgpool
            feat_dim = backbone.classifier[2].in_features
            backbone.classifier = nn.Identity()
            hidden_dim = 512
            kind = "needs_flatten"  # backbone(x) -> (B, C, 1, 1) after avgpool if classifier removed
            return backbone, feat_dim, hidden_dim, kind

        if name == "convnext_small":
            backbone = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
            backbone.features[0][0] = _replace_conv_in_channels(backbone.features[0][0], in_ch=4, zero_init_extra=True)
            feat_dim = backbone.classifier[2].in_features
            backbone.classifier = nn.Identity()
            hidden_dim = 768
            kind = "needs_flatten"
            return backbone, feat_dim, hidden_dim, kind

        # -------------------------
        # Swin Transformer (torchvision)
        # -------------------------
        if name == "swin_tiny":
            backbone = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
            # patch embedding conv is features[0][0]
            backbone.features[0][0] = _replace_conv_in_channels(backbone.features[0][0], in_ch=4, zero_init_extra=True)
            feat_dim = backbone.head.in_features
            backbone.head = nn.Identity()
            hidden_dim = 512
            kind = "vector_out"
            return backbone, feat_dim, hidden_dim, kind

        if name == "swin_small":
            backbone = models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1)
            backbone.features[0][0] = _replace_conv_in_channels(backbone.features[0][0], in_ch=4, zero_init_extra=True)
            feat_dim = backbone.head.in_features
            backbone.head = nn.Identity()
            hidden_dim = 768
            kind = "vector_out"
            return backbone, feat_dim, hidden_dim, kind

        # -------------------------
        # ViT (tiny/small via timm; torchvision has vit_b_16 etc.)
        # -------------------------
        if name in ("vit_tiny", "vit_small"):
            try:
                import timm
            except Exception as e:
                raise RuntimeError(
                    "vit_tiny/vit_small requires `timm`. Install it (pip install timm) "
                    "or use torchvision ViT like backbone_name='vit_b_16'."
                ) from e

            timm_name = "vit_tiny_patch16_224" if name == "vit_tiny" else "vit_small_patch16_224"
            backbone = timm.create_model(timm_name, pretrained=True, num_classes=0)  # num_classes=0 -> feature extractor
            # patch embedding conv: backbone.patch_embed.proj
            backbone.patch_embed.proj = _replace_conv_in_channels(backbone.patch_embed.proj, in_ch=4, zero_init_extra=True)
            feat_dim = backbone.num_features
            hidden_dim = 512 if name == "vit_tiny" else 768
            kind = "vector_out"
            return backbone, feat_dim, hidden_dim, kind

        if name == "vit_b_16":
            # torchvision fallback (base, not tiny/small)
            backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            backbone.conv_proj = _replace_conv_in_channels(backbone.conv_proj, in_ch=4, zero_init_extra=True)
            # heads is a Sequential; replace with Identity to get features from CLS token
            feat_dim = backbone.heads.head.in_features
            backbone.heads = nn.Identity()
            hidden_dim = 768
            kind = "vector_out"
            return backbone, feat_dim, hidden_dim, kind

        raise ValueError(
            f"Unknown backbone_name='{name}'. "
            "Choose from: resnet18, resnet50, convnext_tiny, convnext_small, "
            "swin_tiny, swin_small, vit_tiny, vit_small, vit_b_16."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)

        # ConvNeXt with classifier removed often yields (B, C, 1, 1)
        if self._backbone_kind == "needs_flatten":
            feats = torch.flatten(feats, 1)

        logits = self.head(feats)
        return logits
