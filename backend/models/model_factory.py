"""
model_factory.py — MindTrace FER Research
==========================================
Provides a unified `get_model(name, num_classes)` factory that returns a
fine-tunable model with a consistent enhanced classification head.

Supported architectures
-----------------------
  resnet18    – ResNet-18  (baseline, ~11M params)
  resnet50    – ResNet-50  (deeper CNN, ~25M params)
  vgg16       – VGG-16     (classic baseline, ~138M params)
  efficientnet– EfficientNet-B0 (~5M params, lightweight)
  mobilenet   – MobileNetV2    (~3.4M params, edge-friendly)
  vit         – ViT-Small/16   (~22M params, transformer; requires timm)

All models use the same 2-layer EnhancedHead for a fair comparison.
Early conv/transformer layers are frozen; the deeper layers and head
are left trainable to fine-tune to the FER domain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    VGG16_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
)

# ──────────────────────────────────────────────────────────────────────
# RECOMMENDED INPUT SIZE PER MODEL
# ──────────────────────────────────────────────────────────────────────
MODEL_IMG_SIZES = {
    "resnet18":    224,
    "resnet50":    224,
    "vgg16":       224,
    "efficientnet": 224,
    "mobilenet":   224,
    "vit":         224,   # ViT patch=16 requires 224×224
}


# ──────────────────────────────────────────────────────────────────────
# SHARED CLASSIFICATION HEAD
# ──────────────────────────────────────────────────────────────────────
class EnhancedHead(nn.Module):
    """
    A two-layer MLP head shared across all backbone architectures.
    in_features → 256 → num_classes  (with BN + Dropout for regularisation).
    """
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────
# MODEL WRAPPER CLASSES
# ──────────────────────────────────────────────────────────────────────
class ResNetModel(nn.Module):
    """Wraps a ResNet backbone (fc replaced with Identity) + enhanced head."""
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone   # backbone.fc == nn.Identity()
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class VGGModel(nn.Module):
    """Uses only VGG's feature extractor + avgpool, discards its classifier."""
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.features = backbone.features
        self.avgpool  = backbone.avgpool
        self.head     = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)          # → (B, 512, 7, 7)
        x = torch.flatten(x, 1)      # → (B, 25088)
        return self.head(x)


class EfficientNetModel(nn.Module):
    """Uses EfficientNet features + avgpool, replaces its classifier."""
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.features = backbone.features
        self.avgpool  = backbone.avgpool
        self.head     = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


class MobileNetModel(nn.Module):
    """Uses MobileNetV2 feature extractor + adaptive avg pool."""
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.features = backbone.features
        self.head     = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.head(x)


class ViTModel(nn.Module):
    """Wraps a timm ViT backbone (num_classes=0) + enhanced head."""
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone   # timm model with num_classes=0
        self.head     = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ──────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────────────────────────────
def _freeze(*modules: nn.Module):
    """Freeze all parameters in the given modules."""
    for m in modules:
        for p in m.parameters():
            p.requires_grad = False


def _unfreeze(*modules: nn.Module):
    """Unfreeze all parameters in the given modules."""
    for m in modules:
        for p in m.parameters():
            p.requires_grad = True


# ──────────────────────────────────────────────────────────────────────
# FACTORY
# ──────────────────────────────────────────────────────────────────────
def get_model(name: str, num_classes: int) -> nn.Module:
    """
    Return a fine-tunable model with an EnhancedHead.

    Parameters
    ----------
    name : str
        One of: resnet18 | resnet50 | vgg16 | efficientnet | mobilenet | vit
    num_classes : int
        Number of output emotion classes (typically 7).

    Returns
    -------
    nn.Module
        Wrapped model ready for training.
    """
    name = name.lower()

    # ── ResNet-18 ──────────────────────────────────────────────────
    if name == "resnet18":
        bb = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        in_f = bb.fc.in_features        # 512
        bb.fc = nn.Identity()
        _freeze(bb)
        _unfreeze(bb.layer2, bb.layer3, bb.layer4)  # unfreeze deeper blocks
        return ResNetModel(bb, EnhancedHead(in_f, num_classes))

    # ── ResNet-50 ──────────────────────────────────────────────────
    elif name == "resnet50":
        bb = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        in_f = bb.fc.in_features        # 2048
        bb.fc = nn.Identity()
        _freeze(bb)
        _unfreeze(bb.layer3, bb.layer4)             # deeper model → freeze more
        return ResNetModel(bb, EnhancedHead(in_f, num_classes))

    # ── VGG-16 ────────────────────────────────────────────────────
    elif name == "vgg16":
        bb = models.vgg16(weights=VGG16_Weights.DEFAULT)
        in_f = 512 * 7 * 7             # 25088 after avgpool(7,7)
        _freeze(bb.features)
        # Unfreeze the last two conv blocks (features[24:] ≈ block 4+5)
        _unfreeze(*[bb.features[i] for i in range(24, len(bb.features))])
        return VGGModel(bb, EnhancedHead(in_f, num_classes))

    # ── EfficientNet-B0 ───────────────────────────────────────────
    elif name == "efficientnet":
        bb = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_f = bb.classifier[1].in_features   # 1280
        _freeze(bb.features)
        # Unfreeze last 3 MBConv blocks
        _unfreeze(*list(bb.features[6:]))
        return EfficientNetModel(bb, EnhancedHead(in_f, num_classes))

    # ── MobileNetV2 ───────────────────────────────────────────────
    elif name == "mobilenet":
        bb = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        in_f = bb.last_channel         # 1280
        _freeze(bb.features)
        # Unfreeze last 5 inverted residual blocks
        _unfreeze(*list(bb.features[14:]))
        return MobileNetModel(bb, EnhancedHead(in_f, num_classes))

    # ── ViT-Small/16 (timm) ───────────────────────────────────────
    elif name == "vit":
        try:
            import timm
        except ImportError:
            raise ImportError(
                "ViT requires the timm library.\n"
                "Install it with:  pip install timm"
            )
        bb = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            num_classes=0,   # removes the default classification head
        )
        in_f = bb.num_features         # 384 for vit_small
        _freeze(bb)
        # Unfreeze the last 4 transformer blocks + layer norm
        _unfreeze(*list(bb.blocks[-4:]), bb.norm)
        return ViTModel(bb, EnhancedHead(in_f, num_classes))

    else:
        raise ValueError(
            f"Unknown model '{name}'. "
            "Choose from: resnet18 | resnet50 | vgg16 | efficientnet | mobilenet | vit"
        )


# ──────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────
def count_parameters(model: nn.Module):
    """Returns (total_params, trainable_params) as integers."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_inference_time(
    model: nn.Module,
    device: torch.device,
    img_size: int = 224,
    n_runs: int = 100,
) -> float:
    """
    Measure average single-image inference time in milliseconds.

    Parameters
    ----------
    model   : nn.Module  – model in eval mode
    device  : torch.device
    img_size: int        – spatial dimension (assumes square input)
    n_runs  : int        – number of timed forward passes

    Returns
    -------
    float : average ms per image
    """
    import time
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size).to(device)

    # GPU warm-up
    with torch.no_grad():
        for _ in range(10):
            model(dummy)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy)
    elapsed = time.perf_counter() - start

    return (elapsed / n_runs) * 1000.0   # convert to ms
