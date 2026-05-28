from .base import BaseEmbedder
from .dinov2 import DINOv2Embedder
from .clip_model import CLIPEmbedder
from .siglip import SigLIPEmbedder
from .convnextv2 import ConvNeXtV2Embedder
from .efficientnetv2 import EfficientNetV2Embedder
from .timm_model import TimmEmbedder

_MODEL_MAP: dict[str, tuple[type[BaseEmbedder], str]] = {}

for _variant in ("dinov2-small", "dinov2-base", "dinov2-large", "dinov2-giant"):
    _MODEL_MAP[_variant] = (DINOv2Embedder, _variant)

for _variant in ("clip-vit-base-patch32", "clip-vit-base-patch16",
                 "clip-vit-large-patch14", "clip-vit-large-patch14-336"):
    _MODEL_MAP[_variant] = (CLIPEmbedder, _variant)

for _variant in ("siglip-base-patch16-224", "siglip-large-patch16-256",
                 "siglip-so400m-patch14-384", "siglip2-so400m-patch16-384"):
    _MODEL_MAP[_variant] = (SigLIPEmbedder, _variant)

for _variant in ("convnextv2-tiny", "convnextv2-base", "convnextv2-large", "convnextv2-huge"):
    _MODEL_MAP[_variant] = (ConvNeXtV2Embedder, _variant)

for _variant in ("efficientnetv2-s", "efficientnetv2-m", "efficientnetv2-l", "efficientnetv2-xl"):
    _MODEL_MAP[_variant] = (EfficientNetV2Embedder, _variant)


def list_models() -> list[str]:
    return sorted(_MODEL_MAP) + ["timm://<any-timm-id>"]


def load_model(name: str, device: str = "auto", batch_size: int = 32) -> BaseEmbedder:
    if name.startswith("timm://"):
        return TimmEmbedder(timm_id=name[len("timm://"):], device=device, batch_size=batch_size)
    if name not in _MODEL_MAP:
        raise ValueError(f"Unknown model '{name}'. Available: {list_models()}")
    cls, variant = _MODEL_MAP[name]
    return cls(variant=variant, device=device, batch_size=batch_size)


__all__ = [
    "BaseEmbedder", "DINOv2Embedder", "CLIPEmbedder", "SigLIPEmbedder",
    "ConvNeXtV2Embedder", "EfficientNetV2Embedder", "TimmEmbedder",
    "load_model", "list_models",
]
