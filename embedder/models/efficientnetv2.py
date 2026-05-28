import numpy as np
import torch
from PIL import Image
from .base import BaseEmbedder


_VARIANTS = {
    "efficientnetv2-s": ("tf_efficientnetv2_s.in21k_ft_in1k", 1280),
    "efficientnetv2-m": ("tf_efficientnetv2_m.in21k_ft_in1k", 1280),
    "efficientnetv2-l": ("tf_efficientnetv2_l.in21k_ft_in1k", 1280),
    "efficientnetv2-xl": ("tf_efficientnetv2_xl.in21k_ft_in1k", 1280),
}

_INPUT_SIZE = {
    "efficientnetv2-s":  384,
    "efficientnetv2-m":  480,
    "efficientnetv2-l":  480,
    "efficientnetv2-xl": 512,
}


class EfficientNetV2Embedder(BaseEmbedder):
    def __init__(self, variant: str = "efficientnetv2-l", device: str = "auto", batch_size: int = 32):
        super().__init__(device)
        if variant not in _VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(_VARIANTS)}")
        self._variant = variant
        self._timm_id, self._dim = _VARIANTS[variant]
        self._input_size = _INPUT_SIZE[variant]
        self.batch_size = batch_size
        self._model = None
        self._transform = None

    def load(self) -> None:
        import timm
        from timm.data import resolve_data_config, create_transform
        self._model = timm.create_model(self._timm_id, pretrained=True, num_classes=0).to(self.device).eval()
        cfg = resolve_data_config({}, model=self._model)
        self._transform = create_transform(**cfg)

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        if self._model is None:
            self.load()
        import torch
        all_embeddings = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]
            tensors = torch.stack([self._transform(img) for img in batch]).to(self.device)
            with torch.no_grad():
                emb = self._model(tensors)
            all_embeddings.append(emb.cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._variant
