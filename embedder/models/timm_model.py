"""Generic timm-backed embedder. Use model_name='timm://hf-hub:org/repo' or any timm id."""
import numpy as np
import torch
from PIL import Image
from .base import BaseEmbedder


class TimmEmbedder(BaseEmbedder):
    """Wraps any timm model. num_classes=0 returns penultimate features."""

    def __init__(self, timm_id: str, device: str = "auto", batch_size: int = 32):
        super().__init__(device)
        self._timm_id = timm_id
        self.batch_size = batch_size
        self._model = None
        self._transform = None
        self._dim: int | None = None

    def load(self) -> None:
        import timm
        from timm.data import resolve_data_config, create_transform
        self._model = timm.create_model(self._timm_id, pretrained=True, num_classes=0).to(self.device).eval()
        cfg = resolve_data_config({}, model=self._model)
        self._transform = create_transform(**cfg)
        # infer dim with a dummy forward pass
        dummy = torch.zeros(1, 3, cfg["input_size"][1], cfg["input_size"][2]).to(self.device)
        with torch.no_grad():
            out = self._model(dummy)
        self._dim = out.shape[-1]

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        if self._model is None:
            self.load()
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
        if self._dim is None:
            self.load()
        return self._dim

    @property
    def model_name(self) -> str:
        return f"timm://{self._timm_id}"
