import numpy as np
import torch
from PIL import Image
from .base import BaseEmbedder


_VARIANTS = {
    "dinov2-small":  ("facebook/dinov2-small",  384),
    "dinov2-base":   ("facebook/dinov2-base",   768),
    "dinov2-large":  ("facebook/dinov2-large",  1024),
    "dinov2-giant":  ("facebook/dinov2-giant",  1536),
}


class DINOv2Embedder(BaseEmbedder):
    def __init__(self, variant: str = "dinov2-large", device: str = "auto", batch_size: int = 32):
        super().__init__(device)
        if variant not in _VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(_VARIANTS)}")
        self._variant = variant
        self._hf_id, self._dim = _VARIANTS[variant]
        self.batch_size = batch_size
        self._model = None
        self._processor = None

    def load(self) -> None:
        from transformers import AutoImageProcessor, AutoModel
        self._processor = AutoImageProcessor.from_pretrained(self._hf_id)
        self._model = AutoModel.from_pretrained(self._hf_id).to(self.device).eval()

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        if self._model is None:
            self.load()
        all_embeddings = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]
            inputs = self._processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            # CLS token
            emb = outputs.last_hidden_state[:, 0].cpu().numpy()
            all_embeddings.append(emb)
        return np.concatenate(all_embeddings, axis=0)

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._variant
