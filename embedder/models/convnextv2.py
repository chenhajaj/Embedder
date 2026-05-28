import numpy as np
import torch
from PIL import Image
from .base import BaseEmbedder


_VARIANTS = {
    "convnextv2-tiny":  ("facebook/convnextv2-tiny-1k-224",  768),
    "convnextv2-base":  ("facebook/convnextv2-base-1k-224",  1024),
    "convnextv2-large": ("facebook/convnextv2-large-1k-224", 1536),
    "convnextv2-huge":  ("facebook/convnextv2-huge-1k-224",  2816),
}


class ConvNeXtV2Embedder(BaseEmbedder):
    def __init__(self, variant: str = "convnextv2-base", device: str = "auto", batch_size: int = 32):
        super().__init__(device)
        if variant not in _VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(_VARIANTS)}")
        self._variant = variant
        self._hf_id, self._dim = _VARIANTS[variant]
        self.batch_size = batch_size
        self._model = None
        self._processor = None

    def load(self) -> None:
        from transformers import AutoImageProcessor, ConvNextV2Model
        self._processor = AutoImageProcessor.from_pretrained(self._hf_id)
        self._model = ConvNextV2Model.from_pretrained(self._hf_id).to(self.device).eval()

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        if self._model is None:
            self.load()
        all_embeddings = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]
            inputs = self._processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            # global average pool over spatial dims
            emb = outputs.last_hidden_state.mean(dim=[-2, -1])
            all_embeddings.append(emb.cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._variant
