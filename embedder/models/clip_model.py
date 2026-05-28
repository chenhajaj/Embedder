import numpy as np
import torch
from PIL import Image
from .base import BaseEmbedder


_VARIANTS = {
    # dims are raw ViT encoder dims (CLIPVisionModel.pooler_output), not projected
    "clip-vit-base-patch32":      ("openai/clip-vit-base-patch32",      768),
    "clip-vit-base-patch16":      ("openai/clip-vit-base-patch16",      768),
    "clip-vit-large-patch14":     ("openai/clip-vit-large-patch14",     1024),
    "clip-vit-large-patch14-336": ("openai/clip-vit-large-patch14-336", 1024),
}


class CLIPEmbedder(BaseEmbedder):
    def __init__(self, variant: str = "clip-vit-large-patch14", device: str = "auto", batch_size: int = 32):
        super().__init__(device)
        if variant not in _VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(_VARIANTS)}")
        self._variant = variant
        self._hf_id, self._dim = _VARIANTS[variant]
        self.batch_size = batch_size
        self._model = None
        self._processor = None

    def load(self) -> None:
        from transformers import CLIPImageProcessor, CLIPVisionModel
        self._processor = CLIPImageProcessor.from_pretrained(self._hf_id)
        self._model = CLIPVisionModel.from_pretrained(self._hf_id).to(self.device).eval()

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        if self._model is None:
            self.load()
        all_embeddings = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]
            inputs = self._processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                emb = self._model(**inputs).pooler_output
            # L2-normalize (standard for CLIP)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embeddings.append(emb.cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._variant
