from pathlib import Path
from typing import Union
import numpy as np
from .models import load_model, list_models, BaseEmbedder

ImageInput = Union[str, Path, list[Union[str, Path]]]


class Embedder:
    """High-level interface. Lazy-loads the model on first call."""

    def __init__(self, model: str = "dinov2-large", device: str = "auto", batch_size: int = 32):
        self._model_name = model
        self._device = device
        self._batch_size = batch_size
        self._backend: BaseEmbedder | None = None

    def _ensure_loaded(self) -> None:
        if self._backend is None:
            self._backend = load_model(self._model_name, self._device, self._batch_size)
            self._backend.load()

    def __call__(self, paths: ImageInput) -> np.ndarray:
        return self.embed(paths)

    def embed(self, paths: ImageInput) -> np.ndarray:
        self._ensure_loaded()
        if isinstance(paths, (str, Path)):
            paths = [paths]
        paths = [Path(p) for p in paths]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Images not found: {missing}")
        return self._backend.embed_paths(paths)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        self._ensure_loaded()
        return self._backend.embedding_dim

    @staticmethod
    def available_models() -> list[str]:
        return list_models()
