from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image


class BaseEmbedder(ABC):
    def __init__(self, device: str = "auto"):
        import torch
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def embed(self, images: list[Image.Image]) -> np.ndarray:
        ...

    def embed_paths(self, paths: list[Union[str, Path]]) -> np.ndarray:
        images = [Image.open(p).convert("RGB") for p in paths]
        return self.embed(images)

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...
