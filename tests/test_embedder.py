"""Tests use small synthetic images — no network required if models are cached."""
import numpy as np
import pytest
from pathlib import Path
from PIL import Image


@pytest.fixture
def tmp_image(tmp_path) -> Path:
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    p = tmp_path / "test.png"
    img.save(p)
    return p


@pytest.fixture
def tmp_images(tmp_path) -> list[Path]:
    paths = []
    for i in range(4):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        p = tmp_path / f"img_{i}.png"
        img.save(p)
        paths.append(p)
    return paths


def test_list_models():
    from embedder import list_models
    models = list_models()
    assert "dinov2-large" in models
    assert "clip-vit-large-patch14" in models
    assert "siglip-so400m-patch14-384" in models


def test_embedder_single_image(tmp_image):
    from embedder import Embedder
    e = Embedder(model="dinov2-small", device="cpu", batch_size=1)
    emb = e.embed(tmp_image)
    assert emb.shape == (1, 384)
    assert emb.dtype == np.float32


def test_embedder_batch(tmp_images):
    from embedder import Embedder
    e = Embedder(model="dinov2-small", device="cpu", batch_size=2)
    emb = e.embed(tmp_images)
    assert emb.shape == (4, 384)


def test_embedder_callable(tmp_image):
    from embedder import Embedder
    e = Embedder(model="dinov2-small", device="cpu")
    emb = e(tmp_image)
    assert emb.shape[0] == 1


def test_clip_embed(tmp_image):
    from embedder import Embedder
    e = Embedder(model="clip-vit-base-patch32", device="cpu")
    emb = e.embed(tmp_image)
    assert emb.shape == (1, 768)  # raw ViT-B encoder dim, not projected
    # CLIP embeddings are L2-normalized
    norm = np.linalg.norm(emb, axis=-1)
    np.testing.assert_allclose(norm, 1.0, atol=1e-5)


def test_missing_file_raises():
    from embedder import Embedder
    e = Embedder(model="dinov2-small", device="cpu")
    with pytest.raises(FileNotFoundError):
        e.embed("/nonexistent/image.png")


def test_unknown_model_raises():
    from embedder.models import load_model
    with pytest.raises(ValueError, match="Unknown model"):
        load_model("not-a-model")
