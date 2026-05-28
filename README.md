# Embedder

Image embeddings from SOTA CV models — DINOv2, CLIP, SigLIP.

## Install

```bash
pip install -e .
# or
pip install -r requirements.txt && pip install -e . --no-deps
```

## Python API

```python
from embedder import Embedder

# single image
e = Embedder(model="dinov2-large")          # lazy-loads on first call
emb = e.embed("photo.jpg")                  # np.ndarray (1, 1024)

# batch
emb = e.embed(["a.jpg", "b.png", "c.webp"]) # np.ndarray (3, 1024)

# callable shorthand
emb = e("photo.jpg")

# list available models
from embedder import list_models
print(list_models())
```

## CLI

```bash
# embed one image, print JSON to stdout
embedder photo.jpg -m dinov2-large

# embed multiple, save as .npy
embedder *.jpg -m clip-vit-large-patch14 -o embeddings.npy

# glob, JSON output file
embedder "images/*.png" -m siglip-so400m-patch14-384 -o out.json --format json

# list models
embedder --list-models
```

## Models

| Name | Family | Dim | Notes |
|---|---|---|---|
| `dinov2-small` | DINOv2 | 384 | |
| `dinov2-base` | DINOv2 | 768 | |
| `dinov2-large` | DINOv2 | 1024 | **default** |
| `dinov2-giant` | DINOv2 | 1536 | |
| `clip-vit-base-patch32` | CLIP | 768 | raw ViT-B encoder |
| `clip-vit-base-patch16` | CLIP | 768 | raw ViT-B encoder |
| `clip-vit-large-patch14` | CLIP | 1024 | raw ViT-L encoder |
| `clip-vit-large-patch14-336` | CLIP | 1024 | raw ViT-L encoder, higher res |
| `siglip-base-patch16-224` | SigLIP | 768 | |
| `siglip-large-patch16-256` | SigLIP | 1024 | |
| `siglip-so400m-patch14-384` | SigLIP | 1152 | |
| `siglip2-so400m-patch16-384` | SigLIP 2 | 1152 | |
| `convnextv2-tiny` | ConvNeXt V2 | 768 | |
| `convnextv2-base` | ConvNeXt V2 | 1024 | |
| `convnextv2-large` | ConvNeXt V2 | 1536 | |
| `convnextv2-huge` | ConvNeXt V2 | 2816 | |
| `efficientnetv2-s` | EfficientNet V2 | 1280 | |
| `efficientnetv2-m` | EfficientNet V2 | 1280 | |
| `efficientnetv2-l` | EfficientNet V2 | 1280 | |
| `efficientnetv2-xl` | EfficientNet V2 | 1280 | |
| `timm://<id>` | any timm model | varies | escape hatch for 1000+ models |

## Tests

```bash
pytest tests/ -v
```
