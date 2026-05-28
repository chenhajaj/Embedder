import argparse
import json
import sys
from pathlib import Path
import numpy as np
from .pipeline import Embedder
from .models import list_models


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="embedder",
        description="Embed images with SOTA CV models (DINOv2, CLIP, SigLIP).",
    )
    p.add_argument("images", nargs="+", type=Path, metavar="IMAGE",
                   help="Path(s) to image file(s) or glob pattern")
    p.add_argument("-m", "--model", default="dinov2-large",
                   choices=list_models(), metavar="MODEL",
                   help=f"Model to use (default: dinov2-large). Choices: {list_models()}")
    p.add_argument("-d", "--device", default="auto",
                   help="Device: auto | cpu | cuda | mps (default: auto)")
    p.add_argument("-b", "--batch-size", type=int, default=32, dest="batch_size",
                   help="Batch size (default: 32)")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Save embeddings as .npy file (default: print to stdout)")
    p.add_argument("--format", choices=["npy", "json"], default="npy",
                   help="Output format when --output is given (default: npy)")
    p.add_argument("--list-models", action="store_true",
                   help="List available models and exit")
    return p


def expand_globs(paths: list[Path]) -> list[Path]:
    expanded = []
    for p in paths:
        if any(c in str(p) for c in ("*", "?", "[")):
            matches = sorted(Path(".").glob(str(p)))
            if not matches:
                print(f"Warning: no files matched '{p}'", file=sys.stderr)
            expanded.extend(matches)
        else:
            expanded.append(p)
    return expanded


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_models:
        for m in list_models():
            print(m)
        sys.exit(0)

    paths = expand_globs(args.images)
    if not paths:
        parser.error("No images provided.")

    embedder = Embedder(model=args.model, device=args.device, batch_size=args.batch_size)
    print(f"Model: {args.model}  |  Images: {len(paths)}  |  Device: {embedder._device}",
          file=sys.stderr)

    embeddings = embedder.embed(paths)
    print(f"Embeddings shape: {embeddings.shape}", file=sys.stderr)

    if args.output:
        if args.format == "npy":
            np.save(args.output, embeddings)
        else:
            data = {str(p): embeddings[i].tolist() for i, p in enumerate(paths)}
            with open(args.output, "w") as f:
                json.dump(data, f)
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        # print as JSON to stdout
        data = {str(p): embeddings[i].tolist() for i, p in enumerate(paths)}
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
