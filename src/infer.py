import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import TrainConfig
from .model import ImageTagger
from .transforms import val_transforms
from .defaults import DEFAULT_TOP_K, DEFAULT_MIN_SCORE, DEFAULT_IGNORE_FILE
from .onnx_utils import is_onnx, load_tag_to_id, load_config, Predictor


def load_ignored_tags(path="data/ignored_tags.txt"):
    p = Path(path)
    if not p.exists():
        return set()
    return {
        line.strip()
        for line in p.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }


def predict(image_path, checkpoint, top_k=None, min_score=0.0, ignored_tags=None):
    predictor = Predictor(checkpoint)
    tag_to_id = predictor.tag_to_id

    # prepare image tensor
    with Image.open(image_path) as image:
        tensor = predictor.val_transform(image.convert("RGB")).unsqueeze(0)

    scores = predictor.run(tensor).flatten()
    inverse = {v: k for k, v in tag_to_id.items()}
    indices = np.argsort(scores)[::-1]
    results = [(inverse[int(i)], float(scores[i])) for i in indices]

    if ignored_tags:
        results = [(tag, score) for tag, score in results if tag not in ignored_tags]
    if min_score > 0.0:
        results = [(tag, score) for tag, score in results if score >= min_score]
    if top_k is not None:
        results = results[:top_k]
    return results


def _save_labels(results: list[tuple[str, float]], dest: Path, min_score: float = 0.0):
    """Save comma-separated tags (underscores -> spaces) to dest."""
    tags = [t for t, s in results if s >= min_score]
    labels = ", ".join(t.replace("_", " ") for t in tags)
    dest.write_text(labels)


def _image_extensions() -> set[str]:
    return {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif"}


def batch_infer(
    directory: Path,
    checkpoint: str,
    top_k: int | None = None,
    min_score: float = 0.0,
    ignored_tags: set[str] | None = None,
    save_labels: bool = False,
):
    """Run inference on all images in directory."""
    predictor = Predictor(checkpoint)
    tag_to_id = predictor.tag_to_id
    inverse = {v: k for k, v in tag_to_id.items()}
    exts = _image_extensions()

    images = sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in exts)
    if not images:
        print(f"No images found in {directory}")
        return

    print(f"Processing {len(images)} images in {directory} ...")
    for img_path in images:
        try:
            with Image.open(img_path) as im:
                tensor = predictor.val_transform(im.convert("RGB")).unsqueeze(0)
            scores = predictor.run(tensor).flatten()
            indices = np.argsort(scores)[::-1]
            results = [(inverse[int(i)], float(scores[i])) for i in indices]
            if ignored_tags:
                results = [r for r in results if r[0] not in ignored_tags]
            if min_score > 0.0:
                results = [r for r in results if r[1] >= min_score]
            if top_k is not None:
                results = results[:top_k]

            # print
            for tag, score in results:
                print(f"  {tag:<28} {score:.4f}")

            # save labels
            if save_labels:
                label_path = img_path.with_suffix(".txt")
                _save_labels(results, label_path, min_score=0.0)
                print(f"    -> saved to {label_path.name}")
        except Exception as exc:
            print(f"  FAIL {img_path.name}: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Image file path (not needed with --dir)",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--ignore-file", default=DEFAULT_IGNORE_FILE)
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Directory of images to batch infer",
    )
    parser.add_argument(
        "--save-labels",
        action="store_true",
        help="Save tags as comma-separated {image}.txt (underscores -> spaces); "
        "works with --dir or single image",
    )
    args = parser.parse_args()

    if args.dir is not None:
        ignored = load_ignored_tags(args.ignore_file)
        batch_infer(
            args.dir,
            args.checkpoint,
            top_k=args.top_k,
            min_score=args.min_score or 0.0,
            ignored_tags=ignored,
            save_labels=args.save_labels,
        )
    else:
        if args.image is None:
            parser.error("provide an image or --dir")
        ignored = load_ignored_tags(args.ignore_file)
        results = predict(
            args.image, args.checkpoint, args.top_k, args.min_score, ignored
        )
        print(
            "\n".join(
                f"{tag:<24} {score:.4f}" for tag, score in results
            )
        )
        if args.save_labels:
            dest = Path(args.image).with_suffix(".txt")
            _save_labels(results, dest, min_score=args.min_score or 0.0)
            print(f"Labels saved to {dest}")


if __name__ == "__main__":
    main()
