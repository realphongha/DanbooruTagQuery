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
from .ui import launch as ui_launch


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

    scores = predictor.run(tensor)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Image file path (not needed with --ui)",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--ignore-file", default=DEFAULT_IGNORE_FILE)
    parser.add_argument("--ui", action="store_true", help="Launch web UI (Gradio)")
    parser.add_argument(
        "--share", action="store_true", help="Create public Gradio share link"
    )
    args = parser.parse_args()
    if args.ui:
        ui_launch(args.checkpoint, share=args.share)
    else:
        if args.image is None:
            parser.error("image is required when not using --ui")
        ignored = load_ignored_tags(args.ignore_file)
        print(
            "\n".join(
                f"{tag:<24} {score:.4f}"
                for tag, score in predict(
                    args.image, args.checkpoint, args.top_k, args.min_score, ignored
                )
            )
        )
