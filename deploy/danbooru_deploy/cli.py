#!/usr/bin/env python3
"""Command line entry point for DanbooruTagQuery deploy.

Run single-image inference or launch the Gradio web UI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .core import Predictor, preprocess


def _make_scores(predictor: Predictor, image: Image.Image):
    tensor = preprocess(image, predictor.image_size)
    logits = predictor.run(tensor)[0]
    inv = {v: k for k, v in predictor.tag_to_id.items()}
    indices = np.argsort(logits)[::-1]
    return [(inv[int(i)], float(logits[i])) for i in indices]


def _make_predict_fn(checkpoint: str | Path, use_cuda: bool):
    predictor = Predictor(checkpoint, use_cuda=use_cuda)

    def predict_fn(image: Image.Image) -> list[tuple[str, float]]:
        return _make_scores(predictor, image)

    return predict_fn, predictor


def main():
    parser = argparse.ArgumentParser(
        description="DanbooruTagQuery - ONNX deploy",
    )
    parser.add_argument("checkpoint", help="Path to model directory or model.onnx file")
    parser.add_argument(
        "image", nargs="?", default=None,
        help="Image path (run CLI inference; omit to launch web UI)",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--no-ui", action="store_true", help="Force CLI mode (requires image)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    parser.add_argument("--share", action="store_true", help="Public Gradio share link")
    parser.add_argument("--host", default=None, help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Port (default: 7860)")
    args = parser.parse_args()

    use_cuda = not args.cpu

    if args.image:
        predictor = Predictor(args.checkpoint, use_cuda=use_cuda)
        results = _make_scores(predictor, Image.open(args.image).convert("RGB"))
        if args.min_score > 0.0:
            results = [(t, s) for t, s in results if s >= args.min_score]
        if args.top_k is not None and args.top_k > 0:
            results = results[:args.top_k]
        for tag, score in results:
            cat = predictor.category_name(tag)
            print(f"{tag.replace('_', ' '):<28} {score:.4f}  [{cat}]")
    elif args.no_ui:
        parser.error("provide an image path with --no-ui")
    else:
        from .ui import build_app

        predict_fn, predictor = _make_predict_fn(args.checkpoint, use_cuda)
        app = build_app(predict_fn, predictor.cat_map)
        app.launch(
            share=args.share,
            server_name=args.host,
            server_port=args.port,
        )


if __name__ == "__main__":
    sys.exit(main())