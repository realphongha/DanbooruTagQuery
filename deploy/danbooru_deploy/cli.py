#!/usr/bin/env python3
"""Command line entry point for DanbooruTagQuery deploy.

Run single-image inference or launch the Gradio web UI. ``main`` is also the
programmatic entry point used by the Hugging Face Space:

    from danbooru_deploy import main
    main(use_cuda=False)          # CPU-forced for the Space
"""

from __future__ import annotations

import argparse
import os
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


def _resolve_use_cuda(value) -> bool:
    """Tri-state: 'auto' → CUDA-if-possible, True/False forced."""
    if value is True or value is False:
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "yes", "cuda", "gpu"):
            return True
        if v in ("0", "false", "no", "cpu"):
            return False
    # default: try CUDA, fall back to CPU inside Predictor
    return True


def main(argv=None, *, use_cuda="auto", host=None, port=None):
    """DanbooruTagQuery deploy entry point.

    Parameters
    ----------
    argv : list[str] | None
        CLI args (``checkpoint`` positional + optional ``image``). When None,
        ``sys.argv[1:]`` is used.
    use_cuda : bool | str
        ``True`` force CUDA, ``False`` force CPU, ``"auto"`` CUDA-if-possible.
        The HF Space passes ``False`` to stay CPU-default.
    host, port : str | int | None
        Override host/port; falls back to ``GRADIO_SERVER_NAME``/``HOST`` and
        ``GRADIO_SERVER_PORT``/``PORT`` env vars, then Gradio defaults.
    """
    parser = argparse.ArgumentParser(
        description="DanbooruTagQuery - ONNX deploy",
    )
    parser.add_argument(
        "checkpoint", nargs="?", default=None,
        help="Path to model directory or model.onnx file "
             "(default: MODEL_DIR env, then HF hub default variant)",
    )
    parser.add_argument(
        "image", nargs="?", default=None,
        help="Image path (run CLI inference; omit to launch web UI)",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--no-ui", action="store_true", help="Force CLI mode (requires image)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    parser.add_argument("--share", action="store_true", help="Public Gradio share link")
    args = parser.parse_args(argv)

    cuda = _resolve_use_cuda(use_cuda)
    if args.cpu:
        cuda = False

    # ── resolve model (CLI arg → MODEL_DIR env → HF hub) ──────────────────
    from .hub import resolve_checkpoint

    ckpt: Path | None = None
    variant: str | None = None
    variants: list[str] = []
    model_env = os.environ.get("MODEL_DIR")

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    elif model_env:
        env_path = Path(model_env)
        if not env_path.is_dir():
            parser.error(f"MODEL_DIR {env_path} is not a directory")
        onnx_files = list(env_path.glob("*.onnx"))
        if not onnx_files:
            parser.error(f"no .onnx files in {env_path}")
        ckpt = onnx_files[0]
        print(f"Loading local model from MODEL_DIR: {ckpt}")
    else:
        print("Discovering model variants on HF hub …")
        from .hub import (
            discover_model_variants,
            download_model_variant,
            hf_available,
        )
        if not hf_available():
            parser.error(
                "No checkpoint given and huggingface_hub not installed. "
                "Install the [hf] extra or pass a checkpoint."
            )
        variants = discover_model_variants()
        if not variants:
            parser.error("No models found on hub.")
        variant = variants[0]
        print(f"Found variants: {variants}")
        print(f"Downloading default model: {variant} …")
        ckpt = download_model_variant(variant)
        print(f"Downloaded: {ckpt}")

    if not ckpt.exists():
        parser.error(f"{ckpt} not found")

    # ── CLI inference mode ─────────────────────────────────────────────────
    if args.image:
        predictor = Predictor(ckpt, use_cuda=cuda)
        results = _make_scores(predictor, Image.open(args.image).convert("RGB"))
        if args.min_score > 0.0:
            results = [(t, s) for t, s in results if s >= args.min_score]
        if args.top_k is not None and args.top_k > 0:
            results = results[:args.top_k]
        for tag, score in results:
            cat = predictor.category_name(tag)
            print(f"{tag.replace('_', ' '):<28} {score:.4f}  [{cat}]")
        return

    if args.no_ui:
        parser.error("provide an image path with --no-ui")

    # ── web UI ─────────────────────────────────────────────────────────────
    from .ui import build_app

    predict_fn, predictor = _make_predict_fn(ckpt, cuda)

    get_model_predict_fn = None
    if variants:
        def get_model_predict_fn(variant_name: str):
            from .hub import download_model_variant
            onnx_path = download_model_variant(variant_name)
            fn, pred = _make_predict_fn(onnx_path, cuda)
            return fn, pred.cat_map

    app = build_app(
        predict_fn,
        predictor.cat_map,
        model_choices=variants or None,
        get_model_predict_fn=get_model_predict_fn,
        hf_repo="realphongha/danbooru-tag-query",
    )

    h = host or os.environ.get("GRADIO_SERVER_NAME") or os.environ.get("HOST")
    p_str = str(port) if port is not None else (
        os.environ.get("GRADIO_SERVER_PORT") or os.environ.get("PORT")
    )
    p = int(p_str) if p_str else None

    app.launch(server_name=h, server_port=p, share=args.share, ssr_mode=False)


if __name__ == "__main__":
    sys.exit(main())
