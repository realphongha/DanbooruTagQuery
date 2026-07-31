#!/usr/bin/env python3
"""
Standalone deployment script for DanbooruTagQuery.

Dependencies: onnxruntime / onnxruntime-gpu, numpy, Pillow, gradio
No PyTorch or torchvision needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ── constants ───────────────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── category map ────────────────────────────────────────────────────────────

CATEGORY_MAP = {
    0: "general",
    1: "artist",
    3: "copyright",
    4: "character",
    5: "meta",
}


def get_category_name(cat_map: dict[str, int], tag: str) -> str:
    cat_id = cat_map.get(tag, 0)
    return CATEGORY_MAP.get(cat_id, "general")


# ── sidecar loading ─────────────────────────────────────────────────────────


def _resolve_model_dir(checkpoint: str | Path) -> Path:
    """Resolve model directory from path.

    Accepts:
        - path/to/model_dir          (directory)
        - path/to/model_dir/model.onnx  (model.onnx file → parent dir)
    """
    ckpt = Path(checkpoint)
    if ckpt.name == "model.onnx":
        return ckpt.parent
    return ckpt


def load_tag_to_id(checkpoint: str | Path) -> dict[str, int]:
    model_dir = _resolve_model_dir(checkpoint)
    sidecar = model_dir / "tag_to_id.json"
    if not sidecar.exists():
        raise FileNotFoundError(f"Missing tag map: {sidecar}")
    return json.loads(sidecar.read_text())


def load_config(checkpoint: str | Path) -> dict:
    model_dir = _resolve_model_dir(checkpoint)
    sidecar = model_dir / "config.json"
    if not sidecar.exists():
        return {"image_size": 448}
    return json.loads(sidecar.read_text())


def load_category_map(checkpoint: str | Path) -> dict[str, int]:
    model_dir = _resolve_model_dir(checkpoint)
    sidecar = model_dir / "tag_category.json"
    if not sidecar.exists():
        raise FileNotFoundError(
            f"Missing tag_category.json: {sidecar}\n"
            f"Run: python -m tools.prebuild_cache --tag-map {model_dir}/tag_to_id.json"
        )
    return json.loads(sidecar.read_text())


# ── image preprocessing (PIL + numpy, no torch) ─────────────────────────────


def preprocess(image: Image.Image, image_size: int = 448) -> np.ndarray:
    """Resize + pad + normalize to (1, 3, H, W) float32 numpy array."""
    w, h = image.size
    scale = image_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    left = (image_size - new_w) // 2
    top = (image_size - new_h) // 2
    canvas.paste(image, (left, top))

    arr = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
    arr[0] = (arr[0] - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    arr[1] = (arr[1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
    arr[2] = (arr[2] - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
    return arr[np.newaxis, ...]


# ── ONNX predictor ──────────────────────────────────────────────────────────


class Predictor:
    """ONNX-only predictor. No PyTorch dependency."""

    def __init__(self, checkpoint: str | Path):
        import onnxruntime as ort

        self.checkpoint = str(checkpoint)
        self.tag_to_id = load_tag_to_id(self.checkpoint)
        self.cat_map = load_category_map(self.checkpoint)
        cfg = load_config(self.checkpoint)
        self.image_size = cfg.get("image_size", 448)

        onnx_file = _resolve_model_dir(checkpoint) / "model.onnx"
        providers = [
            ("CUDAExecutionProvider", {}),
            "CPUExecutionProvider",
        ]
        try:
            self._sess = ort.InferenceSession(str(onnx_file), providers=providers)
        except Exception:
            self._sess = ort.InferenceSession(
                str(onnx_file), providers=["CPUExecutionProvider"]
            )

        self._input_name = self._sess.get_inputs()[0].name
        self._output_name = self._sess.get_outputs()[0].name

    def run(self, pixel_values: np.ndarray) -> np.ndarray:
        """Raw logits to sigmoid scores, shape (N, num_classes)."""
        raw = self._sess.run([self._output_name], {self._input_name: pixel_values})[0]
        return 1.0 / (1.0 + np.exp(-raw))  # sigmoid

    @property
    def num_classes(self) -> int:
        return len(self.tag_to_id)

    def category_name(self, tag: str) -> str:
        return get_category_name(self.cat_map, tag)


def predict(
    image_path: str | Path,
    checkpoint: str | Path,
    top_k: int = 20,
    min_score: float = 0.0,
) -> list[tuple[str, float]]:
    """Run inference on a single image. Returns [(tag, score), ...]."""
    predictor = Predictor(checkpoint)
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image, predictor.image_size)
    scores = predictor.run(tensor)[0]
    inv = {v: k for k, v in predictor.tag_to_id.items()}
    indices = np.argsort(scores)[::-1]
    results = [(inv[int(i)], float(scores[i])) for i in indices]
    if min_score > 0.0:
        results = [(t, s) for t, s in results if s >= min_score]
    if top_k is not None:
        results = results[:top_k]
    return results


# ── Gradio UI (delegated to src.ui) ─────────────────────────────────────────


def _make_predict_fn(checkpoint: str | Path):
    """Build a predict_fn callable for src.ui.build_app."""
    predictor = Predictor(checkpoint)

    def predict_fn(image: Image.Image) -> list[tuple[str, float]]:
        tensor = preprocess(image, predictor.image_size)
        logits = predictor.run(tensor)[0]
        inv = {v: k for k, v in predictor.tag_to_id.items()}
        indices = np.argsort(logits)[::-1]
        return [(inv[int(i)], float(logits[i])) for i in indices]

    return predict_fn, predictor


# ── CLI ─────────────────────────────────────────────────────────────────────


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
    parser.add_argument("--share", action="store_true", help="Public Gradio share link")
    parser.add_argument("--host", default=None, help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Port (default: 7860)")
    args = parser.parse_args()

    if args.image:
        predictor = Predictor(args.checkpoint)
        image = Image.open(args.image).convert("RGB")
        tensor = preprocess(image, predictor.image_size)
        scores = predictor.run(tensor)[0]
        inv = {v: k for k, v in predictor.tag_to_id.items()}
        indices = np.argsort(scores)[::-1]
        results = [(inv[int(i)], float(scores[i])) for i in indices]
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
        from tools.ui import build_app

        predict_fn, predictor = _make_predict_fn(args.checkpoint)
        app = build_app(predict_fn, predictor.cat_map)
        app.launch(
            share=args.share,
            server_name=args.host,
            server_port=args.port,
        )


if __name__ == "__main__":
    main()
