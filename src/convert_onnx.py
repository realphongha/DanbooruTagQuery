"""
Convert a PyTorch checkpoint to ONNX for fast CPU/GPU inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from .config import TrainConfig
from .model import ImageTagger


def convert(
    checkpoint: str,
    output: str | None = None,
    opset: int = 18,
    verbose: bool = True,
) -> Path:
    """Export a trained checkpoint to FP32 ONNX.

    Produces a model directory with four files:
        <output>/
            model.onnx            — the ONNX model (FP32)
            tag_to_id.json        — tag name → index mapping
            config.json           — export-time config metadata
            tag_category.json     — tag → category mapping (via prebuild_cache)

    Returns the path to the produced ONNX file.
    """
    chk = Path(checkpoint)
    if output is None:
        output = str(chk.with_suffix(""))
    model_dir = Path(output)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── load checkpoint ──
    if verbose:
        print(f"Loading checkpoint: {chk}")
    state = torch.load(str(chk), map_location="cpu", weights_only=False)

    tag_to_id: dict[str, int] = state.get("tag_to_id")
    if tag_to_id is None:
        sys.exit("error: checkpoint missing 'tag_to_id'")

    # model weights — prefer "model_ema" over "model"
    weights = state.get("model_ema") or state.get("model")
    if weights is None:
        sys.exit("error: checkpoint has neither 'model' nor 'model_ema' keys")

    config = TrainConfig()
    config.num_classes = len(tag_to_id)

    # Build from the checkpoint's own saved config so non-default architectures
    # (e.g. B/16 students, projector models) export correctly instead of using
    # TrainConfig() defaults.  Falls back to defaults for legacy checkpoints.
    ckpt_cfg = state.get("config", {})
    model_name = ckpt_cfg.get("model_name", config.model_name)
    image_size = ckpt_cfg.get("image_size", config.image_size)
    projector = ckpt_cfg.get("projector") or ""
    head_embed = None
    if projector:
        parts = str(projector).split(":")
        if len(parts) == 2:
            head_embed = int(parts[1])
        else:
            sys.exit(f"error: invalid projector field in checkpoint config: {projector!r}")

    # ── build model & load weights ──
    model = ImageTagger(
        model_name,
        config.num_classes,
        pretrained=False,
        head_embed_dim=head_embed,
    )
    model.load_state_dict(weights)
    model.eval()

    # ── export FP32 ONNX ──
    dummy = torch.randn(1, 3, image_size, image_size)
    onnx_path = model_dir / "model.onnx"

    if verbose:
        print(f"Exporting to ONNX (opset {opset}): {onnx_path}")

    export_kwargs = dict(
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=opset,
        do_constant_folding=False,
    )
    try:
        # dynamo=False avoids torch.export strict tracing issues (torch ≥2.13)
        torch.onnx.export(model, dummy, str(onnx_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(onnx_path), **export_kwargs)

    # ── save metadata ──
    tag_map_path = model_dir / "tag_to_id.json"
    tag_map_path.write_text(json.dumps(tag_to_id, indent=2))

    config_out = {
        "image_size": image_size,
        "model_name": model_name,
        "num_classes": config.num_classes,
        "projector": projector,
    }
    config_path = model_dir / "config.json"
    config_path.write_text(json.dumps(config_out, indent=2))

    # ── tag_category.json (HF dataset + Danbooru API fallback) ────────────
    # The deploy Packages expect a 4-file model dir; prebuild_cache fills
    # categories for every tag in the exported tag map, so exports are
    # self-contained (no manual hub upload needed).
    try:
        from tools.prebuild_cache import prebuild
        if verbose:
            print("Generating tag_category.json …")
        prebuild(tag_map_path)
    except Exception as exc:
        if verbose:
            print(f"Warning: could not prebuild tag categories: {exc}")
        print("  (tag_to_id.json + config.json still written; "
              "tag_category.json is required by the deploy Predictor)")

    if verbose:
        print(f"Tag map saved:      {tag_map_path}")
        print(f"Config saved:       {config_path}")
        print(f"Done — {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")

    return onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert checkpoint to ONNX")
    parser.add_argument("checkpoint", help="Path to .pt checkpoint")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output stem (default: checkpoint stem without suffix)",
    )
    parser.add_argument(
        "--opset", type=int, default=18,
        help="ONNX opset version (default: 18)",
    )
    args = parser.parse_args()
    convert(args.checkpoint, args.output, args.opset)
