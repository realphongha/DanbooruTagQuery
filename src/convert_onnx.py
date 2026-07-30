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

    Produces three files alongside *output*:
        <output>.onnx             — the ONNX model (FP32)
        <output>.tag_to_id.json   — tag name → index mapping
        <output>.config.json      — export-time config metadata

    Returns the path to the produced ONNX file.
    """
    chk = Path(checkpoint)
    if output is None:
        output = str(chk.with_suffix(""))
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

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

    # ── build model & load weights ──
    model = ImageTagger(
        config.model_name,
        config.num_classes,
        pretrained=False,
    )
    model.load_state_dict(weights)
    model.eval()

    # ── export FP32 ONNX ──
    dummy = torch.randn(1, 3, config.image_size, config.image_size)
    onnx_path = out.with_suffix(".onnx")

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
    tag_map_path = out.with_name(out.stem + ".tag_to_id.json")
    Path(tag_map_path).write_text(json.dumps(tag_to_id, indent=2))

    config_out = {
        "image_size": config.image_size,
        "model_name": config.model_name,
        "num_classes": config.num_classes,
    }
    config_path = out.with_name(out.stem + ".config.json")
    Path(config_path).write_text(json.dumps(config_out, indent=2))

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
