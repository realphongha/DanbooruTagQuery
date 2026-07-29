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
    fp16: bool = False,
    mixed: bool = False,
    keep_io_types: bool = True,
    rtol: float = 0.01,
    atol: float = 0.001,
    verbose: bool = True,
) -> Path:
    """Export a trained checkpoint to ONNX.

    Produces three files alongside *output*:
        <output>.onnx             — the ONNX model (FP32)
        <output>.tag_to_id.json   — tag name → index mapping
        <output>.config.json      — export-time config metadata

    When *fp16* is True the model is additionally converted to full FP16
    and saved as <output>.fp16.onnx.

    When *mixed* is True the model is converted via
    auto_mixed_precision (selective FP32 retention) and saved as
    <output>.mixed.onnx.

    Returns the path to the most-optimised ONNX file produced.
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
        head_type=config.head_type,
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

    result_path = onnx_path

    # ── FP16 conversion ──
    if fp16:
        try:
            import onnx
            from onnxconverter_common import float16
        except ImportError:
            sys.exit("error: --fp16 requires 'onnx' and 'onnxconverter-common' "
                      "(pip install onnx onnxconverter-common)")

        model_onnx = onnx.load(str(onnx_path))
        model_fp16 = float16.convert_float_to_float16(
            model_onnx, keep_io_types=keep_io_types
        )
        model_fp16 = onnx.shape_inference.infer_shapes(model_fp16)
        fp16_path = out.with_suffix(".fp16.onnx")
        onnx.save(model_fp16, str(fp16_path))
        result_path = fp16_path

        if verbose:
            print(f"Converted to FP16:  {fp16_path} ({fp16_path.stat().st_size / 1024:.1f} KB)")

    # ── Mixed-precision conversion ──
    if mixed:
        try:
            import onnx
            from onnxconverter_common import auto_mixed_precision
        except ImportError:
            sys.exit("error: --mixed requires 'onnx' and 'onnxconverter-common' "
                      "(pip install onnx onnxconverter-common)")

        model_onnx = onnx.load(str(onnx_path))
        feed_dict = {"pixel_values": dummy.numpy()}
        model_mixed = auto_mixed_precision.auto_convert_mixed_precision(
            model_onnx, feed_dict, rtol=rtol, atol=atol, keep_io_types=keep_io_types,
        )
        model_mixed = onnx.shape_inference.infer_shapes(model_mixed)
        mixed_path = out.with_suffix(".mixed.onnx")
        onnx.save(model_mixed, str(mixed_path))
        result_path = mixed_path

        if verbose:
            print(f"Mixed precision:    {mixed_path} ({mixed_path.stat().st_size / 1024:.1f} KB)")

    # ── save metadata ──
    tag_map_path = out.with_name(out.stem + ".tag_to_id.json")
    Path(tag_map_path).write_text(json.dumps(tag_to_id, indent=2))

    config_out = {
        "image_size": config.image_size,
        "model_name": config.model_name,
        "head_type": config.head_type,
        "num_classes": config.num_classes,
    }
    config_path = out.with_name(out.stem + ".config.json")
    Path(config_path).write_text(json.dumps(config_out, indent=2))

    if verbose:
        print(f"Tag map saved:      {tag_map_path}")
        print(f"Config saved:       {config_path}")
        extras = [fp16, mixed]
        if any(extras):
            print(f"FP32 ONNX:          {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")
        print(f"Done — {result_path} ({result_path.stat().st_size / 1024:.1f} KB)")

    return result_path


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
    parser.add_argument(
        "--fp16", action="store_true",
        help="Convert exported ONNX to FP16 via onnxconverter-common",
    )
    parser.add_argument(
        "--no-keep-io-types", action="store_false", dest="keep_io_types",
        help="Convert input/output tensors to float16 (default: keep as float32)",
    )
    parser.add_argument(
        "--mixed", action="store_true",
        help="Convert via auto_mixed_precision (selective FP32 retention; requires GPU)",
    )
    parser.add_argument(
        "--rtol", type=float, default=0.01,
        help="Relative tolerance for mixed-precision validation (default: 0.01)",
    )
    parser.add_argument(
        "--atol", type=float, default=0.001,
        help="Absolute tolerance for mixed-precision validation (default: 0.001)",
    )
    args = parser.parse_args()
    convert(args.checkpoint, args.output, args.opset,
            fp16=args.fp16, mixed=args.mixed,
            keep_io_types=args.keep_io_types,
            rtol=args.rtol, atol=args.atol)
