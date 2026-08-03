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

# ── tag category lookup ───────────────────────────────────────────────────

HF_DATASET = "qdlabs/danbooru-tags"
API_WORKERS = 2


def _load_hf_category_map() -> dict[str, int]:
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split="train")
    return {row["name"]: row["category"] for row in ds}


def _api_tag_category(tag: str) -> int | None:
    """Fetch tag category from Danbooru API. Returns None on failure."""
    from tools.api import tag_info

    try:
        info = tag_info(tag)
        if info is not None:
            return info["category"]
    except Exception as exc:
        print(f"  API error for '{tag}': {exc}")
    return None


def _api_lookup_parallel(tags: list[str]) -> dict[str, int]:
    """Fetch categories in parallel via thread pool."""
    import concurrent.futures

    from tqdm import tqdm

    results: dict[str, int] = {}

    def fetch(tag: str) -> tuple[str, int]:
        cat = _api_tag_category(tag)
        return tag, cat if cat is not None else 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=API_WORKERS) as pool:
        futs = [pool.submit(fetch, tag) for tag in tags]
        for fut in tqdm(
            concurrent.futures.as_completed(futs),
            total=len(futs),
            desc="API",
        ):
            tag, cat = fut.result()
            results[tag] = cat
    return results


def _build_tag_category(
    tag_to_id: dict[str, int], out_dir: Path, verbose: bool = True
) -> Path:
    """Fill category for every tag; write tag_category.json into out_dir.

    Sources, in order: HF dataset (qdlabs/danbooru-tags), then Danbooru API
    fallback (missing/failed default to general). Requires `datasets`,
    `tqdm`, and optionally `tools.api` (curl_cffi).
    """
    tags = sorted(tag_to_id.keys(), key=lambda t: tag_to_id[t])
    json_path = out_dir / "tag_category.json"

    existing: dict[str, int] = {}
    if json_path.exists():
        existing = json.loads(json_path.read_text())

    missing = [t for t in tags if t not in existing]
    print(f"Categories: {len(tags)} total, {len(existing)} cached, {len(missing)} missing")

    lookup: dict[str, int] = existing
    if missing:
        print(f"Loading category map from {HF_DATASET} …")
        cat_map = _load_hf_category_map()
        found = sum(1 for t in missing if t in cat_map)
        print(f"  {found}/{len(missing)} found in dataset")

        api_needed: list[str] = []
        for tag in missing:
            cat = cat_map.get(tag)
            if cat is not None:
                lookup[tag] = cat
            else:
                api_needed.append(tag)

        if api_needed:
            print(f"  {len(api_needed)} not in dataset, fetching from Danbooru API "
                  f"({API_WORKERS} workers) …")
            lookup.update(_api_lookup_parallel(api_needed))

    json_path.write_text(json.dumps(lookup, indent=0) + "\n")
    if verbose:
        print(f"Category map saved: {json_path} ({len(lookup)} tags)")
    return json_path


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
            tag_category.json     — tag → category mapping (HF dataset + API)

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
    # The deploy Predictor expects a 4-file model dir; fill categories for
    # every tag so exports are self-contained (no manual hub upload needed).
    if verbose:
        print("Generating tag_category.json …")
    try:
        _build_tag_category(tag_to_id, model_dir, verbose=verbose)
    except Exception as exc:
        if verbose:
            print(f"Warning: could not build tag categories: {exc}")
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
