#!/usr/bin/env python3
"""Model comparison arena.

Two-phase design:
  1. Phase 1 — load each model (CPU only) → collect tag lists → compute intersection.
  2. Phase 2 — for each model: load → evaluate all images → unload → compute metric.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .config import TrainConfig
from .metrics import multilabel_metrics, print_metrics
from .utils import load_json

from .compared_models import list_available_models


def _build_model(name: str, args) -> object | None:
    """Instantiate and load one model; return None on failure."""
    registry = list_available_models()
    cls = registry.get(name)
    if cls is None:
        print(f"  [SKIP] '{name}' not available (missing dependencies)")
        return None

    kwargs = {}
    if name == "DeepDanbooru":
        kwargs["checkpoint"] = args.deepdanbooru_checkpoint
    elif name == "JoyTag":
        kwargs["model_dir"] = args.joytag_model_dir
    elif name == "WD-SwinV2":
        kwargs["model_name"] = args.wd_model
    elif name == "ML-Danbooru":
        kwargs["repo_dir"] = args.mldanbooru_repo_dir

    try:
        model = cls(**kwargs)
        model.load()
        return model
    except Exception as e:
        print(f"  [ERROR] Failed to load '{name}': {e}")
        return None


def _free_model(model) -> None:
    """Unload model from GPU and free memory."""
    if model is not None:
        model._model = None
        if model.device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Model comparison arena")
    parser.add_argument("--val-parquet", type=Path, default="./data/danbooru_after2024_testset.parquet")
    parser.add_argument("--tag-to-id", type=Path, default="./data/tag_to_id_after2024_testset.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--deepdanbooru-checkpoint", type=Path,
                        default=Path("models/deepdanbooru/model-resnet_custom_v3.pt"))
    parser.add_argument("--joytag-model-dir", type=Path, default=Path("models/joytag"))
    parser.add_argument("--wd-model", type=str, default="swinv2", choices=["vit", "swinv2", "convnext"])
    parser.add_argument("--mldanbooru-repo-dir", type=Path, default=Path("models/ml-danbooru"))
    parser.add_argument("--mldanbooru-checkpoint", type=str, default=None,
                        help="Checkpoint filename in repo_dir")
    parser.add_argument("--mldanbooru-model-name", type=str, default="caformer_m36")
    parser.add_argument("--mldanbooru-fp16", action="store_true", default=False)
    parser.add_argument("--skip-models", type=str, nargs="*", default=[])
    args = parser.parse_args()

    dev_type = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev_type}")
    print(f"Validation: {args.val_parquet}")

    # ── 1. Our vocabulary ──────────────────────────────────────────
    tag_to_id = load_json(args.tag_to_id)
    our_tags = set(tag_to_id.keys())
    print(f"Our vocabulary: {len(tag_to_id)} tags\n")

    # ── 2. Decide models ───────────────────────────────────────────
    available = list_available_models()
    model_names = [n for n in available if n not in args.skip_models]
    if not model_names:
        print("No models selected. Exiting.")
        return
    print(f"Models: {', '.join(model_names)}")

    # ── 3. Load validation images ──────────────────────────────────
    print(f"Loading validation set from {args.val_parquet} ...")
    frame = pd.read_parquet(args.val_parquet)
    records = frame.to_dict("records")
    pil_images = [Image.open(row["image_path"]).convert("RGB") for row in records]
    print(f"  {len(records)} images\n")

    # ── 4. Phase 1: load each model on CPU → collect tag lists ─────
    model_tag_lists: dict[str, list[str]] = {}
    for name in model_names:
        model = _build_model(name, args)
        if model is None:
            continue
        model_tag_lists[name] = list(model.tag_names)
        _free_model(model)

    if not model_tag_lists:
        print("No models loaded. Exiting.")
        return

    # Intersection: ours ∩ all models
    intersection = our_tags
    for tag_list in model_tag_lists.values():
        intersection = intersection & set(tag_list)
    intersection_tags = sorted(intersection)
    n_int = len(intersection_tags)
    print(f"\nIntersection tags (ours ∩ all models): {n_int}")

    shared_index = {tag: i for i, tag in enumerate(intersection_tags)}
    tag_to_id_subset = {tag: idx for idx, tag in enumerate(intersection_tags)}

    # Per-model: model output index → shared index
    model_to_shared: dict[str, dict[int, int]] = {}
    for name, tag_list in model_tag_lists.items():
        mapping = {}
        for shared_idx, tag in enumerate(intersection_tags):
            try:
                mapping[tag_list.index(tag)] = shared_idx
            except ValueError:
                pass
        model_to_shared[name] = mapping

    # Ground truth targets (same for all models)
    targets_list: list[torch.Tensor] = []
    for row in records:
        t = torch.zeros(n_int, dtype=torch.float32)
        for tag in row["tags"]:
            idx = shared_index.get(tag)
            if idx is not None:
                t[idx] = 1.0
        targets_list.append(t)
    targets_tensor = torch.stack(targets_list)
    tag_counts = {i: int(targets_tensor[:, i].sum().item()) for i in range(n_int)}

    # ── 5. Phase 2: evaluate one model at a time ───────────────────
    all_metrics: dict[str, dict] = {}

    for name in model_names:
        if name not in model_tag_lists:
            continue

        print(f"\n{'=' * 50}")
        print(f"  Evaluating {name}")
        print(f"{'=' * 50}")

        model = _build_model(name, args)
        if model is None:
            continue

        mapping = model_to_shared[name]
        preds_list: list[torch.Tensor] = []

        for start in tqdm(range(0, len(records), args.batch_size), desc=f"  {name}"):
            end = start + args.batch_size
            for idx in range(start, end):
                scores = model.predict(pil_images[idx])
                shared = torch.zeros(n_int, dtype=torch.float32)
                for model_idx, shared_idx in mapping.items():
                    shared[shared_idx] = float(scores[model_idx])
                preds_list.append(shared)

        preds_tensor = torch.stack(preds_list)
        metrics = multilabel_metrics(preds_tensor, targets_tensor)
        all_metrics[name] = metrics
        print_metrics(metrics, tag_to_id_subset,
                      f"{name} — Intersection ({n_int} tags)",
                      tag_counts=tag_counts)

        _free_model(model)

    # ── 6. Comparison table ────────────────────────────────────────
    if all_metrics:
        print(f"\n{'=' * 60}")
        print(f"  COMPARISON TABLE — Intersection ({n_int} tags)")
        print(f"{'=' * 60}")
        print(f"  {'Model':<20s} {'mAP':>8s} {'Macro F1':>10s} {'Micro F1':>10s} {'Thresh':>8s}")
        print(f"  {'-' * 20} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 8}")
        for name in all_metrics:
            m = all_metrics[name]
            print(f"  {name:<20s} {m['map']:>8.4f} {m['macro_f1']:>10.4f} "
                  f"{m['micro_f1']:>10.4f} {m['best_threshold']:>8.2f}")
        print(f"{'=' * 60}")

        best = max(all_metrics.keys(), key=lambda n: all_metrics[n]["map"])
        print(f"\nBest model by mAP: {best} ({all_metrics[best]['map']:.4f})")

    # ── 7. Optional CSV export ─────────────────────────────────────
    if args.output_csv and all_metrics:
        rows = []
        for tag in intersection_tags:
            tid = tag_to_id_subset[tag]
            row = {"tag": tag, "count": int(tag_counts[tid])}
            for name in all_metrics:
                ap = all_metrics[name].get("per_tag_ap", {}).get(tid, float("nan"))
                row[f"{name}_AP"] = ap
            rows.append(row)
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)
        print(f"\nPer-tag AP saved to {args.output_csv}")

    return all_metrics


if __name__ == "__main__":
    main()
