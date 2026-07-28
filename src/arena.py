#!/usr/bin/env python3
"""Model comparison arena.

Two-phase design:
  1. Phase 1 — load each model (CPU only) → collect tag lists → compute intersection.
  2. Phase 2 — for each model: load → evaluate all images → unload → compute metric.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import TrainConfig
from .metrics import multilabel_metrics, print_metrics
from .model import ImageTagger
from .transforms import val_transforms
from .utils import load_json

from .compared_models import BaseModel, list_available_models


class OursModel(BaseModel):
    """Wrapper for our ImageTagger model."""

    def __init__(self, checkpoint: Path):
        self._checkpoint = checkpoint
        self._tags: list[str] = []
        self._model = None
        self._config = TrainConfig()

    @property
    def name(self) -> str:
        return "Ours"

    @property
    def tag_names(self) -> list[str]:
        return self._tags

    @property
    def input_size(self) -> tuple[int, int]:
        s = self._config.image_size
        return (s, s)

    def load(self) -> None:
        state = torch.load(self._checkpoint, map_location="cpu", weights_only=False)
        self._tags = list(state["tag_to_id"].keys())
        self._config.num_classes = len(self._tags)
        model = ImageTagger(
            self._config.model_name,
            self._config.num_classes,
            pretrained=False,
            head_type=self._config.head_type,
        )
        model.load_state_dict(state["model_ema"])
        model.eval()
        self._model = model
        print(f"    {len(self._tags)} tags")

    def predict(self, image: Image.Image) -> np.ndarray:
        tensor = val_transforms(self._config.image_size)(image.convert("RGB")).unsqueeze(0)
        dev = self.device
        if next(self._model.parameters()).device != dev:
            self._model.to(dev)
        tensor = tensor.to(dev)
        with torch.no_grad():
            scores = torch.sigmoid(self._model(tensor)).cpu().numpy()
        return scores[0].astype(np.float32)


def _build_model(name: str, args) -> object | None:
    """Instantiate and load one model; return None on failure."""
    if name == "Ours":
        assert args.ours is not None and args.ours.exists()
        try:
            model = OursModel(args.ours)
            model.load()
            return model
        except Exception as e:
            print(f"  [ERROR] Failed to load 'Ours': {e}")
            return None

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


def _fmt_params(n: int | None) -> str:
    if n is None:
        return "N/A"
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    return f"{n / 1e6:.1f}M"

def _fmt_input(size: tuple[int, int] | None) -> str:
    if size is None:
        return "N/A"
    return f"{size[0]}x{size[1]}"

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
    parser.add_argument("--wd-model", type=str, default="swinv2",
                        choices=["vit", "swinv2", "convnext", "eva02-large"])
    parser.add_argument("--mldanbooru-repo-dir", type=Path, default=Path("models/ml-danbooru"))
    parser.add_argument("--mldanbooru-checkpoint", type=str, default=None,
                        help="Checkpoint filename in repo_dir")
    parser.add_argument("--mldanbooru-model-name", type=str, default="caformer_m36")
    parser.add_argument("--mldanbooru-fp16", action="store_true", default=False)
    parser.add_argument("--ours", type=Path, default=None,
                        help="Path to our model checkpoint to evaluate alongside others")
    parser.add_argument("--skip-models", type=str, nargs="*", default=[],
                        help="Skip models by name, separated by spaces. "
                             "Model names: DeepDanbooru, JoyTag, WD-SwinV2, ML-Danbooru, Ours")
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
    if args.ours is not None:
        if not args.ours.exists():
            print(f"[ERROR] Our checkpoint not found: {args.ours}")
            return
        if "Ours" not in args.skip_models:
            model_names = ["Ours"] + model_names
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
    model_stats: dict[str, dict] = {}

    for name in model_names:
        if name not in model_tag_lists:
            continue

        print(f"\n{'=' * 50}")
        print(f"  Evaluating {name}")
        print(f"{'=' * 50}")

        model = _build_model(name, args)
        if model is None:
            continue

        # Warmup: one predict call to avoid cold-start CUDA overhead
        _ = model.predict(pil_images[0])

        mapping = model_to_shared[name]
        preds_list: list[torch.Tensor] = []
        latencies: list[float] = []

        for start in tqdm(range(0, len(records), args.batch_size), desc=f"  {name}"):
            end = start + args.batch_size
            for idx in range(start, end):
                t0 = time.perf_counter()
                scores = model.predict(pil_images[idx])
                latencies.append(time.perf_counter() - t0)
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

        param_count = model.param_count
        avg_latency_ms = (sum(latencies) / len(latencies)) * 1000 if latencies else 0.0
        model_stats[name] = {
            "param_count": param_count,
            "avg_latency_ms": avg_latency_ms,
            "input_size": model.input_size,
        }
        print(f"    Params: {_fmt_params(param_count)}  |  "
              f"Input: {_fmt_input(model.input_size)}  |  "
              f"Latency: {avg_latency_ms:.2f} ms/image")

        _free_model(model)

    # ── 6. Comparison table ────────────────────────────────────────
    if all_metrics:
        print(f"\n{'=' * 90}")
        print(f"  COMPARISON TABLE — Intersection ({n_int} tags)")
        print(f"{'=' * 90}")
        print(f"  {'Model':<20s} {'Params':>9s} {'Input':>8s} {'Latency':>10s} "
              f"{'mAP':>8s} {'Macro F1':>10s} {'Micro F1':>10s} {'Thresh':>8s}")
        print(f"  {'-' * 20} {'-' * 9} {'-' * 8} {'-' * 10} "
              f"{'-' * 8} {'-' * 10} {'-' * 10} {'-' * 8}")
        for name in all_metrics:
            m = all_metrics[name]
            s = model_stats.get(name, {})
            params = _fmt_params(s.get("param_count"))
            inp = _fmt_input(s.get("input_size"))
            lat = f"{s.get('avg_latency_ms', 0):.1f}ms" if s.get("avg_latency_ms") else "N/A"
            print(f"  {name:<20s} {params:>9s} {inp:>8s} {lat:>10s} "
                  f"{m['map']:>8.4f} {m['macro_f1']:>10.4f} "
                  f"{m['micro_f1']:>10.4f} {m['best_threshold']:>8.2f}")
        print(f"{'=' * 90}")

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

        # Also save model-level stats
        stats_path = args.output_csv.with_name(args.output_csv.stem + "_stats.csv")
        stats_rows = []
        for name in all_metrics:
            m = all_metrics[name]
            s = model_stats.get(name, {})
            stats_rows.append({
                "model": name,
                "params": s.get("param_count"),
                "input_size": _fmt_input(s.get("input_size")),
                "avg_latency_ms": f"{s.get('avg_latency_ms', 0):.2f}",
                "map": m["map"],
                "macro_f1": m["macro_f1"],
                "micro_f1": m["micro_f1"],
                "best_threshold": m["best_threshold"],
            })
        pd.DataFrame(stats_rows).to_csv(stats_path, index=False)
        print(f"Model-level stats saved to {stats_path}")

    return all_metrics


if __name__ == "__main__":
    main()
