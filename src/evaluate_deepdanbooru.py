import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    import deepdanbooru as dd
    import tensorflow as tf
except ImportError as e:
    print(f"ERROR: {e}")
    print("Install DeepDanbooru and TensorFlow:")
    print("  pip install deepdanbooru[tensorflow]")
    sys.exit(1)

from .config import TrainConfig
from .metrics import multilabel_metrics
from .utils import load_json


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepDanbooru on our validation set")
    parser.add_argument("--model-path", type=Path, default=Path("models/deepdanbooru/model-resnet_custom_v2.keras"))
    parser.add_argument("--tags-path", type=Path, default=Path("models/deepdanbooru/tags.txt"))
    parser.add_argument("--val-parquet", type=Path, default=TrainConfig().val_parquet)
    parser.add_argument("--tag-to-id", type=Path, default=TrainConfig().tag_to_id)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference for TensorFlow")
    args = parser.parse_args()

    if args.cpu:
        tf.config.set_visible_devices([], "GPU")

    # Load our tag mapping
    tag_to_id = load_json(args.tag_to_id)
    our_tags = set(tag_to_id.keys())

    # Load DeepDanbooru
    print(f"Loading model from {args.model_path} ...")
    model = tf.keras.models.load_model(str(args.model_path), compile=False)
    _, h, w, _ = model.input_shape
    print(f"  Input: {w}x{h}")
    print(f"  Output: {model.output_shape[-1]} tags")

    print(f"Loading tags from {args.tags_path} ...")
    dd_tags = dd.data.load_tags(str(args.tags_path))
    print(f"  {len(dd_tags)} tags")
    dd_tag_to_id = {tag: i for i, tag in enumerate(dd_tags)}

    # Intersection mapping: DeepDanbooru output index → our class index
    dd_to_our = {}
    for tag, our_idx in tag_to_id.items():
        dd_idx = dd_tag_to_id.get(tag)
        if dd_idx is not None:
            dd_to_our[dd_idx] = our_idx

    print(f"\nOur tags: {len(tag_to_id)}")
    print(f"DeepDanbooru tags: {len(dd_tags)}")
    print(f"Intersection: {len(dd_to_our)}")
    missing = [t for t in our_tags if t not in dd_tag_to_id]
    if missing:
        print(f"Tags NOT in DeepDanbooru vocabulary: {', '.join(missing)}")

    # Load validation data
    print(f"\nLoading validation set from {args.val_parquet} ...")
    frame = pd.read_parquet(args.val_parquet)
    records = frame.to_dict("records")
    print(f"  {len(records)} images")

    # Evaluate in batches
    predictions = []
    targets = []

    for start in tqdm(range(0, len(records), args.batch_size), desc="Evaluating"):
        batch = records[start : start + args.batch_size]

        batch_images = []
        batch_labels = []

        for row in batch:
            image = dd.data.load_image_for_evaluate(
                row["image_path"], width=w, height=h
            )
            batch_images.append(image)

            labels = torch.zeros(len(tag_to_id), dtype=torch.float32)
            for tag in row["tags"]:
                idx = tag_to_id.get(tag)
                if idx is not None:
                    labels[idx] = 1.0
            batch_labels.append(labels)

        batch_np = np.stack(batch_images, axis=0)
        dd_scores = model.predict(batch_np, verbose=0)

        for b in range(len(batch)):
            our_scores = torch.zeros(len(tag_to_id), dtype=torch.float32)
            for dd_idx, our_idx in dd_to_our.items():
                our_scores[our_idx] = float(dd_scores[b, dd_idx])
            predictions.append(our_scores)
            targets.append(batch_labels[b])

    preds_tensor = torch.stack(predictions)
    targets_tensor = torch.stack(targets)
    metrics = multilabel_metrics(preds_tensor, targets_tensor)

    print("\n========================================")
    print("  DeepDanbooru Evaluation Results")
    print("========================================")
    print(f"  mAP            : {metrics['map']:.4f}")
    print(f"  Macro F1       : {metrics['macro_f1']:.4f}")
    print(f"  Best threshold : {metrics['best_threshold']:.2f}")
    print(f"  Micro F1       : {metrics['micro_f1']:.4f}")
    print("----------------------------------------")
    print("  Per-tag AP:")
    sorted_tags = sorted(tag_to_id.keys(), key=lambda t: tag_to_id[t])
    for tag in sorted_tags:
        tid = tag_to_id[tag]
        ap = metrics["per_tag_ap"].get(tid, float("nan"))
        marker = " *" if tid not in dd_to_our.values() else ""
        if not np.isnan(ap):
            print(f"    {tag:<30s} {ap:.4f}{marker}")
    print("========================================")

    return metrics


if __name__ == "__main__":
    main()
