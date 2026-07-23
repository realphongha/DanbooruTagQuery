import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .config import TrainConfig
from .metrics import multilabel_metrics
from .utils import device, load_json

try:
    from .deepdanbooru_model import DeepDanbooruModel
except ImportError:
    print("ERROR: deepdanbooru_model.py not found.")
    print("Download it from https://raw.githubusercontent.com/AUTOMATIC1111/TorchDeepDanbooru/master/deep_danbooru_model.py")
    print("and place it next to this script.")
    raise


def main():
    parser = argparse.ArgumentParser(description="Evaluate TorchDeepDanbooru on our validation set")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/deepdanbooru/model-resnet_custom_v3.pt"))
    parser.add_argument("--val-parquet", type=Path, default=TrainConfig().val_parquet)
    parser.add_argument("--tag-to-id", type=Path, default=TrainConfig().tag_to_id)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # Load our tag mapping
    tag_to_id = load_json(args.tag_to_id)
    our_tags = set(tag_to_id.keys())

    # Load TorchDeepDanbooru
    print(f"Loading checkpoint from {args.checkpoint} ...")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    dd_tags = state.get("tags", [])
    print(f"  {len(dd_tags)} tags")
    dd_tag_to_id = {tag: i for i, tag in enumerate(dd_tags)}

    model = DeepDanbooruModel()
    model.load_state_dict(state)
    model.eval()

    dev = device()
    model.to(dev)

    # Intersection: DeepDanbooru output index → our class index
    dd_to_our = {}
    for tag, our_idx in tag_to_id.items():
        dd_idx = dd_tag_to_id.get(tag)
        if dd_idx is not None:
            dd_to_our[dd_idx] = our_idx

    print(f"\nOur tags: {len(tag_to_id)}")
    print(f"TorchDeepDanbooru tags: {len(dd_tags)}")
    print(f"Intersection: {len(dd_to_our)}")
    missing = [t for t in our_tags if t not in dd_tag_to_id]
    if missing:
        print(f"Tags NOT in DeepDanbooru vocabulary: {', '.join(missing)}")

    # Load validation data
    print(f"\nLoading validation set from {args.val_parquet} ...")
    frame = pd.read_parquet(args.val_parquet)
    records = frame.to_dict("records")
    print(f"  {len(records)} images")

    predictions = []
    targets = []

    for start in tqdm(range(0, len(records), args.batch_size), desc="Evaluating"):
        batch = records[start : start + args.batch_size]

        batch_images = []
        batch_labels = []

        for row in batch:
            image = Image.open(row["image_path"]).convert("RGB").resize((512, 512))
            a = np.array(image, dtype=np.float32) / 255.0
            batch_images.append(a)

            labels = torch.zeros(len(tag_to_id), dtype=torch.float32)
            for tag in row["tags"]:
                idx = tag_to_id.get(tag)
                if idx is not None:
                    labels[idx] = 1.0
            batch_labels.append(labels)

        batch_np = np.stack(batch_images, axis=0)
        batch_t = torch.from_numpy(batch_np).to(dev)

        with torch.no_grad():
            dd_scores = model(batch_t).cpu().numpy()

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
    print("  TorchDeepDanbooru Evaluation Results")
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
