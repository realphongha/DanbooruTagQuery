import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .config import TrainConfig
from .metrics import multilabel_metrics, print_metrics
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
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
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

    # Build DD-space ground truth indices: intersection DD tag indices
    intersection_tag_names = {tag for tag in our_tags if tag in dd_tag_to_id}
    intersection_dd_indices = sorted(dd_tag_to_id[tag] for tag in intersection_tag_names)
    dd_tag_names_for_intersection = [dd_tags[i] for i in intersection_dd_indices]
    print(f"Intersection tags for DD-space eval: {len(intersection_dd_indices)}")

    predictions = []
    targets = []
    dd_predictions = []  # scores in DD tag space (intersection only)
    dd_targets = []      # labels in DD tag space

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

            # DD-space: scores and targets for intersection tags
            dd_score_vec = torch.zeros(len(intersection_dd_indices), dtype=torch.float32)
            dd_target_vec = torch.zeros(len(intersection_dd_indices), dtype=torch.float32)
            for j, dd_idx in enumerate(intersection_dd_indices):
                dd_score_vec[j] = float(dd_scores[b, dd_idx])
                tag_name = dd_tags[dd_idx]
                our_idx = tag_to_id.get(tag_name)
                if our_idx is not None and batch_labels[b][our_idx] > 0.5:
                    dd_target_vec[j] = 1.0
            dd_predictions.append(dd_score_vec)
            dd_targets.append(dd_target_vec)

    preds_tensor = torch.stack(predictions)
    targets_tensor = torch.stack(targets)
    tag_counts = {tid: int(targets_tensor[:, tid].sum().item()) for tid in range(len(tag_to_id))}
    metrics = multilabel_metrics(preds_tensor, targets_tensor)
    print_metrics(metrics, tag_to_id, "TorchDeepDanbooru Evaluation Results (Our Tag Space)", tag_counts=tag_counts)

    # DD-space metrics on intersection tags
    dd_preds_tensor = torch.stack(dd_predictions)
    dd_targets_tensor = torch.stack(dd_targets)
    dd_tag_to_id_subset = {name: i for i, name in enumerate(dd_tag_names_for_intersection)}
    dd_tag_counts = {i: int(dd_targets_tensor[:, i].sum().item()) for i in range(len(dd_tag_names_for_intersection))}
    dd_metrics = multilabel_metrics(dd_preds_tensor, dd_targets_tensor)
    print_metrics(dd_metrics, dd_tag_to_id_subset, "TorchDeepDanbooru Evaluation Results (DD Tag Space, Intersection Only)", tag_counts=dd_tag_counts)
    return metrics


if __name__ == "__main__":
    main()
