import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from .config import TrainConfig
from .dataset import make_loader
from .metrics import multilabel_metrics, print_metrics
from .model import ImageTagger
from .transforms import val_transforms
from .utils import device, load_json

def evaluate(checkpoint, config=TrainConfig()):
    tag_to_id = load_json(config.tag_to_id)
    config.num_classes = len(tag_to_id)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    dev = device()
    model = ImageTagger(config.model_name, config.num_classes, pretrained=False, head_type=config.head_type)
    model.load_state_dict(state["model"])
    model.to(dev).eval()
    loader = make_loader(config.val_parquet, config, val_transforms(config.image_size), False)
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in tqdm(loader):
            predictions.append(torch.sigmoid(model(batch["image"].to(dev))).cpu())
            targets.append(batch["labels"])
    preds_tensor = torch.cat(predictions).numpy()
    targets_tensor = torch.cat(targets)
    tag_counts = {tid: int(targets_tensor[:, tid].sum().item()) for tid in range(len(tag_to_id))}
    targets_np = targets_tensor.numpy()
    ap_per_tag = average_precision_score(targets_np, preds_tensor, average=None)
    tags_with_any = targets_np.any(axis=0)
    ap_dict = {tid: float(ap_per_tag[tid]) if tags_with_any[tid] else float("nan") for tid in range(len(tag_to_id))}
    metrics = multilabel_metrics(preds_tensor, targets_np)
    metrics["per_tag_ap"] = ap_dict
    return metrics, tag_to_id, tag_counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--run-name", default=None, help="Run name for output files (default: checkpoint stem)")
    args = parser.parse_args()

    run_name = args.run_name or Path(args.checkpoint).stem
    out_dir = Path("runs")
    out_dir.mkdir(exist_ok=True)

    metrics, tag_to_id, tag_counts = evaluate(args.checkpoint)
    print_metrics(metrics, tag_to_id, tag_counts=tag_counts)

    txt_path = out_dir / f"{run_name}_eval.txt"
    with open(txt_path, "w") as f:
        f.write(f"{'=' * 40}\n")
        f.write(f"  Evaluation Results\n")
        f.write(f"{'=' * 40}\n")
        f.write(f"  mAP            : {metrics['map']:.4f}\n")
        f.write(f"  Macro F1       : {metrics['macro_f1']:.4f}\n")
        f.write(f"  Best threshold : {metrics['best_threshold']:.2f}\n")
        f.write(f"  Micro F1       : {metrics['micro_f1']:.4f}\n")
        f.write(f"{'-' * 40}\n")
        f.write("  Per-tag AP:\n")
        sorted_tags = sorted(tag_to_id.keys(), key=lambda t: tag_to_id[t])
        for tag in sorted_tags:
            tid = tag_to_id[tag]
            ap = metrics.get("per_tag_ap", {}).get(tid, float("nan"))
            if not np.isnan(ap):
                count_str = f"  #{tag_counts[tid]}" if tag_counts and tid in tag_counts else ""
                f.write(f"    {tag:<30s} {ap:.4f}{count_str}\n")
        f.write(f"{'=' * 40}\n")
    print(f"\nResults saved to {txt_path}")

    csv_path = out_dir / f"{run_name}_eval.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "count", "ap"])
        sorted_tags = sorted(tag_to_id.keys(), key=lambda t: tag_to_id[t])
        for tag in sorted_tags:
            tid = tag_to_id[tag]
            ap = metrics.get("per_tag_ap", {}).get(tid, float("nan"))
            writer.writerow([tag, tag_counts.get(tid, 0), f"{ap:.6f}" if not np.isnan(ap) else ""])
    print(f"Per-tag AP saved to {csv_path}")
