import argparse
import csv
from pathlib import Path

import numpy as np
import torch
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
    model.load_state_dict(state["model_ema"])
    model.to(dev).eval()
    loader = make_loader(config.val_parquet, config, val_transforms(config.image_size), False)
    N = len(loader.dataset)
    preds = torch.empty(N, config.num_classes, dtype=torch.float32)
    targets = torch.empty(N, config.num_classes, dtype=torch.float32)
    offset = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            B = batch["image"].size(0)
            preds[offset:offset+B] = torch.sigmoid(model(batch["image"].to(dev))).cpu()
            targets[offset:offset+B] = batch["labels"]
            offset += B

    tag_counts = {tid: int(targets[:, tid].sum().item()) for tid in range(len(tag_to_id))}
    metrics = multilabel_metrics(preds, targets)
    del preds, targets
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
