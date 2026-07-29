import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import TrainConfig
from .dataset import make_loader
from .metrics import multilabel_metrics, print_metrics
from .onnx_utils import Predictor, load_tag_to_id, load_config, is_onnx
from .transforms import val_transforms
from .utils import device, load_json

def evaluate(checkpoint, config_override=None):
    predictor = Predictor(checkpoint)
    tag_to_id = predictor.tag_to_id
    cfg = predictor.config
    if config_override:
        for k, v in config_override.__dict__.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    loader = make_loader(cfg.val_parquet, cfg, val_transforms(cfg.image_size), False)
    N = len(loader.dataset)
    num_classes = len(tag_to_id)
    preds = np.empty((N, num_classes), dtype=np.float32)
    targets_np = np.empty((N, num_classes), dtype=np.float32)
    offset = 0

    for batch in tqdm(loader):
        B = batch["image"].size(0)
        preds[offset : offset + B] = predictor.run(batch["image"])
        targets_np[offset : offset + B] = batch["labels"].numpy()
        offset += B

    tag_counts = {tid: int(targets_np[:, tid].sum()) for tid in range(num_classes)}
    metrics = multilabel_metrics(torch.from_numpy(preds), torch.from_numpy(targets_np))
    return metrics, tag_to_id, tag_counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--run-name", default=None, help="Run name for output files (default: checkpoint stem)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for dataloader")
    args = parser.parse_args()

    run_name = args.run_name or Path(args.checkpoint).stem
    out_dir = Path("runs")
    out_dir.mkdir(exist_ok=True)

    config_override = argparse.Namespace(batch_size=args.batch_size) if args.batch_size else None
    metrics, tag_to_id, tag_counts = evaluate(args.checkpoint, config_override=config_override)
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
