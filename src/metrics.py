import numpy as np
from sklearn.metrics import average_precision_score, f1_score


def multilabel_metrics(probabilities, targets):
    probabilities = np.asarray(probabilities); targets = np.asarray(targets).astype(int)
    per_tag = {index: float(average_precision_score(targets[:, index], probabilities[:, index])) if targets[:, index].any() else float("nan") for index in range(targets.shape[1])}
    map_score = float(np.nanmean(list(per_tag.values())))
    best_f1, best_threshold = 0.0, 0.5
    for threshold in np.arange(0.1, 1.0, 0.05):
        f1 = f1_score(targets, probabilities >= threshold, average="macro", zero_division=0)
        if f1 > best_f1: best_f1, best_threshold = f1, threshold
    return {"map": map_score, "macro_f1": best_f1, "best_threshold": best_threshold, "micro_f1": float(f1_score(targets, probabilities >= best_threshold, average="micro", zero_division=0)), "per_tag_ap": per_tag}


def print_metrics(metrics, tag_to_id, title="Evaluation Results", extra=None, show_per_tag=True):
    print(f"\n{'=' * 40}")
    print(f"  {title}")
    print(f"{'=' * 40}")
    print(f"  mAP            : {metrics['map']:.4f}")
    print(f"  Macro F1       : {metrics['macro_f1']:.4f}")
    print(f"  Best threshold : {metrics['best_threshold']:.2f}")
    print(f"  Micro F1       : {metrics['micro_f1']:.4f}")
    if extra:
        print(f"{'-' * 40}")
        for key, value in extra.items():
            if isinstance(value, float):
                print(f"  {key:<14s} : {value:.6f}")
            else:
                print(f"  {key:<14s} : {value}")
    if show_per_tag:
        print(f"{'-' * 40}")
        print("  Per-tag AP:")
        sorted_tags = sorted(tag_to_id.keys(), key=lambda t: tag_to_id[t])
        for tag in sorted_tags:
            tid = tag_to_id[tag]
            ap = metrics["per_tag_ap"].get(tid, float("nan"))
            if not np.isnan(ap):
                print(f"    {tag:<30s} {ap:.4f}")
    print(f"{'=' * 40}")
