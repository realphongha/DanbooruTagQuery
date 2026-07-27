import numpy as np
import torch

_BATCH_SIZE = 500


def _torch_ap_batched(scores, targets, batch_size=_BATCH_SIZE):
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU AP calculation")
        return _cpu_ap_fallback(scores.numpy(), targets.numpy(), batch_size)
    dev = torch.device("cuda")
    n_tags = targets.shape[1]
    ap = torch.full((n_tags,), float("nan"), device=dev, dtype=torch.float32)
    for start in range(0, n_tags, batch_size):
        end = min(start + batch_size, n_tags)
        b_scores = scores[:, start:end].contiguous().to(dev)
        b_targets = targets[:, start:end].contiguous().to(dev)

        _, indices = b_scores.sort(dim=0, descending=True)
        s_targets = torch.gather(b_targets, 0, indices)
        del indices

        cumsum = s_targets.cumsum(dim=0)
        positions = torch.arange(1, s_targets.size(0) + 1, device=dev, dtype=torch.float32).unsqueeze(1)
        precision = cumsum / positions
        del cumsum, positions

        pos_mask = s_targets == 1.0
        del s_targets
        n_pos = pos_mask.sum(dim=0).float()
        prec_at_pos = (precision * pos_mask.float()).sum(dim=0)
        del precision, pos_mask

        has_pos = n_pos > 0
        ap[start:end] = torch.where(has_pos, prec_at_pos / n_pos, torch.tensor(float("nan"), device=dev))
        del b_scores, b_targets, n_pos, has_pos, prec_at_pos
    return ap.cpu().numpy().astype(np.float32)


def _cpu_ap_fallback(scores_np, targets_np, batch_size=_BATCH_SIZE):
    from sklearn.metrics import average_precision_score as sk_aps
    n_tags = scores_np.shape[1]
    ap = np.full(n_tags, np.nan, dtype=np.float32)
    tags_with_any = targets_np.any(axis=0)
    for start in range(0, n_tags, batch_size):
        end = min(start + batch_size, n_tags)
        mask = tags_with_any[start:end]
        if mask.any():
            ap[start:end][mask] = sk_aps(
                targets_np[:, start:end][:, mask],
                scores_np[:, start:end][:, mask],
                average=None,
            ).astype(np.float32)
    return ap


def _f1_at_threshold(probs_np, targets_np, threshold, batch_size=_BATCH_SIZE):
    n_tags = probs_np.shape[1]
    all_tp = np.zeros(n_tags, dtype=np.float64)
    all_fp = np.zeros(n_tags, dtype=np.float64)
    all_fn = np.zeros(n_tags, dtype=np.float64)
    for start in range(0, n_tags, batch_size):
        end = min(start + batch_size, n_tags)
        pred = probs_np[:, start:end] >= threshold
        true = targets_np[:, start:end].astype(bool)
        all_tp[start:end] = (pred & true).sum(axis=0)
        all_fp[start:end] = (pred & ~true).sum(axis=0)
        all_fn[start:end] = (~pred & true).sum(axis=0)
    return all_tp, all_fp, all_fn


def _f1_from_counts(tp, fp, fn):
    denom = 2 * tp + fp + fn
    per_class = np.where(denom > 0, 2 * tp / denom, 0.0)
    macro = float(per_class.mean())
    total_tp = tp.sum()
    total_fp = fp.sum()
    total_fn = fn.sum()
    micro_denom = 2 * total_tp + total_fp + total_fn
    micro = float(2 * total_tp / micro_denom) if micro_denom > 0 else 0.0
    return macro, micro


def multilabel_metrics(probabilities, targets):
    if torch.is_tensor(probabilities):
        probs_np = probabilities.float().numpy()
        targets_np = targets.float().numpy()
    else:
        probs_np = np.asarray(probabilities, dtype=np.float32)
        targets_np = np.asarray(targets, dtype=np.float32)

    n_tags = targets_np.shape[1]

    ap_per_tag = _torch_ap_batched(
        torch.from_numpy(probs_np), torch.from_numpy(targets_np)
    )

    map_score = float(np.nanmean(ap_per_tag))

    ap_dict = {tid: float(ap_per_tag[tid]) for tid in range(n_tags)}

    best_f1, best_threshold = 0.0, 0.5
    best_micro = 0.0
    for threshold in np.arange(0.1, 1.0, 0.05):
        tp, fp, fn = _f1_at_threshold(probs_np, targets_np, threshold)
        macro, micro = _f1_from_counts(tp, fp, fn)
        if macro > best_f1:
            best_f1 = macro
            best_threshold = threshold
            best_micro = micro

    return {
        "map": map_score,
        "macro_f1": best_f1,
        "best_threshold": best_threshold,
        "micro_f1": best_micro,
        "per_tag_ap": ap_dict,
    }


def print_metrics(metrics, tag_to_id, title="Evaluation Results", extra=None, show_per_tag=True, tag_counts=None):
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
            ap = metrics.get("per_tag_ap", {}).get(tid, float("nan"))
            if not np.isnan(ap):
                count_str = f"  #{tag_counts[tid]}" if tag_counts and tid in tag_counts else ""
                print(f"    {tag:<30s} {ap:.4f}{count_str}")
    print(f"{'=' * 40}")
