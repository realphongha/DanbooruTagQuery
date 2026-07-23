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
