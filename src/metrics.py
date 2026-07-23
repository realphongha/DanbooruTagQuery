import numpy as np
from sklearn.metrics import average_precision_score, f1_score

def multilabel_metrics(probabilities, targets, threshold=0.5):
    probabilities = np.asarray(probabilities); targets = np.asarray(targets).astype(int)
    per_tag = {index: float(average_precision_score(targets[:, index], probabilities[:, index])) if targets[:, index].any() else float("nan") for index in range(targets.shape[1])}
    return {"map": float(np.nanmean(list(per_tag.values()))), "macro_f1": float(f1_score(targets, probabilities >= threshold, average="macro", zero_division=0)), "micro_f1": float(f1_score(targets, probabilities >= threshold, average="micro", zero_division=0)), "per_tag_ap": per_tag}
