"""Public offline boundaries for classifier inputs and scoring."""

import math
from typing import Any


def score_predictions(
    cases: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    if not cases:
        msg = "No cases selected."
        raise ValueError(msg)
    by_id = {r["case_id"]: r for r in predictions}
    if len(by_id) != len(predictions):
        msg = "Duplicate prediction identity."
        raise ValueError(msg)
    if len({c["case_id"] for c in cases}) != len(cases):
        msg = "Duplicate case identity."
        raise ValueError(msg)
    if set(by_id) != {c["case_id"] for c in cases}:
        msg = "Predictions must cover exactly the selected cases."
        raise ValueError(msg)
    values = []
    for case in cases:
        label = case["gold"]["should_reply"]
        if type(label) is not bool:
            msg = "Expert label must be boolean."
            raise ValueError(msg)
        probability = by_id[case["case_id"]].get("probability")
        if (
            not isinstance(probability, (int, float))
            or isinstance(probability, bool)
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            msg = "Every prediction needs a finite probability in [0, 1]."
            raise ValueError(msg)
        values.append((float(probability), label))
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        msg = "Threshold must be in [0, 1]."
        raise ValueError(msg)
    positives = sum(label for _, label in values)
    negatives = len(values) - positives

    def point(cutoff: float) -> dict[str, Any]:
        tp = sum(label and p >= cutoff for p, label in values)
        fp = sum(not label and p >= cutoff for p, label in values)
        fn, tn = positives - tp, negatives - fp
        return {
            "threshold": cutoff,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "recall": tp / positives if positives else None,
            "fpr": fp / negatives if negatives else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "specificity": tn / negatives if negatives else None,
            "accuracy": (tp + tn) / len(values),
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        }

    cutoffs = sorted({p for p, _ in values}, reverse=True)
    points = [point(cutoff) for cutoff in cutoffs]
    auc = None
    if positives and negatives:
        previous_fpr = previous_recall = 0.0
        auc = 0.0
        for row in points:
            auc += (row["fpr"] - previous_fpr) * (row["recall"] + previous_recall) / 2
            previous_fpr, previous_recall = row["fpr"], row["recall"]
    return {
        "count": len(values),
        "positives": positives,
        "negatives": negatives,
        "roc_auc": auc,
        "operating_point": point(threshold),
        "thresholds": points,
    }


def export_inputs(
    cases: list[dict[str, Any]],
    inputs: list[dict[str, Any]],
    *,
    partition: str | None = None,
    panel: str = "core",
) -> list[dict[str, Any]]:
    if panel not in {"core", "all"}:
        msg = "Panel must be core or all."
        raise ValueError(msg)
    by_id = {row["case_id"]: row for row in inputs}
    if len(by_id) != len(inputs) or len({c["case_id"] for c in cases}) != len(cases):
        msg = "Duplicate case identity."
        raise ValueError(msg)
    if set(by_id) != {case["case_id"] for case in cases}:
        msg = "Cases and inputs must have identical IDs."
        raise ValueError(msg)
    result = []
    for case in cases:
        if partition is not None and case["partition"] != partition:
            continue
        if panel == "core" and not case["metric_eligible"]:
            continue
        row = by_id[case["case_id"]]
        if set(row) != {"case_id", "input"} or not isinstance(row["input"], str):
            msg = "Model inputs allow only case_id and input text."
            raise ValueError(msg)
        result.append({"case_id": row["case_id"], "input": row["input"]})
    if not result:
        msg = "No cases selected."
        raise ValueError(msg)
    return result
