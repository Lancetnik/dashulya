"""Check classifier input isolation and scoring using synthetic cases."""

import pytest

from evals.suites.addressing.benchmark import export_inputs, score_predictions


def test_export_preserves_model_input_and_keeps_expert_labels_private() -> None:
    cases = [
        {
            "case_id": "direct",
            "partition": "calibration",
            "metric_eligible": True,
            "gold": {"should_reply": True, "reason": "Evaluator-only explanation"},
        },
        {
            "case_id": "heldout",
            "partition": "test",
            "metric_eligible": True,
            "gold": {"should_reply": False},
        },
        {
            "case_id": "ambiguous",
            "partition": "calibration",
            "metric_eligible": False,
            "gold": {"should_reply": True},
        },
    ]
    inputs = [
        {"case_id": "direct", "input": 'Original "source"\nwith context'},
        {"case_id": "heldout", "input": "Other input"},
        {"case_id": "ambiguous", "input": "Uncertain input"},
    ]
    assert export_inputs(cases, inputs, partition="calibration") == [
        {"case_id": "direct", "input": 'Original "source"\nwith context'}
    ]


def test_auc_groups_ties_and_threshold_includes_equal_probabilities() -> None:
    cases = [
        {"case_id": "a", "gold": {"should_reply": True}},
        {"case_id": "b", "gold": {"should_reply": False}},
        {"case_id": "c", "gold": {"should_reply": True}},
        {"case_id": "d", "gold": {"should_reply": False}},
    ]
    predictions = [
        {"case_id": "d", "probability": 0.1},
        {"case_id": "b", "probability": 0.8},
        {"case_id": "a", "probability": 0.9},
        {"case_id": "c", "probability": 0.8},
    ]
    result = score_predictions(cases, predictions, threshold=0.8)
    assert result["roc_auc"] == pytest.approx(0.875)
    assert result["operating_point"] == {
        "threshold": 0.8,
        "tp": 2,
        "fp": 1,
        "tn": 1,
        "fn": 0,
        "recall": 1.0,
        "fpr": 0.5,
        "precision": 2 / 3,
        "specificity": 0.5,
        "accuracy": 0.75,
        "f1": 0.8,
    }


def test_export_rejects_evaluator_fields_in_model_input() -> None:
    cases = [{"case_id": "a", "metric_eligible": True}]
    with pytest.raises(ValueError, match="only case_id and input"):
        export_inputs(
            cases,
            [{"case_id": "a", "input": "Original", "gold": {"should_reply": True}}],
        )


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "unknown",
        "failed",
        "nan",
        "infinity",
        "boolean",
        "negative",
        "above_one",
    ],
)
def test_scoring_requires_complete_valid_probabilities(fault: str) -> None:
    cases = [
        {"case_id": "a", "gold": {"should_reply": True}},
        {"case_id": "b", "gold": {"should_reply": False}},
    ]
    predictions = [
        {"case_id": "a", "probability": 0.8},
        {"case_id": "b", "probability": 0.2},
    ]
    if fault == "missing":
        predictions.pop()
    elif fault == "duplicate":
        predictions.append(dict(predictions[0]))
    elif fault == "unknown":
        predictions[1]["case_id"] = "unknown"
    else:
        invalid = {
            "failed": None,
            "nan": float("nan"),
            "infinity": float("inf"),
            "boolean": True,
            "negative": -0.1,
            "above_one": 1.1,
        }
        predictions[0]["probability"] = invalid[fault]
    with pytest.raises(ValueError, match=r"Duplicate|exactly|finite probability"):
        score_predictions(cases, predictions)


def test_constant_scores_and_single_class_report_auc_explicitly() -> None:
    cases = [
        {"case_id": "a", "gold": {"should_reply": True}},
        {"case_id": "b", "gold": {"should_reply": False}},
    ]
    predictions = [
        {"case_id": "a", "probability": 0.5},
        {"case_id": "b", "probability": 0.5},
    ]
    assert score_predictions(cases, predictions)["roc_auc"] == pytest.approx(0.5)
    assert score_predictions(cases[:1], predictions[:1])["roc_auc"] is None
