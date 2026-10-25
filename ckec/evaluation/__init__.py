"""
ckec.evaluation — Metrics, threshold calibration, ROC/PR curves, and reporting.

All evaluation functions accept NumPy arrays or pandas Series.
Thresholds MUST be calibrated on validation data; do not pass test-set data
to the calibration functions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_curve,
        precision_recall_curve,
    )
except ImportError:
    def precision_score(y_true, y_pred, zero_division=0):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else float(zero_division)

    def recall_score(y_true, y_pred, zero_division=0):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else float(zero_division)

    def f1_score(y_true, y_pred, zero_division=0):
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        return (2 * p * r) / (p + r) if (p + r) > 0 else float(zero_division)

    def roc_auc_score(y_true, scores):
        y_true, scores = np.asarray(y_true), np.asarray(scores)
        pos = scores[y_true == 1]
        neg = scores[y_true == 0]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        # Mann-Whitney U calculation
        n_pos, n_neg = len(pos), len(neg)
        all_scores = np.concatenate([pos, neg])
        ranks = np.argsort(np.argsort(all_scores)) + 1.0
        pos_ranks = ranks[:n_pos]
        return float((np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

    def average_precision_score(y_true, scores):
        y_true, scores = np.asarray(y_true), np.asarray(scores)
        desc_idx = np.argsort(-scores)
        y_sorted = y_true[desc_idx]
        tp = np.cumsum(y_sorted == 1)
        fp = np.cumsum(y_sorted == 0)
        precisions = tp / (tp + fp)
        recall_diffs = np.diff(np.concatenate([[0], tp / max(1, np.sum(y_true == 1))]))
        return float(np.sum(precisions * recall_diffs))

    def roc_curve(y_true, scores):
        y_true, scores = np.asarray(y_true), np.asarray(scores)
        thresholds = np.unique(scores)[::-1]
        tpr = [np.sum((scores >= th) & (y_true == 1)) / max(1, np.sum(y_true == 1)) for th in thresholds]
        fpr = [np.sum((scores >= th) & (y_true == 0)) / max(1, np.sum(y_true == 0)) for th in thresholds]
        return np.array(fpr), np.array(tpr), thresholds

    def precision_recall_curve(y_true, scores):
        y_true, scores = np.asarray(y_true), np.asarray(scores)
        thresholds = np.unique(scores)[::-1]
        precisions = [np.sum((scores >= th) & (y_true == 1)) / max(1, np.sum(scores >= th)) for th in thresholds]
        recalls = [np.sum((scores >= th) & (y_true == 1)) / max(1, np.sum(y_true == 1)) for th in thresholds]
        return np.array(precisions), np.array(recalls), thresholds

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DetectorResult:
    """All evaluation metrics for a single detector on a single dataset split."""

    detector_name: str
    split: str                    # "val" | "test"
    target_far: float

    # Core metrics
    detection_probability: float = float("nan")
    false_alarm_rate: float = float("nan")
    precision: float = float("nan")
    recall: float = float("nan")
    f1: float = float("nan")
    roc_auc: float = float("nan")
    pr_auc: float = float("nan")

    # Detection delay
    detection_delay_samples: float = float("nan")

    # Runtime
    runtime_seconds: float = float("nan")

    # Bootstrap confidence intervals (95%)
    ci_detection_probability: tuple[float, float] = (float("nan"), float("nan"))
    ci_false_alarm_rate: tuple[float, float] = (float("nan"), float("nan"))

    # Extra notes (failure cases, data limitations)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector_name,
            "split": self.split,
            "target_far": self.target_far,
            "detection_probability": self.detection_probability,
            "false_alarm_rate": self.false_alarm_rate,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "detection_delay_samples": self.detection_delay_samples,
            "runtime_seconds": self.runtime_seconds,
            "ci_dp_lo": self.ci_detection_probability[0],
            "ci_dp_hi": self.ci_detection_probability[1],
            "ci_far_lo": self.ci_false_alarm_rate[0],
            "ci_far_hi": self.ci_false_alarm_rate[1],
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    detector_name: str,
    split: str = "test",
    target_far: float = 0.01,
) -> DetectorResult:
    """
    Compute evaluation metrics for a single detector output.

    Labels must be binary: 0 = benign, 1 = attack.
    Scores must be finite. Thresholds must be calibrated on validation data.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)

    # Validate inputs before calculating metrics.
    if not (len(y_true) == len(y_pred) == len(scores)):
        raise ValueError(
            "y_true, y_pred, and scores must have identical lengths."
        )

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError(
            "Expected binary labels (0=benign, 1=attack). "
            "Map multi-class labels explicitly before evaluation."
        )

    if not np.isin(y_pred, [0, 1]).all():
        raise ValueError("y_pred must contain only 0 and 1.")

    if not np.isfinite(scores).all():
        raise ValueError(
            "Anomaly scores contain NaN or infinity. "
            "Report the invalid-score count and evaluation failure "
            "instead of silently filtering samples."
        )

    benign_mask = y_true == 0
    attack_mask = y_true == 1

    n_benign = int(benign_mask.sum())
    n_attack = int(attack_mask.sum())

    result = DetectorResult(
        detector_name=detector_name,
        split=split,
        target_far=target_far,
    )

    if n_benign == 0:
        log.warning(
            "%s: no benign samples in %s split.",
            detector_name, split,
        )

    if n_attack == 0:
        log.warning(
            "%s: no attack samples in %s split.",
            detector_name, split,
        )

    # Detection probability and false-alarm rate.
    if n_attack > 0:
        result.detection_probability = float(y_pred[attack_mask].mean())
        result.recall = result.detection_probability

    if n_benign > 0:
        result.false_alarm_rate = float(y_pred[benign_mask].mean())

    # Precision and F1.
    if np.any(y_pred == 1):
        result.precision = float(
            precision_score(y_true, y_pred, zero_division=0)
        )
        result.f1 = float(
            f1_score(y_true, y_pred, zero_division=0)
        )
    else:
        result.precision = 0.0
        result.f1 = 0.0

    # AUC metrics require both classes.
    if n_benign > 0 and n_attack > 0:
        result.roc_auc = float(roc_auc_score(y_true, scores))
        result.pr_auc = float(average_precision_score(y_true, scores))

    return result


def detection_delay(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attack_onset_idx: int,
) -> float:
    """
    Compute the number of samples from attack onset to first correct detection.

    Returns NaN if no detection occurred after the onset.

    Parameters
    ----------
    y_true          : ground-truth labels
    y_pred          : binary predictions
    attack_onset_idx : sample index where the attack begins
    """
    after_onset = np.where(y_pred[attack_onset_idx:] == 1)[0]
    if len(after_onset) == 0:
        return float("nan")
    return float(after_onset[0])


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_iter: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """
    Bootstrap confidence intervals for detection probability and FAR.

    Parameters
    ----------
    y_true  : ground-truth binary labels
    y_pred  : binary predictions
    n_iter  : number of bootstrap resamples
    alpha   : significance level (default 0.05 → 95% CI)
    seed    : random seed for reproducibility

    Returns
    -------
    dict with keys 'detection_probability' and 'false_alarm_rate', each
    mapping to a (lower, upper) tuple.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    dp_samples, far_samples = [], []

    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        attack_mask = yt != 0
        benign_mask = yt == 0
        dp = float(yp[attack_mask].sum() / attack_mask.sum()) if attack_mask.sum() > 0 else float("nan")
        far = float(yp[benign_mask].sum() / benign_mask.sum()) if benign_mask.sum() > 0 else float("nan")
        dp_samples.append(dp)
        far_samples.append(far)

    lo, hi = alpha / 2, 1 - alpha / 2
    return {
        "detection_probability": (
            float(np.nanquantile(dp_samples, lo)),
            float(np.nanquantile(dp_samples, hi)),
        ),
        "false_alarm_rate": (
            float(np.nanquantile(far_samples, lo)),
            float(np.nanquantile(far_samples, hi)),
        ),
    }


# ---------------------------------------------------------------------------
# Results table formatting
# ---------------------------------------------------------------------------

def results_to_dataframe(results: list[DetectorResult]) -> pd.DataFrame:
    """Convert a list of DetectorResult objects to a summary DataFrame."""
    return pd.DataFrame([r.to_dict() for r in results])


def print_results_table(results: list[DetectorResult]) -> None:
    """Print a formatted summary table to stdout."""
    df = results_to_dataframe(results)
    cols = [
        "detector", "split",
        "detection_probability", "false_alarm_rate",
        "precision", "f1", "roc_auc", "pr_auc",
        "detection_delay_samples",
    ]
    print(df[cols].to_string(index=False, float_format="{:.4f}".format))
