"""
CET-KG Conformal Calibration
==============================
Split-conformal prediction sets for multi-class relation forecasting.

Method: Adaptive Prediction Sets (APS) variant of split-conformal inference.
- Calibration scores are computed on a held-out calibration period.
- Non-conformity score = 1 − P(true class).
- Prediction sets include all classes whose cumulative sorted probability
  exceeds the conformal quantile threshold.

IMPORTANT CAVEAT:
    Split-conformal prediction guarantees marginal coverage under the
    assumption of exchangeability (i.i.d. data). Temporal knowledge graph
    events are NOT exchangeable — they exhibit temporal dependence and
    potential distribution shift. The reported coverage is therefore an
    *empirical approximation*, not a formal guarantee. This is documented
    as a known limitation.

Status: IMPLEMENTED (synthetic proof-of-concept).
"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional


def compute_nonconformity_scores(
    y_true_encoded: np.ndarray,
    y_pred_proba: np.ndarray,
) -> np.ndarray:
    """Compute non-conformity scores for calibration instances.

    score_i = 1 − P(y_true_i)

    Parameters
    ----------
    y_true_encoded : np.ndarray of int, shape (n,)
        True class indices.
    y_pred_proba : np.ndarray of float, shape (n, n_classes)
        Predicted class probabilities.

    Returns
    -------
    scores : np.ndarray of float, shape (n,)
    """
    n = len(y_true_encoded)
    scores = np.zeros(n)
    for i in range(n):
        scores[i] = 1.0 - y_pred_proba[i, y_true_encoded[i]]
    return scores


def compute_quantile(scores: np.ndarray, target_coverage: float) -> float:
    """Compute the conformal quantile from calibration scores.

    q_hat = ceil((n+1) * alpha) / n  -th quantile of scores,
    where alpha = target_coverage.

    This is the standard finite-sample-corrected quantile.
    """
    n = len(scores)
    # Adjusted quantile level for finite-sample coverage
    adjusted_level = np.ceil((n + 1) * target_coverage) / n
    adjusted_level = min(adjusted_level, 1.0)
    q_hat = float(np.quantile(scores, adjusted_level))
    return q_hat


def build_prediction_sets(
    y_pred_proba: np.ndarray,
    q_hat: float,
    classes: List[str],
) -> List[List[str]]:
    """Construct prediction sets for each test instance.

    For each instance, the prediction set includes all classes c
    where P(c) >= 1 − q_hat.

    Parameters
    ----------
    y_pred_proba : np.ndarray, shape (n, n_classes)
    q_hat : float
        Conformal quantile threshold.
    classes : list of str
        Class names corresponding to columns of y_pred_proba.

    Returns
    -------
    prediction_sets : list of list of str
    """
    threshold = 1.0 - q_hat
    prediction_sets: List[List[str]] = []

    for i in range(len(y_pred_proba)):
        pset = []
        for j, cls in enumerate(classes):
            if y_pred_proba[i, j] >= threshold:
                pset.append(cls)
        # Guarantee at least the top-1 prediction is in the set
        if not pset:
            top_idx = int(np.argmax(y_pred_proba[i]))
            pset = [classes[top_idx]]
        prediction_sets.append(pset)

    return prediction_sets


def evaluate_coverage(
    prediction_sets: List[List[str]],
    y_true: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Evaluate empirical coverage and efficiency of prediction sets.

    Returns
    -------
    dict with:
        empirical_coverage : float
        avg_set_size : float
        set_sizes : list of int
        coverage_by_bucket : dict  (early/mid/late thirds of test set)
    """
    n = len(y_true)
    covered = 0
    set_sizes: List[int] = []

    for i in range(n):
        size = len(prediction_sets[i])
        set_sizes.append(size)
        if y_true[i] in prediction_sets[i]:
            covered += 1

    empirical_coverage = covered / n if n > 0 else 0.0
    avg_set_size = float(np.mean(set_sizes)) if set_sizes else 0.0

    # Coverage by time bucket
    coverage_by_bucket: Dict[str, float] = {}
    if timestamps is not None and n > 0:
        third = n // 3
        for label, start, end in [("early", 0, third), ("mid", third, 2*third), ("late", 2*third, n)]:
            bucket_covered = sum(
                1 for i in range(start, end) if y_true[i] in prediction_sets[i]
            )
            bucket_n = end - start
            coverage_by_bucket[label] = bucket_covered / bucket_n if bucket_n > 0 else 0.0
    else:
        coverage_by_bucket = {"early": 0.0, "mid": 0.0, "late": 0.0}

    return {
        "empirical_coverage": empirical_coverage,
        "avg_set_size": avg_set_size,
        "set_sizes": set_sizes,
        "coverage_by_bucket": coverage_by_bucket,
        "n_test": n,
    }


def run_conformal_calibration(
    calib_y_true_encoded: np.ndarray,
    calib_y_pred_proba: np.ndarray,
    test_y_true: np.ndarray,
    test_y_pred_proba: np.ndarray,
    classes: List[str],
    target_coverage: float = 0.90,
    test_timestamps: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Full conformal calibration pipeline.

    1. Compute non-conformity scores on calibration set.
    2. Compute quantile threshold.
    3. Build prediction sets on test set.
    4. Evaluate coverage and efficiency.

    Returns dict with all results.
    """
    # Step 1: calibration scores
    calib_scores = compute_nonconformity_scores(calib_y_true_encoded, calib_y_pred_proba)
    print(f"[conformal] Calibration scores: n={len(calib_scores)}, "
          f"mean={calib_scores.mean():.4f}, median={np.median(calib_scores):.4f}")

    # Step 2: quantile
    q_hat = compute_quantile(calib_scores, target_coverage)
    print(f"[conformal] Quantile q_hat = {q_hat:.4f} for target coverage {target_coverage}")

    # Step 3: prediction sets
    prediction_sets = build_prediction_sets(test_y_pred_proba, q_hat, classes)

    # Step 4: evaluate
    eval_results = evaluate_coverage(prediction_sets, test_y_true, test_timestamps)

    print(f"[conformal] Empirical coverage: {eval_results['empirical_coverage']:.4f} "
          f"(target: {target_coverage})")
    print(f"[conformal] Average set size: {eval_results['avg_set_size']:.2f}")
    print(f"[conformal] Coverage by bucket: {eval_results['coverage_by_bucket']}")

    return {
        "target_coverage": target_coverage,
        "q_hat": q_hat,
        "n_calibration": len(calib_scores),
        "calib_score_mean": float(calib_scores.mean()),
        "calib_score_median": float(np.median(calib_scores)),
        "empirical_coverage": eval_results["empirical_coverage"],
        "avg_set_size": eval_results["avg_set_size"],
        "set_sizes": eval_results["set_sizes"],
        "coverage_by_bucket": eval_results["coverage_by_bucket"],
        "n_test": eval_results["n_test"],
        "prediction_sets": prediction_sets,
        "exchangeability_caveat": (
            "Split-conformal prediction assumes exchangeability (i.i.d. data). "
            "Temporal knowledge graph events are temporally dependent, so the "
            "marginal coverage guarantee is approximate, not exact. Distribution "
            "shift between calibration and test periods may cause the empirical "
            "coverage to deviate from the target."
        ),
    }


def save_conformal_results(results: Dict, save_path: str) -> None:
    """Save conformal results to JSON (excluding large arrays for readability)."""
    out = {k: v for k, v in results.items() if k != "prediction_sets"}
    # Convert set_sizes to summary stats instead of full list
    sizes = out.pop("set_sizes", [])
    out["set_size_distribution"] = {
        "min": int(min(sizes)) if sizes else 0,
        "max": int(max(sizes)) if sizes else 0,
        "mean": float(np.mean(sizes)) if sizes else 0,
        "median": float(np.median(sizes)) if sizes else 0,
        "std": float(np.std(sizes)) if sizes else 0,
    }

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[conformal] Saved results to {save_path}")
