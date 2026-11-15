"""
CET-KG Noise Robustness Evaluation
====================================
Controlled experiments to evaluate how the forecasting pipeline and
conformal calibration degrade under two types of noise:

1. Historical noise  — Gaussian perturbation of event_intensity values.
2. Label noise       — Random flipping of relation labels in training data.

Status: IMPLEMENTED (synthetic proof-of-concept).
"""

import os
import json
import copy
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def inject_intensity_noise(
    df: pd.DataFrame,
    sigma: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Add Gaussian noise to event_intensity values.

    Parameters
    ----------
    df : pd.DataFrame
        Original event data.
    sigma : float
        Standard deviation of Gaussian noise.
    seed : int

    Returns
    -------
    pd.DataFrame with perturbed intensity values clipped to [0.01, 0.99].
    """
    rng = np.random.RandomState(seed)
    noisy = df.copy()
    noise = rng.normal(0, sigma, size=len(noisy))
    noisy["event_intensity"] = np.clip(
        noisy["event_intensity"].values + noise, 0.01, 0.99
    )
    return noisy


def inject_label_noise(
    df: pd.DataFrame,
    flip_rate: float,
    relations: List[str],
    seed: int = 42,
) -> pd.DataFrame:
    """Randomly flip relation labels for a fraction of events.

    Parameters
    ----------
    df : pd.DataFrame
    flip_rate : float
        Fraction of events whose label is randomly replaced.
    relations : list of str
        All possible relation types.
    seed : int

    Returns
    -------
    pd.DataFrame with some relation labels corrupted.
    """
    rng = np.random.RandomState(seed)
    noisy = df.copy()
    n = len(noisy)
    n_flip = int(n * flip_rate)

    flip_indices = rng.choice(n, size=n_flip, replace=False)
    for idx in flip_indices:
        original = noisy.iloc[idx]["relation"]
        alternatives = [r for r in relations if r != original]
        noisy.at[noisy.index[idx], "relation"] = rng.choice(alternatives)

    return noisy


def run_noise_experiment(
    full_df: pd.DataFrame,
    relations: List[str],
    build_graph_fn,
    compute_temporal_features_fn,
    compute_pair_features_fn,
    build_features_fn,
    train_model_fn,
    evaluate_model_fn,
    run_conformal_fn,
    get_splits_fn,
    target_coverage: float = 0.90,
    intensity_sigmas: Optional[List[float]] = None,
    label_flip_rates: Optional[List[float]] = None,
    seed: int = 42,
    feature_window: int = 30,
    pair_window: int = 60,
) -> Dict[str, object]:
    """Run full noise experiment suite.

    Returns dict with results for each noise type and level.
    """
    if intensity_sigmas is None:
        intensity_sigmas = [0.0, 0.1, 0.2, 0.3]
    if label_flip_rates is None:
        label_flip_rates = [0.0, 0.05, 0.1, 0.2]

    results = {
        "intensity_noise": [],
        "label_noise": [],
        "target_coverage": target_coverage,
    }

    # --- Intensity noise experiments ---
    print("\n[noise_eval] === Intensity Noise Experiments ===")
    for sigma in intensity_sigmas:
        print(f"\n[noise_eval] Intensity noise sigma={sigma}")
        if sigma == 0.0:
            noisy_df = full_df.copy()
        else:
            noisy_df = inject_intensity_noise(full_df, sigma, seed=seed)

        res = _run_single_experiment(
            noisy_df, relations, build_graph_fn,
            compute_temporal_features_fn, compute_pair_features_fn,
            build_features_fn, train_model_fn, evaluate_model_fn,
            run_conformal_fn, get_splits_fn,
            target_coverage, seed, feature_window, pair_window,
            label=f"intensity_sigma={sigma}",
        )
        res["sigma"] = sigma
        results["intensity_noise"].append(res)

    # --- Label noise experiments ---
    print("\n[noise_eval] === Label Noise Experiments ===")
    for flip_rate in label_flip_rates:
        print(f"\n[noise_eval] Label noise flip_rate={flip_rate}")
        train, calib, test = get_splits_fn(full_df)

        if flip_rate == 0.0:
            noisy_train = train.copy()
        else:
            noisy_train = inject_label_noise(train, flip_rate, relations, seed=seed)

        # Reconstruct full df with noisy train
        noisy_full = pd.concat([noisy_train, calib, test]).sort_values("timestamp").reset_index(drop=True)

        res = _run_single_experiment(
            noisy_full, relations, build_graph_fn,
            compute_temporal_features_fn, compute_pair_features_fn,
            build_features_fn, train_model_fn, evaluate_model_fn,
            run_conformal_fn, get_splits_fn,
            target_coverage, seed, feature_window, pair_window,
            label=f"label_flip={flip_rate}",
        )
        res["flip_rate"] = flip_rate
        results["label_noise"].append(res)

    return results


def _run_single_experiment(
    df, relations, build_graph_fn,
    compute_temporal_features_fn, compute_pair_features_fn,
    build_features_fn, train_model_fn, evaluate_model_fn,
    run_conformal_fn, get_splits_fn,
    target_coverage, seed, feature_window, pair_window, label,
):
    """Run a single noise condition: build graph, train, evaluate, conformal."""
    from sklearn.preprocessing import LabelEncoder

    train, calib, test = get_splits_fn(df)
    graph = build_graph_fn(df, max_timestamp=None)

    # Build features
    X_train, y_train, _ = build_features_fn(
        train, graph, relations,
        compute_temporal_features_fn, compute_pair_features_fn,
        feature_window=feature_window, pair_window=pair_window,
        progress_label=f"{label}/train",
    )
    X_calib, y_calib, _ = build_features_fn(
        calib, graph, relations,
        compute_temporal_features_fn, compute_pair_features_fn,
        feature_window=feature_window, pair_window=pair_window,
        progress_label=f"{label}/calib",
    )
    X_test, y_test, meta_test = build_features_fn(
        test, graph, relations,
        compute_temporal_features_fn, compute_pair_features_fn,
        feature_window=feature_window, pair_window=pair_window,
        progress_label=f"{label}/test",
    )

    # Train
    clf, le = train_model_fn(X_train, y_train, seed=seed, n_estimators=100, max_depth=12)

    # Evaluate
    test_results = evaluate_model_fn(clf, le, X_test, y_test)

    # Conformal
    calib_y_encoded = le.transform(y_calib)
    calib_proba = clf.predict_proba(X_calib)
    test_timestamps = meta_test["timestamp"].values if "timestamp" in meta_test.columns else None

    conformal_results = run_conformal_fn(
        calib_y_encoded, calib_proba,
        y_test, test_results["y_pred_proba"],
        list(le.classes_), target_coverage, test_timestamps,
    )

    return {
        "accuracy": test_results["accuracy"],
        "macro_f1": test_results["macro_f1"],
        "top3_accuracy": test_results["top3_accuracy"],
        "empirical_coverage": conformal_results["empirical_coverage"],
        "avg_set_size": conformal_results["avg_set_size"],
    }


def save_noise_results(results: Dict, save_path: str) -> None:
    """Save noise experiment results to JSON."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[noise_eval] Saved noise results to {save_path}")
