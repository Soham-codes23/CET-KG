"""
CET-KG Forecasting Layer
=========================
Multi-class relation prediction using temporal graph features and
a lightweight RandomForest classifier.

Task: given (head, tail, day), predict the relation type using only
historical information up to that day.

Status: IMPLEMENTED (synthetic proof-of-concept).
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings
import json

# Avoid sklearn convergence warnings during batch runs
warnings.filterwarnings("ignore", category=UserWarning)


def _build_feature_vector(
    head_feats: Dict[str, float],
    tail_feats: Dict[str, float],
    pair_feats: Dict[str, float],
) -> List[float]:
    """Concatenate entity and pair features into a flat vector."""
    vec = []
    for key in sorted(head_feats.keys()):
        vec.append(head_feats[key])
    for key in sorted(tail_feats.keys()):
        vec.append(tail_feats[key])
    for key in sorted(pair_feats.keys()):
        vec.append(pair_feats[key])
    return vec


def build_features_for_split(
    events_df: pd.DataFrame,
    full_graph,  # nx.MultiDiGraph
    relations: List[str],
    compute_temporal_features_fn,
    compute_pair_features_fn,
    feature_window: int = 30,
    pair_window: int = 60,
    progress_label: str = "split",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build feature matrix and label vector for a set of events.

    For each event (head, rel, tail, timestamp), compute features using
    only data from *before* the event's timestamp.

    Parameters
    ----------
    events_df : pd.DataFrame
        Events in this split.
    full_graph : nx.MultiDiGraph
        The full graph (features are computed with temporal filtering).
    relations : list of str
        All possible relation labels.
    compute_temporal_features_fn : callable
        temporal_graph.compute_temporal_features
    compute_pair_features_fn : callable
        temporal_graph.compute_pair_features
    feature_window : int
        Window for entity-level features.
    pair_window : int
        Window for pair-level features.

    Returns
    -------
    X : np.ndarray  (n_samples, n_features)
    y : np.ndarray  (n_samples,)  — relation labels as strings
    meta_df : pd.DataFrame with columns [timestamp, head_entity, tail_entity, true_relation]
    """
    X_rows: List[List[float]] = []
    y_labels: List[str] = []
    meta_rows: List[Dict] = []

    total = len(events_df)
    report_every = max(1, total // 10)

    for i, (_, row) in enumerate(events_df.iterrows()):
        if i % report_every == 0:
            print(f"  [{progress_label}] Building features: {i}/{total} ({100*i//total}%)")

        head = row["head_entity"]
        tail = row["tail_entity"]
        ts = int(row["timestamp"])

        head_feats = compute_temporal_features_fn(full_graph, head, ts, window=feature_window)
        tail_feats = compute_temporal_features_fn(full_graph, tail, ts, window=feature_window)
        pair_feats = compute_pair_features_fn(full_graph, head, tail, ts, relations, window=pair_window)

        vec = _build_feature_vector(head_feats, tail_feats, pair_feats)
        X_rows.append(vec)
        y_labels.append(row["relation"])
        meta_rows.append({
            "timestamp": ts,
            "head_entity": head,
            "tail_entity": tail,
            "true_relation": row["relation"],
        })

    print(f"  [{progress_label}] Feature building complete: {total} samples")
    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_labels)
    meta_df = pd.DataFrame(meta_rows)

    return X, y, meta_df


def train_forecasting_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    n_estimators: int = 200,
    max_depth: int = 15,
) -> Tuple[RandomForestClassifier, LabelEncoder]:
    """Train a RandomForest multi-class classifier.

    Returns the trained model and the label encoder.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_encoded)
    print(f"[forecasting] Trained RandomForest on {X_train.shape[0]} samples, "
          f"{X_train.shape[1]} features, {len(le.classes_)} classes")

    return clf, le


def evaluate_model(
    clf: RandomForestClassifier,
    le: LabelEncoder,
    X: np.ndarray,
    y_true: np.ndarray,
) -> Dict[str, object]:
    """Evaluate the model and return metrics + predictions.

    Returns dict with:
        accuracy, macro_f1, top3_accuracy,
        y_pred, y_pred_proba, confusion_matrix, classes
    """
    y_encoded_true = le.transform(y_true)
    y_pred_encoded = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)

    y_pred = le.inverse_transform(y_pred_encoded)

    acc = accuracy_score(y_encoded_true, y_pred_encoded)
    f1 = f1_score(y_encoded_true, y_pred_encoded, average="macro", zero_division=0)

    # Top-3 accuracy
    top3_correct = 0
    for i, true_label in enumerate(y_encoded_true):
        top3_indices = np.argsort(y_pred_proba[i])[-3:]
        if true_label in top3_indices:
            top3_correct += 1
    top3_acc = top3_correct / len(y_encoded_true) if len(y_encoded_true) > 0 else 0.0

    cm = confusion_matrix(y_encoded_true, y_pred_encoded)

    return {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "top3_accuracy": float(top3_acc),
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "y_true": y_true,
        "confusion_matrix": cm,
        "classes": list(le.classes_),
    }


def save_predictions(
    meta_df: pd.DataFrame,
    results: Dict,
    save_path: str,
) -> None:
    """Save predictions to CSV."""
    out = meta_df.copy()
    out["predicted_relation"] = results["y_pred"]
    out["correct"] = (out["true_relation"] == out["predicted_relation"])

    # Add top prediction probability
    y_pred_encoded = np.argmax(results["y_pred_proba"], axis=1)
    out["pred_confidence"] = results["y_pred_proba"][
        np.arange(len(y_pred_encoded)), y_pred_encoded
    ]

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    out.to_csv(save_path, index=False)
    print(f"[forecasting] Saved predictions to {save_path}")


def save_metrics(results: Dict, save_path: str) -> None:
    """Save scalar metrics to JSON."""
    metrics = {
        "accuracy": results["accuracy"],
        "macro_f1": results["macro_f1"],
        "top3_accuracy": results["top3_accuracy"],
        "n_test_samples": len(results["y_true"]),
        "n_classes": len(results["classes"]),
        "classes": results["classes"],
    }
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[forecasting] Saved metrics to {save_path}")
