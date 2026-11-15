"""
CET-KG Run Pipeline
====================
End-to-end orchestrator: data → graph → forecasting → conformal → explanation → noise.

Usage:
    python code/run_pipeline.py --seed 42 --target-coverage 0.90
    python code/run_pipeline.py --skip-noise          # faster run
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Resolve project root so imports work when run from any directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from data_layer import (
    generate_synthetic_dataset, load_dataset, validate_dataset,
    get_temporal_splits, RELATIONS, ENTITIES,
)
from temporal_graph import (
    build_temporal_graph, compute_temporal_features, compute_pair_features,
)
from forecasting import (
    build_features_for_split, train_forecasting_model,
    evaluate_model, save_predictions, save_metrics,
)
from conformal import run_conformal_calibration, save_conformal_results
from explainability import (
    generate_explanation, visualize_explanation_subgraph, save_explanations,
)
from noise_eval import run_noise_experiment, save_noise_results

# Paths
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "synthetic_tkg_events.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")


def setup_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def set_style():
    """Set matplotlib style."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass  # Use default


# ===================================================================
# Visualization helpers
# ===================================================================

def plot_temporal_distribution(df: pd.DataFrame, save_path: str):
    """Bar chart of event count per day."""
    fig, ax = plt.subplots(figsize=(14, 4))
    counts = df.groupby("timestamp").size()
    ax.bar(counts.index, counts.values, width=1.0, color="#1976D2", alpha=0.7)
    ax.axvline(x=249.5, color="#F44336", ls="--", lw=1.5, label="Train | Calib")
    ax.axvline(x=299.5, color="#FF9800", ls="--", lw=1.5, label="Calib | Test")
    ax.set_xlabel("Day")
    ax.set_ylabel("Events")
    ax.set_title("Temporal Event Distribution (Synthetic TKG)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[pipeline] Saved {save_path}")


def plot_relation_distribution(df: pd.DataFrame, save_path: str):
    """Horizontal bar chart of relation frequencies."""
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df["relation"].value_counts().sort_values()
    ax.barh(counts.index, counts.values, color="#42A5F5")
    ax.set_xlabel("Count")
    ax.set_title("Relation Frequency Distribution")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[pipeline] Saved {save_path}")


def plot_confusion_matrix(cm: np.ndarray, classes: list, save_path: str):
    """Confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, shrink=0.8)
    tick_marks = range(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[pipeline] Saved {save_path}")


def plot_coverage_by_time(coverage_by_bucket: dict, target: float, save_path: str):
    """Bar chart of coverage by time bucket."""
    fig, ax = plt.subplots(figsize=(6, 4))
    buckets = list(coverage_by_bucket.keys())
    values = [coverage_by_bucket[b] for b in buckets]
    colors = ["#66BB6A" if v >= target else "#EF5350" for v in values]
    ax.bar(buckets, values, color=colors, alpha=0.8, edgecolor="#333")
    ax.axhline(y=target, color="#1565C0", ls="--", lw=2, label=f"Target ({target:.0%})")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("Coverage by Time Bucket")
    ax.set_ylim(0, 1.05)
    ax.legend()
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[pipeline] Saved {save_path}")


def plot_set_sizes(set_sizes: list, save_path: str):
    """Histogram of prediction set sizes."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(set_sizes, bins=range(1, max(set_sizes) + 2), color="#42A5F5",
            edgecolor="#1565C0", alpha=0.8, align="left")
    ax.axvline(x=np.mean(set_sizes), color="#F44336", ls="--", lw=2,
               label=f"Mean: {np.mean(set_sizes):.1f}")
    ax.set_xlabel("Prediction Set Size")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Prediction Set Sizes")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[pipeline] Saved {save_path}")


def plot_noise_robustness(noise_results: dict, save_path: str):
    """Line plots of accuracy, coverage, set size vs noise level."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Intensity noise
    int_data = noise_results["intensity_noise"]
    sigmas = [d["sigma"] for d in int_data]
    int_acc = [d["accuracy"] for d in int_data]
    int_cov = [d["empirical_coverage"] for d in int_data]
    int_sz = [d["avg_set_size"] for d in int_data]

    axes[0].plot(sigmas, int_acc, "o-", color="#1976D2", label="Intensity noise")
    # Label noise
    lbl_data = noise_results["label_noise"]
    flips = [d["flip_rate"] for d in lbl_data]
    lbl_acc = [d["accuracy"] for d in lbl_data]
    axes[0].plot(flips, lbl_acc, "s--", color="#F44336", label="Label noise")
    axes[0].set_xlabel("Noise Level")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy vs Noise")
    axes[0].legend()

    axes[1].plot(sigmas, int_cov, "o-", color="#1976D2", label="Intensity noise")
    lbl_cov = [d["empirical_coverage"] for d in lbl_data]
    axes[1].plot(flips, lbl_cov, "s--", color="#F44336", label="Label noise")
    axes[1].axhline(y=noise_results["target_coverage"], color="#888", ls=":", lw=1.5,
                    label=f"Target ({noise_results['target_coverage']:.0%})")
    axes[1].set_xlabel("Noise Level")
    axes[1].set_ylabel("Empirical Coverage")
    axes[1].set_title("Coverage vs Noise")
    axes[1].legend()

    axes[2].plot(sigmas, int_sz, "o-", color="#1976D2", label="Intensity noise")
    lbl_sz = [d["avg_set_size"] for d in lbl_data]
    axes[2].plot(flips, lbl_sz, "s--", color="#F44336", label="Label noise")
    axes[2].set_xlabel("Noise Level")
    axes[2].set_ylabel("Avg Set Size")
    axes[2].set_title("Set Size vs Noise")
    axes[2].legend()

    plt.suptitle("Noise Robustness Analysis", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[pipeline] Saved {save_path}")


# ===================================================================
# Main pipeline
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="CET-KG Pipeline")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--target-coverage", type=float, default=0.90,
                        help="Target coverage for conformal calibration")
    parser.add_argument("--skip-noise", action="store_true",
                        help="Skip noise robustness experiments (faster)")
    parser.add_argument("--feature-window", type=int, default=30,
                        help="Window for temporal features")
    parser.add_argument("--pair-window", type=int, default=60,
                        help="Window for pair features")
    args = parser.parse_args()

    np.random.seed(args.seed)
    setup_dirs()
    set_style()
    t_start = time.time()

    print("=" * 70)
    print("  CET-KG Pipeline — Synthetic Proof-of-Concept")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Phase 1: Data
    # ------------------------------------------------------------------
    print("\n[Phase 1] Data Generation & Validation")
    print("-" * 40)

    if os.path.isfile(DATA_PATH):
        print(f"[pipeline] Loading existing dataset from {DATA_PATH}")
        df = load_dataset(DATA_PATH)
    else:
        print(f"[pipeline] Generating synthetic dataset...")
        df = generate_synthetic_dataset(seed=args.seed, save_path=DATA_PATH)

    info = validate_dataset(df)
    print(f"[pipeline] Dataset: {info['n_events']} events, "
          f"{info['n_entities']} entities, {info['n_relations']} relations, "
          f"days {info['timestamp_range'][0]}-{info['timestamp_range'][1]}")
    if info["warnings"]:
        for w in info["warnings"]:
            print(f"  ⚠ {w}")

    train, calib, test = get_temporal_splits(df)

    # Visualizations
    plot_temporal_distribution(df, os.path.join(FIGURES_DIR, "temporal_event_distribution.png"))
    plot_relation_distribution(df, os.path.join(FIGURES_DIR, "relation_distribution.png"))

    # ------------------------------------------------------------------
    # Phase 2: Temporal Graph
    # ------------------------------------------------------------------
    print("\n[Phase 2] Temporal Graph Construction")
    print("-" * 40)

    full_graph = build_temporal_graph(df)
    print(f"[pipeline] Full graph: {full_graph.number_of_nodes()} nodes, "
          f"{full_graph.number_of_edges()} edges")

    # ------------------------------------------------------------------
    # Phase 3: Forecasting
    # ------------------------------------------------------------------
    print("\n[Phase 3] Forecasting")
    print("-" * 40)

    # Sample events for feature building (use all for small datasets)
    # For performance on M2, subsample if > 2000 events per split
    def maybe_subsample(split_df, max_n=1500, seed=42):
        if len(split_df) > max_n:
            return split_df.sample(n=max_n, random_state=seed)
        return split_df

    train_sub = maybe_subsample(train, max_n=1500, seed=args.seed)
    calib_sub = maybe_subsample(calib, max_n=800, seed=args.seed)
    test_sub = maybe_subsample(test, max_n=800, seed=args.seed)

    print(f"[pipeline] Feature building samples — train: {len(train_sub)}, "
          f"calib: {len(calib_sub)}, test: {len(test_sub)}")

    X_train, y_train, meta_train = build_features_for_split(
        train_sub, full_graph, RELATIONS,
        compute_temporal_features, compute_pair_features,
        feature_window=args.feature_window, pair_window=args.pair_window,
        progress_label="train",
    )

    X_calib, y_calib, meta_calib = build_features_for_split(
        calib_sub, full_graph, RELATIONS,
        compute_temporal_features, compute_pair_features,
        feature_window=args.feature_window, pair_window=args.pair_window,
        progress_label="calib",
    )

    X_test, y_test, meta_test = build_features_for_split(
        test_sub, full_graph, RELATIONS,
        compute_temporal_features, compute_pair_features,
        feature_window=args.feature_window, pair_window=args.pair_window,
        progress_label="test",
    )

    # Train model
    clf, le = train_forecasting_model(X_train, y_train, seed=args.seed)

    # Evaluate on test
    test_results = evaluate_model(clf, le, X_test, y_test)
    print(f"[pipeline] Test Accuracy:  {test_results['accuracy']:.4f}")
    print(f"[pipeline] Test Macro-F1:  {test_results['macro_f1']:.4f}")
    print(f"[pipeline] Test Top-3 Acc: {test_results['top3_accuracy']:.4f}")

    # Save
    save_predictions(
        meta_test, test_results,
        os.path.join(RESULTS_DIR, "forecast_predictions.csv"),
    )
    save_metrics(test_results, os.path.join(RESULTS_DIR, "forecast_metrics.json"))

    # Confusion matrix
    plot_confusion_matrix(
        test_results["confusion_matrix"], test_results["classes"],
        os.path.join(FIGURES_DIR, "confusion_matrix.png"),
    )

    # ------------------------------------------------------------------
    # Phase 4: Conformal Calibration
    # ------------------------------------------------------------------
    print("\n[Phase 4] Conformal Calibration")
    print("-" * 40)

    # Get calibration predictions
    calib_y_encoded = le.transform(y_calib)
    calib_proba = clf.predict_proba(X_calib)

    conformal_results = run_conformal_calibration(
        calib_y_true_encoded=calib_y_encoded,
        calib_y_pred_proba=calib_proba,
        test_y_true=y_test,
        test_y_pred_proba=test_results["y_pred_proba"],
        classes=test_results["classes"],
        target_coverage=args.target_coverage,
        test_timestamps=meta_test["timestamp"].values,
    )

    save_conformal_results(
        conformal_results,
        os.path.join(RESULTS_DIR, "conformal_results.json"),
    )

    # Visualizations
    plot_coverage_by_time(
        conformal_results["coverage_by_bucket"], args.target_coverage,
        os.path.join(FIGURES_DIR, "coverage_by_time.png"),
    )
    plot_set_sizes(
        conformal_results["set_sizes"],
        os.path.join(FIGURES_DIR, "prediction_set_sizes.png"),
    )

    # ------------------------------------------------------------------
    # Phase 5: Explanations
    # ------------------------------------------------------------------
    print("\n[Phase 5] Graph-Grounded Explanations")
    print("-" * 40)

    # Pick 5 diverse test predictions to explain
    sample_indices = np.linspace(0, len(test_sub) - 1, 5, dtype=int)
    explanations = []

    for idx in sample_indices:
        row = test_sub.iloc[idx]
        pred_rel = test_results["y_pred"][idx]
        expl = generate_explanation(
            full_graph,
            predicted_head=row["head_entity"],
            predicted_relation=pred_rel,
            predicted_tail=row["tail_entity"],
            predicted_timestamp=int(row["timestamp"]),
            window=60,
            top_k=10,
        )
        explanations.append(expl)
        print(f"  Explained: {row['head_entity']} --[{pred_rel}]--> "
              f"{row['tail_entity']} @ day {row['timestamp']} "
              f"({expl['n_candidates_considered']} evidence candidates)")

    save_explanations(explanations, os.path.join(RESULTS_DIR, "explanations.json"))

    # Visualize first explanation
    if explanations:
        visualize_explanation_subgraph(
            explanations[0],
            save_path=os.path.join(FIGURES_DIR, "sample_explanation_subgraph.png"),
        )

    # ------------------------------------------------------------------
    # Phase 6: Noise Robustness (optional)
    # ------------------------------------------------------------------
    if not args.skip_noise:
        print("\n[Phase 6] Noise Robustness Evaluation")
        print("-" * 40)

        noise_results = run_noise_experiment(
            full_df=df,
            relations=RELATIONS,
            build_graph_fn=build_temporal_graph,
            compute_temporal_features_fn=compute_temporal_features,
            compute_pair_features_fn=compute_pair_features,
            build_features_fn=build_features_for_split,
            train_model_fn=train_forecasting_model,
            evaluate_model_fn=evaluate_model,
            run_conformal_fn=run_conformal_calibration,
            get_splits_fn=get_temporal_splits,
            target_coverage=args.target_coverage,
            intensity_sigmas=[0.0, 0.1, 0.2, 0.3],
            label_flip_rates=[0.0, 0.05, 0.1, 0.2],
            seed=args.seed,
            feature_window=args.feature_window,
            pair_window=args.pair_window,
        )

        save_noise_results(noise_results, os.path.join(RESULTS_DIR, "noise_results.json"))
        plot_noise_robustness(noise_results, os.path.join(FIGURES_DIR, "noise_robustness.png"))
    else:
        print("\n[Phase 6] Noise experiments skipped (--skip-noise)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("  CET-KG Pipeline Complete")
    print("=" * 70)
    print(f"  Time elapsed:      {elapsed:.1f}s")
    print(f"  Dataset:           {info['n_events']} synthetic events")
    print(f"  Accuracy:          {test_results['accuracy']:.4f}")
    print(f"  Macro-F1:          {test_results['macro_f1']:.4f}")
    print(f"  Top-3 Accuracy:    {test_results['top3_accuracy']:.4f}")
    print(f"  Target Coverage:   {args.target_coverage:.2f}")
    print(f"  Empirical Coverage:{conformal_results['empirical_coverage']:.4f}")
    print(f"  Avg Set Size:      {conformal_results['avg_set_size']:.2f}")
    print(f"  Explanations:      {len(explanations)} generated")
    print(f"  Results:           {RESULTS_DIR}")
    print(f"  Figures:           {FIGURES_DIR}")
    print("=" * 70)
    print("  NOTE: All results are from SYNTHETIC data.")
    print("        This is a proof-of-concept, not benchmark validation.")
    print("=" * 70)


if __name__ == "__main__":
    main()
