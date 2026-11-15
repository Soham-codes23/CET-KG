"""
CET-KG Tests
=============
Basic tests for data processing, temporal splitting, graph construction,
conformal calibration, and no-data-leakage verification.

Run:  python3 code/tests.py
"""

import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

PASS = 0
FAIL = 0


def report(name, passed, detail=""):
    global PASS, FAIL
    if passed:
        PASS += 1
        print(f"  ✅ PASS: {name}")
    else:
        FAIL += 1
        print(f"  ❌ FAIL: {name} — {detail}")


def test_data_loading():
    """Test: dataset loads with correct columns and no nulls."""
    from data_layer import generate_synthetic_dataset, validate_dataset, REQUIRED_COLUMNS

    df = generate_synthetic_dataset(seed=42)
    # Correct columns
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    report("data_loading/columns", len(missing) == 0,
           f"Missing columns: {missing}")

    # No nulls
    null_count = df[REQUIRED_COLUMNS].isnull().sum().sum()
    report("data_loading/no_nulls", null_count == 0,
           f"Null count: {null_count}")

    # Positive event count
    report("data_loading/has_events", len(df) > 1000,
           f"Only {len(df)} events generated")

    # Validate
    info = validate_dataset(df)
    report("data_loading/validation", info["valid"],
           f"Warnings: {info.get('warnings', [])}")

    # No self-loops
    self_loops = (df["head_entity"] == df["tail_entity"]).sum()
    report("data_loading/no_self_loops", self_loops == 0,
           f"{self_loops} self-loops found")


def test_temporal_splits():
    """Test: train < calib < test timestamps with no overlap."""
    from data_layer import generate_synthetic_dataset, get_temporal_splits

    df = generate_synthetic_dataset(seed=42)
    train, calib, test = get_temporal_splits(df)

    train_max = train["timestamp"].max()
    calib_min = calib["timestamp"].min()
    calib_max = calib["timestamp"].max()
    test_min = test["timestamp"].min()

    report("temporal_splits/train_before_calib", train_max < calib_min,
           f"train_max={train_max}, calib_min={calib_min}")

    report("temporal_splits/calib_before_test", calib_max < test_min,
           f"calib_max={calib_max}, test_min={test_min}")

    # No overlap
    train_ts = set(train["timestamp"])
    calib_ts = set(calib["timestamp"])
    test_ts = set(test["timestamp"])

    report("temporal_splits/no_train_calib_overlap",
           len(train_ts & calib_ts) == 0,
           f"Overlap: {train_ts & calib_ts}")

    report("temporal_splits/no_calib_test_overlap",
           len(calib_ts & test_ts) == 0,
           f"Overlap: {calib_ts & test_ts}")

    # All events accounted for
    total = len(train) + len(calib) + len(test)
    report("temporal_splits/complete_coverage", total == len(df),
           f"train+calib+test={total}, total={len(df)}")


def test_graph_construction():
    """Test: graph has correct nodes and edges."""
    from data_layer import generate_synthetic_dataset
    from temporal_graph import build_temporal_graph

    df = generate_synthetic_dataset(seed=42)
    G = build_temporal_graph(df)

    n_entities = len(set(df["head_entity"]) | set(df["tail_entity"]))
    report("graph/correct_nodes", G.number_of_nodes() == n_entities,
           f"graph={G.number_of_nodes()}, expected={n_entities}")

    report("graph/correct_edges", G.number_of_edges() == len(df),
           f"graph={G.number_of_edges()}, expected={len(df)}")

    # Filtered graph
    G_filtered = build_temporal_graph(df, max_timestamp=100)
    expected = len(df[df["timestamp"] <= 100])
    report("graph/filtered_edges", G_filtered.number_of_edges() == expected,
           f"filtered={G_filtered.number_of_edges()}, expected={expected}")


def test_conformal_coverage():
    """Test: conformal coverage computation is correct on toy data."""
    from conformal import compute_nonconformity_scores, compute_quantile, build_prediction_sets, evaluate_coverage

    # Toy: 5 samples, 3 classes, perfect predictions
    y_true = np.array([0, 1, 2, 0, 1])
    proba = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1],
        [0.1, 0.7, 0.2],
    ])
    classes = ["A", "B", "C"]
    y_true_str = np.array(["A", "B", "C", "A", "B"])

    scores = compute_nonconformity_scores(y_true, proba)
    report("conformal/scores_range", np.all((scores >= 0) & (scores <= 1)),
           f"scores: {scores}")

    # Score = 1 - P(true), so for first sample: 1-0.8 = 0.2
    report("conformal/score_computation", abs(scores[0] - 0.2) < 1e-6,
           f"Expected 0.2, got {scores[0]}")

    q = compute_quantile(scores, 0.90)
    report("conformal/quantile_valid", 0 <= q <= 1,
           f"q_hat={q}")

    psets = build_prediction_sets(proba, q, classes)
    report("conformal/sets_not_empty", all(len(s) > 0 for s in psets),
           "Empty prediction set found")

    eval_res = evaluate_coverage(psets, y_true_str)
    report("conformal/coverage_range", 0 <= eval_res["empirical_coverage"] <= 1,
           f"Coverage: {eval_res['empirical_coverage']}")


def test_no_data_leakage():
    """Test: verify test timestamps are strictly after training timestamps."""
    from data_layer import generate_synthetic_dataset, get_temporal_splits

    df = generate_synthetic_dataset(seed=42)
    train, calib, test = get_temporal_splits(df)

    report("leakage/test_after_train",
           test["timestamp"].min() > train["timestamp"].max(),
           f"test_min={test['timestamp'].min()}, train_max={train['timestamp'].max()}")

    report("leakage/test_after_calib",
           test["timestamp"].min() > calib["timestamp"].max(),
           f"test_min={test['timestamp'].min()}, calib_max={calib['timestamp'].max()}")


def test_reproducibility():
    """Test: same seed produces identical datasets."""
    from data_layer import generate_synthetic_dataset

    df1 = generate_synthetic_dataset(seed=42)
    df2 = generate_synthetic_dataset(seed=42)
    report("reproducibility/identical_datasets",
           df1.equals(df2),
           "Datasets differ with same seed")


if __name__ == "__main__":
    print("=" * 60)
    print("  CET-KG Test Suite")
    print("=" * 60)

    print("\n[1] Data Loading Tests")
    test_data_loading()

    print("\n[2] Temporal Split Tests")
    test_temporal_splits()

    print("\n[3] Graph Construction Tests")
    test_graph_construction()

    print("\n[4] Conformal Coverage Tests")
    test_conformal_coverage()

    print("\n[5] No Data Leakage Tests")
    test_no_data_leakage()

    print("\n[6] Reproducibility Tests")
    test_reproducibility()

    print("\n" + "=" * 60)
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)

    sys.exit(1 if FAIL > 0 else 0)
