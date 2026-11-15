"""
CET-KG Data Layer
=================
Generates, loads, validates, and splits a synthetic temporal knowledge graph
dataset modelled on simplified ICEWS/GDELT-style geopolitical events.

The dataset is explicitly SYNTHETIC — it is not real-world benchmark data.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENTITIES = [
    "USA", "Russia", "China", "India", "UK", "France", "Germany",
    "Japan", "Brazil", "Iran", "Israel", "Egypt", "Turkey",
    "Saudi_Arabia", "Pakistan",
]

RELATIONS = [
    "Criticize", "Cooperate", "Threaten", "Negotiate", "Sanction",
    "Aid", "Protest", "Attack", "Agree", "Visit",
]

REQUIRED_COLUMNS = [
    "timestamp", "head_entity", "relation", "tail_entity", "event_intensity",
]

# Temporal split boundaries (day indices, inclusive)
TRAIN_END = 249       # days 0-249
CALIB_START = 250
CALIB_END = 299       # days 250-299
TEST_START = 300
TEST_END = 364        # days 300-364

# ---------------------------------------------------------------------------
# Pair-relation affinity matrix  (makes the synthetic data non-uniform)
# ---------------------------------------------------------------------------

_PAIR_RELATION_AFFINITIES: Dict[Tuple[str, str], Dict[str, float]] = {
    ("USA", "Russia"):   {"Criticize": 3.0, "Sanction": 2.5, "Threaten": 2.0, "Negotiate": 1.5},
    ("USA", "China"):    {"Criticize": 2.0, "Negotiate": 2.0, "Cooperate": 1.5, "Sanction": 1.5},
    ("USA", "UK"):       {"Cooperate": 4.0, "Agree": 3.0, "Visit": 2.0},
    ("Russia", "China"): {"Cooperate": 3.0, "Agree": 2.0, "Visit": 1.5},
    ("India", "Pakistan"):{"Threaten": 3.0, "Criticize": 2.5, "Attack": 1.5, "Protest": 1.5},
    ("Israel", "Iran"):  {"Threaten": 3.5, "Attack": 2.0, "Criticize": 2.0, "Sanction": 1.5},
    ("Saudi_Arabia", "Iran"): {"Criticize": 2.5, "Threaten": 2.0, "Sanction": 1.5},
    ("France", "Germany"):{"Cooperate": 3.5, "Agree": 3.0, "Visit": 2.0, "Negotiate": 1.5},
    ("USA", "Israel"):   {"Aid": 3.0, "Cooperate": 2.5, "Visit": 2.0, "Agree": 2.0},
    ("Turkey", "Russia"): {"Negotiate": 2.0, "Cooperate": 1.5, "Criticize": 1.5},
    ("Egypt", "Israel"):  {"Negotiate": 2.0, "Criticize": 1.5, "Cooperate": 1.0},
    ("Brazil", "USA"):    {"Cooperate": 2.0, "Visit": 1.5, "Negotiate": 1.5},
    ("Japan", "China"):   {"Criticize": 1.5, "Negotiate": 2.0, "Cooperate": 1.5},
}

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    seed: int = 42,
    n_days: int = 365,
    base_events_per_day: float = 10.0,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Generate a synthetic temporal knowledge graph dataset.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_days : int
        Number of days to simulate (0 .. n_days-1).
    base_events_per_day : float
        Average number of events per day (before temporal modulation).
    save_path : str or None
        If given, save CSV to this path.

    Returns
    -------
    pd.DataFrame with columns defined by REQUIRED_COLUMNS.
    """
    rng = np.random.RandomState(seed)
    records: List[Dict] = []

    # Pre-compute temporal modulation: a smooth varying rate
    # with a couple of "burst" periods and a quiet period.
    day_rate = np.ones(n_days) * base_events_per_day
    # Burst around day 60-80 (crisis 1)
    day_rate[60:80] *= 1.8
    # Quiet around day 130-160
    day_rate[130:160] *= 0.5
    # Burst around day 200-220 (crisis 2)
    day_rate[200:220] *= 2.0
    # Moderate escalation in test period
    day_rate[310:340] *= 1.4
    # Add sinusoidal weekly pattern
    weekly = 1.0 + 0.2 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    day_rate = day_rate * weekly

    # All ordered entity pairs
    all_pairs = [(h, t) for h in ENTITIES for t in ENTITIES if h != t]

    for day in range(n_days):
        n_events = rng.poisson(day_rate[day])
        for _ in range(n_events):
            # Pick a pair — biased toward pairs with known affinities
            if rng.random() < 0.55:
                # Sample from affinity pairs
                affinity_pairs = list(_PAIR_RELATION_AFFINITIES.keys())
                idx = rng.randint(len(affinity_pairs))
                head, tail = affinity_pairs[idx]
                # Maybe reverse direction
                if rng.random() < 0.35:
                    head, tail = tail, head
            else:
                idx = rng.randint(len(all_pairs))
                head, tail = all_pairs[idx]

            # Pick relation — use affinity weights if available
            key_fwd = (head, tail)
            key_rev = (tail, head)
            affinity = _PAIR_RELATION_AFFINITIES.get(
                key_fwd, _PAIR_RELATION_AFFINITIES.get(key_rev, None)
            )
            if affinity is not None:
                weights = np.array([affinity.get(r, 0.3) for r in RELATIONS])
            else:
                weights = np.ones(len(RELATIONS))
            weights /= weights.sum()
            rel = rng.choice(RELATIONS, p=weights)

            # Intensity: base from relation type + noise
            hostile = {"Criticize", "Threaten", "Sanction", "Attack", "Protest"}
            base_intensity = 0.7 if rel in hostile else 0.4
            intensity = np.clip(base_intensity + rng.normal(0, 0.15), 0.01, 0.99)

            records.append({
                "timestamp": int(day),
                "head_entity": head,
                "relation": rel,
                "tail_entity": tail,
                "event_intensity": round(float(intensity), 4),
            })

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"[data_layer] Saved {len(df)} events to {save_path}")

    return df


# ---------------------------------------------------------------------------
# Loading & Validation
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    """Load a previously saved TKG event CSV."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    return df


def validate_dataset(df: pd.DataFrame) -> Dict[str, object]:
    """Validate the dataset schema and contents.

    Returns a dict with validation results.
    Raises ValueError on critical issues.
    """
    results: Dict[str, object] = {"valid": True, "warnings": []}

    # Check columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for nulls
    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Null values found:\n{null_counts[null_counts > 0]}")

    # Check timestamp range
    if df["timestamp"].min() < 0:
        results["warnings"].append("Negative timestamps found")
    if not pd.api.types.is_integer_dtype(df["timestamp"]):
        results["warnings"].append("Timestamps are not integer type")

    # Check entity validity
    unknown_heads = set(df["head_entity"]) - set(ENTITIES)
    unknown_tails = set(df["tail_entity"]) - set(ENTITIES)
    if unknown_heads:
        results["warnings"].append(f"Unknown head entities: {unknown_heads}")
    if unknown_tails:
        results["warnings"].append(f"Unknown tail entities: {unknown_tails}")

    # Check self-loops
    self_loops = (df["head_entity"] == df["tail_entity"]).sum()
    if self_loops > 0:
        results["warnings"].append(f"{self_loops} self-loop events found")

    # Check intensity range
    out_of_range = ((df["event_intensity"] < 0) | (df["event_intensity"] > 1)).sum()
    if out_of_range > 0:
        results["warnings"].append(f"{out_of_range} intensity values outside [0,1]")

    results["n_events"] = len(df)
    results["n_entities"] = len(set(df["head_entity"]) | set(df["tail_entity"]))
    results["n_relations"] = df["relation"].nunique()
    results["timestamp_range"] = (int(df["timestamp"].min()), int(df["timestamp"].max()))

    return results


# ---------------------------------------------------------------------------
# Temporal Splits
# ---------------------------------------------------------------------------

def get_temporal_splits(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically into train, calibration, and test.

    Train:       days 0 .. 249
    Calibration: days 250 .. 299
    Test:        days 300 .. 364
    """
    train = df[df["timestamp"] <= TRAIN_END].copy()
    calib = df[(df["timestamp"] >= CALIB_START) & (df["timestamp"] <= CALIB_END)].copy()
    test  = df[(df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)].copy()

    print(f"[data_layer] Split sizes — train: {len(train)}, calib: {len(calib)}, test: {len(test)}")
    return train, calib, test


# ---------------------------------------------------------------------------
# CLI entry point for standalone generation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic TKG dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/synthetic_tkg_events.csv")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(project_root, args.output)

    df = generate_synthetic_dataset(seed=args.seed, save_path=out_path)
    info = validate_dataset(df)
    print(f"[data_layer] Validation: {info}")
    train, calib, test = get_temporal_splits(df)
