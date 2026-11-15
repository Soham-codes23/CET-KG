"""
CET-KG Temporal Graph Layer
============================
Builds and queries a time-aware directed multigraph using NetworkX.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter
import math


def build_temporal_graph(events_df: pd.DataFrame, max_timestamp: Optional[int] = None) -> nx.MultiDiGraph:
    """Build a temporal multigraph from event records.

    Each event becomes a directed edge (head → tail) carrying
    relation, timestamp, and intensity attributes.

    Parameters
    ----------
    events_df : pd.DataFrame
        Must have columns: head_entity, tail_entity, relation, timestamp, event_intensity
    max_timestamp : int or None
        If given, only include events with timestamp <= max_timestamp.

    Returns
    -------
    nx.MultiDiGraph
    """
    G = nx.MultiDiGraph()

    df = events_df if max_timestamp is None else events_df[events_df["timestamp"] <= max_timestamp]

    # Add all entities as nodes
    all_entities = set(df["head_entity"]) | set(df["tail_entity"])
    for entity in sorted(all_entities):
        G.add_node(entity)

    # Add edges
    for _, row in df.iterrows():
        G.add_edge(
            row["head_entity"],
            row["tail_entity"],
            relation=row["relation"],
            timestamp=int(row["timestamp"]),
            intensity=float(row["event_intensity"]),
        )

    return G


def get_historical_subgraph(
    graph: nx.MultiDiGraph,
    entity: str,
    timestamp: int,
    window: int = 30,
) -> nx.MultiDiGraph:
    """Return edges from [timestamp-window, timestamp) involving *entity*.

    Parameters
    ----------
    graph : nx.MultiDiGraph
    entity : str
    timestamp : int
        Current time (exclusive upper bound).
    window : int
        Look-back window in days.

    Returns
    -------
    nx.MultiDiGraph  (subgraph)
    """
    t_min = timestamp - window
    sub = nx.MultiDiGraph()
    sub.add_node(entity)

    for u, v, key, data in graph.edges(keys=True, data=True):
        t = data.get("timestamp", -1)
        if t_min <= t < timestamp and (u == entity or v == entity):
            sub.add_node(u)
            sub.add_node(v)
            sub.add_edge(u, v, key=key, **data)

    return sub


def get_entity_neighborhood(
    graph: nx.MultiDiGraph,
    entity: str,
    timestamp: int,
    hops: int = 2,
    window: int = 30,
) -> nx.MultiDiGraph:
    """Multi-hop temporal neighborhood around *entity*.

    First finds 1-hop neighbors from the temporal window, then expands
    outward for the requested number of hops.
    """
    t_min = timestamp - window
    visited: Set[str] = {entity}
    frontier: Set[str] = {entity}
    sub = nx.MultiDiGraph()
    sub.add_node(entity)

    for _ in range(hops):
        next_frontier: Set[str] = set()
        for node in frontier:
            for u, v, key, data in graph.edges(keys=True, data=True):
                t = data.get("timestamp", -1)
                if t_min <= t < timestamp and (u == node or v == node):
                    sub.add_node(u)
                    sub.add_node(v)
                    sub.add_edge(u, v, key=key, **data)
                    if u not in visited:
                        next_frontier.add(u)
                    if v not in visited:
                        next_frontier.add(v)
        visited |= next_frontier
        frontier = next_frontier
        if not frontier:
            break

    return sub


def compute_temporal_features(
    graph: nx.MultiDiGraph,
    entity: str,
    timestamp: int,
    window: int = 30,
) -> Dict[str, float]:
    """Compute temporal features for *entity* using events in [timestamp-window, timestamp).

    Returns dict with keys:
        in_degree, out_degree, unique_relations, unique_neighbors,
        avg_intensity, event_count, relation_entropy
    """
    t_min = timestamp - window
    in_events: List[Dict] = []
    out_events: List[Dict] = []

    for u, v, data in graph.edges(data=True):
        t = data.get("timestamp", -1)
        if t_min <= t < timestamp:
            if v == entity:
                in_events.append(data)
            if u == entity:
                out_events.append(data)

    all_events = in_events + out_events
    event_count = len(all_events)

    if event_count == 0:
        return {
            "in_degree": 0.0,
            "out_degree": 0.0,
            "unique_relations": 0.0,
            "unique_neighbors": 0.0,
            "avg_intensity": 0.0,
            "event_count": 0.0,
            "relation_entropy": 0.0,
        }

    # Collect neighbor set
    neighbors: Set[str] = set()
    for u, v, data in graph.edges(data=True):
        t = data.get("timestamp", -1)
        if t_min <= t < timestamp:
            if u == entity:
                neighbors.add(v)
            elif v == entity:
                neighbors.add(u)

    relations = [e.get("relation", "") for e in all_events]
    intensities = [e.get("intensity", 0.0) for e in all_events]

    # Relation entropy
    rel_counts = Counter(relations)
    total = sum(rel_counts.values())
    entropy = 0.0
    for c in rel_counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)

    return {
        "in_degree": float(len(in_events)),
        "out_degree": float(len(out_events)),
        "unique_relations": float(len(set(relations))),
        "unique_neighbors": float(len(neighbors)),
        "avg_intensity": float(np.mean(intensities)),
        "event_count": float(event_count),
        "relation_entropy": float(entropy),
    }


def compute_pair_features(
    graph: nx.MultiDiGraph,
    head: str,
    tail: str,
    timestamp: int,
    relations: List[str],
    window: int = 60,
) -> Dict[str, float]:
    """Compute pair-specific features between head and tail.

    Returns dict with:
        - freq_<relation> for each relation
        - recency (days since last interaction, 0 if never)
        - total_interactions
    """
    t_min = timestamp - window
    pair_events: List[Dict] = []

    for u, v, data in graph.edges(data=True):
        t = data.get("timestamp", -1)
        if t_min <= t < timestamp:
            if (u == head and v == tail) or (u == tail and v == head):
                pair_events.append(data)

    features: Dict[str, float] = {}

    # Per-relation frequency
    rel_counter = Counter(e.get("relation", "") for e in pair_events)
    for r in relations:
        features[f"freq_{r}"] = float(rel_counter.get(r, 0))

    # Recency
    if pair_events:
        most_recent = max(e.get("timestamp", 0) for e in pair_events)
        features["recency"] = float(timestamp - most_recent)
    else:
        features["recency"] = float(window + 1)  # Never interacted

    features["total_interactions"] = float(len(pair_events))

    return features


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_layer import generate_synthetic_dataset, get_temporal_splits

    df = generate_synthetic_dataset(seed=42)
    train, calib, test = get_temporal_splits(df)

    G = build_temporal_graph(train)
    print(f"[temporal_graph] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    sample_entity = "USA"
    feats = compute_temporal_features(G, sample_entity, timestamp=200, window=30)
    print(f"[temporal_graph] Features for {sample_entity} at t=200: {feats}")
