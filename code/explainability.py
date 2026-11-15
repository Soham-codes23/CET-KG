"""
CET-KG Graph-Grounded Explainability
======================================
Evidence-based prototype explanations for temporal knowledge graph forecasts.

For a given prediction (head, relation, tail, timestamp), this module retrieves
relevant historical events from the temporal graph and scores their relevance
as supporting evidence.

IMPORTANT LABEL:
    This is an **evidence-based prototype explanation**, not a formal
    attribution method (e.g. SHAP, Integrated Gradients, GNNExplainer).
    The relevance scores are heuristic, not derived from model gradients.
    They indicate which historical events are structurally and temporally
    proximate to the predicted event, not their causal contribution to
    the model's output.

Status: IMPLEMENTED (evidence-based prototype).
"""

import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


def score_evidence_event(
    evidence: Dict,
    predicted_head: str,
    predicted_tail: str,
    predicted_relation: str,
    predicted_timestamp: int,
    max_window: int = 60,
) -> float:
    """Score the relevance of a single historical event to the prediction.

    Heuristic scoring based on:
    - Entity overlap (head/tail match)
    - Relation match
    - Temporal recency
    - Pair match (same head-tail pair)

    Returns a relevance score in [0, 1].
    """
    score = 0.0

    ev_head = evidence.get("head_entity", "")
    ev_tail = evidence.get("tail_entity", "")
    ev_rel = evidence.get("relation", "")
    ev_time = evidence.get("timestamp", 0)

    # Entity overlap: +0.2 per entity match
    if ev_head == predicted_head or ev_tail == predicted_head:
        score += 0.2
    if ev_head == predicted_tail or ev_tail == predicted_tail:
        score += 0.2

    # Exact pair match: +0.25
    if (ev_head == predicted_head and ev_tail == predicted_tail) or \
       (ev_head == predicted_tail and ev_tail == predicted_head):
        score += 0.25

    # Relation match: +0.2
    if ev_rel == predicted_relation:
        score += 0.2

    # Temporal recency: decays linearly, max +0.15
    time_diff = predicted_timestamp - ev_time
    if 0 < time_diff <= max_window:
        recency = 1.0 - (time_diff / max_window)
        score += 0.15 * recency

    return min(score, 1.0)


def generate_explanation(
    graph: nx.MultiDiGraph,
    predicted_head: str,
    predicted_relation: str,
    predicted_tail: str,
    predicted_timestamp: int,
    window: int = 60,
    top_k: int = 10,
) -> Dict[str, object]:
    """Generate an evidence-based explanation for a prediction.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        Full temporal graph.
    predicted_head, predicted_relation, predicted_tail : str
        The predicted event.
    predicted_timestamp : int
        The timestamp of the prediction.
    window : int
        How far back to look for evidence.
    top_k : int
        Maximum number of evidence events to return.

    Returns
    -------
    dict with:
        prediction: dict
        evidence_events: list of dicts with relevance scores
        explanation_text: str
        method_label: str
    """
    t_min = predicted_timestamp - window

    # Collect candidate evidence events
    candidates: List[Dict] = []
    for u, v, data in graph.edges(data=True):
        t = data.get("timestamp", -1)
        if t_min <= t < predicted_timestamp:
            # Must involve at least one of the predicted entities
            if u in (predicted_head, predicted_tail) or v in (predicted_head, predicted_tail):
                candidates.append({
                    "head_entity": u,
                    "tail_entity": v,
                    "relation": data.get("relation", ""),
                    "timestamp": t,
                    "intensity": data.get("intensity", 0.0),
                })

    # Score candidates
    for ev in candidates:
        ev["relevance_score"] = score_evidence_event(
            ev, predicted_head, predicted_tail, predicted_relation,
            predicted_timestamp, max_window=window,
        )

    # Sort by relevance (descending) then recency (descending)
    candidates.sort(key=lambda e: (-e["relevance_score"], -(e["timestamp"])))
    top_evidence = candidates[:top_k]

    # Generate explanation text
    explanation_text = _generate_explanation_text(
        predicted_head, predicted_relation, predicted_tail,
        predicted_timestamp, top_evidence,
    )

    return {
        "prediction": {
            "head": predicted_head,
            "relation": predicted_relation,
            "tail": predicted_tail,
            "timestamp": predicted_timestamp,
        },
        "evidence_events": top_evidence,
        "explanation_text": explanation_text,
        "n_candidates_considered": len(candidates),
        "method_label": (
            "Evidence-based prototype explanation. Relevance scores are "
            "heuristic (entity overlap + relation match + temporal recency), "
            "not model-gradient-based attribution."
        ),
    }


def _generate_explanation_text(
    head: str, relation: str, tail: str, timestamp: int,
    evidence: List[Dict],
) -> str:
    """Produce a human-readable explanation paragraph."""
    lines = [
        f"Prediction: {head} --[{relation}]--> {tail} at day {timestamp}",
        "",
        "Supporting temporal evidence:",
    ]

    if not evidence:
        lines.append("  No relevant historical events found in the look-back window.")
    else:
        # Summarize
        pair_events = [e for e in evidence
                       if (e["head_entity"] == head and e["tail_entity"] == tail) or
                          (e["head_entity"] == tail and e["tail_entity"] == head)]
        same_rel = [e for e in evidence if e["relation"] == relation]

        if pair_events:
            lines.append(
                f"  • {len(pair_events)} recent event(s) between {head} and {tail} "
                f"(most recent at day {max(e['timestamp'] for e in pair_events)})."
            )
        if same_rel:
            lines.append(
                f"  • {len(same_rel)} recent '{relation}' event(s) involving "
                f"these entities."
            )

        lines.append("")
        lines.append("Top evidence events (by relevance):")
        for i, ev in enumerate(evidence[:5], 1):
            lines.append(
                f"  {i}. Day {ev['timestamp']}: {ev['head_entity']} "
                f"--[{ev['relation']}]--> {ev['tail_entity']}  "
                f"(relevance: {ev['relevance_score']:.2f})"
            )

    return "\n".join(lines)


def visualize_explanation_subgraph(
    explanation: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """Visualize the evidence subgraph for an explanation.

    Nodes = entities involved in evidence events + prediction entities.
    Edges = evidence events, colored by relation, sized by relevance.
    """
    pred = explanation["prediction"]
    evidence = explanation["evidence_events"]

    G = nx.MultiDiGraph()

    # Add prediction entities
    G.add_node(pred["head"], node_type="prediction")
    G.add_node(pred["tail"], node_type="prediction")

    # Add evidence edges
    for ev in evidence:
        G.add_node(ev["head_entity"], node_type="evidence")
        G.add_node(ev["tail_entity"], node_type="evidence")
        G.add_edge(
            ev["head_entity"], ev["tail_entity"],
            relation=ev["relation"],
            relevance=ev["relevance_score"],
            timestamp=ev["timestamp"],
        )

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No evidence events found", ha="center", va="center",
                fontsize=14, transform=ax.transAxes)
        ax.set_axis_off()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    pos = nx.spring_layout(G, seed=42, k=2.0)

    # Node colors
    node_colors = []
    for node in G.nodes():
        if node in (pred["head"], pred["tail"]):
            node_colors.append("#2196F3")  # Blue for prediction entities
        else:
            node_colors.append("#90CAF9")  # Light blue for neighbors

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=800, alpha=0.9, edgecolors="#1565C0", linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="bold")

    # Draw edges with relation labels
    # Color edges by relation type
    relation_colors = {
        "Criticize": "#F44336", "Cooperate": "#4CAF50", "Threaten": "#FF9800",
        "Negotiate": "#9C27B0", "Sanction": "#795548", "Aid": "#00BCD4",
        "Protest": "#FF5722", "Attack": "#D32F2F", "Agree": "#8BC34A",
        "Visit": "#3F51B5",
    }

    edge_labels = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        rel = data.get("relation", "?")
        ts = data.get("timestamp", "?")
        relevance = data.get("relevance", 0)
        color = relation_colors.get(rel, "#666666")
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], ax=ax,
            edge_color=color, width=1 + 3 * relevance,
            alpha=0.5 + 0.5 * relevance,
            arrows=True, arrowsize=15,
            connectionstyle=f"arc3,rad={0.1 * (key % 3)}",
        )
        edge_labels[(u, v)] = f"{rel}\n(d{ts})"

    # Title
    ax.set_title(
        f"Evidence Subgraph: {pred['head']} --[{pred['relation']}]--> "
        f"{pred['tail']} @ day {pred['timestamp']}\n"
        f"(Evidence-based prototype explanation — not formal attribution)",
        fontsize=11, pad=15,
    )
    ax.set_axis_off()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[explainability] Saved explanation subgraph to {save_path}")

    return fig


def save_explanations(explanations: List[Dict], save_path: str) -> None:
    """Save explanation results to JSON."""
    # Make JSON-serializable
    out = []
    for expl in explanations:
        entry = {
            "prediction": expl["prediction"],
            "evidence_events": expl["evidence_events"],
            "explanation_text": expl["explanation_text"],
            "n_candidates_considered": expl["n_candidates_considered"],
            "method_label": expl["method_label"],
        }
        out.append(entry)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[explainability] Saved {len(out)} explanations to {save_path}")
