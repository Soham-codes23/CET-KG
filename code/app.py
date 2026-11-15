"""
CET-KG Interactive Dashboard
==============================
Streamlit application for exploring the CET-KG pipeline results.

Launch:
    streamlit run code/app.py
"""

import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "synthetic_tkg_events.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CET-KG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def file_exists(path):
    return os.path.isfile(path)


def check_pipeline_outputs():
    """Check which outputs exist."""
    return {
        "dataset": file_exists(DATA_PATH),
        "predictions": file_exists(os.path.join(RESULTS_DIR, "forecast_predictions.csv")),
        "metrics": file_exists(os.path.join(RESULTS_DIR, "forecast_metrics.json")),
        "conformal": file_exists(os.path.join(RESULTS_DIR, "conformal_results.json")),
        "explanations": file_exists(os.path.join(RESULTS_DIR, "explanations.json")),
        "noise": file_exists(os.path.join(RESULTS_DIR, "noise_results.json")),
    }

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🔬 CET-KG")
st.sidebar.markdown("**Calibrated & Explainable Temporal Knowledge Graph Forecasting**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "📊 Dataset Explorer",
        "🕸️ Temporal Graph",
        "🎯 Forecasting Results",
        "📐 Conformal Prediction",
        "🔍 Explanation",
        "🔊 Noise Robustness",
        "⚙️ Run Pipeline",
    ],
)

st.sidebar.markdown("---")
status = check_pipeline_outputs()
st.sidebar.markdown("**Pipeline Status**")
for key, exists in status.items():
    icon = "✅" if exists else "❌"
    st.sidebar.markdown(f"{icon} {key}")

# ===================================================================
# PAGE: Overview
# ===================================================================
if page == "🏠 Overview":
    st.title("CET-KG: Calibrated & Explainable Temporal KG Forecasting")

    st.warning(
        "⚠️ **SYNTHETIC PROOF-OF-CONCEPT** — All results shown are generated from "
        "synthetic data. This is not benchmark validation on real-world datasets."
    )

    st.markdown("""
    ### Research Question
    > *Can calibrated uncertainty, graph-grounded explanations, and noise robustness
    > be integrated into a temporal knowledge graph forecasting pipeline without
    > substantially degrading predictive performance?*

    ### Pipeline Architecture
    ```
    Synthetic Geopolitical Events (3000+ events, 15 countries, 10 relation types)
            ↓
    Temporal Knowledge Graph (NetworkX MultiDiGraph)
            ↓
    Feature Extraction (entity + pair temporal features)
            ↓
    Multi-class Relation Forecasting (RandomForest)
            ↓
    Split-Conformal Calibration (prediction sets with coverage guarantee)
            ↓
    Graph-Grounded Evidence Explanation (heuristic relevance scoring)
            ↓
    Noise Robustness Evaluation (intensity noise + label noise)
    ```
    """)

    st.markdown("### Implementation Status")
    status_data = {
        "Component": [
            "Data Layer", "Temporal Graph", "Forecasting",
            "Conformal Calibration", "Graph-Grounded Explanation",
            "Noise Robustness", "Interactive Dashboard",
            "Real Benchmark Validation",
        ],
        "Status": [
            "✅ IMPLEMENTED", "✅ IMPLEMENTED", "✅ IMPLEMENTED",
            "✅ IMPLEMENTED", "✅ IMPLEMENTED",
            "✅ IMPLEMENTED", "✅ IMPLEMENTED",
            "📋 PLANNED",
        ],
        "Notes": [
            "Synthetic geopolitical TKG events",
            "NetworkX MultiDiGraph with temporal filtering",
            "RandomForest multi-class, temporal features, no data leakage",
            "Split-conformal, configurable coverage, exchangeability caveat",
            "Evidence-based prototype (not formal attribution)",
            "Historical noise + label noise experiments",
            "Streamlit dashboard (this page)",
            "Awaiting real TKG datasets (ICEWS/GDELT/YAGO)",
        ],
    }
    st.dataframe(pd.DataFrame(status_data), use_container_width=True, hide_index=True)

# ===================================================================
# PAGE: Dataset Explorer
# ===================================================================
elif page == "📊 Dataset Explorer":
    st.title("📊 Dataset Explorer")

    if not status["dataset"]:
        st.error("Dataset not found. Run the pipeline first (⚙️ Run Pipeline).")
        st.stop()

    df = pd.read_csv(DATA_PATH)

    st.info(f"📁 Loaded **{len(df):,}** events from `data/synthetic_tkg_events.csv`  \n"
            f"⚠️ This is **synthetic** data, not a real-world benchmark.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Events", f"{len(df):,}")
    col2.metric("Entities", df["head_entity"].nunique() + df["tail_entity"].nunique())
    col3.metric("Relations", df["relation"].nunique())
    col4.metric("Day Range", f"{df['timestamp'].min()}–{df['timestamp'].max()}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 Temporal Distribution", "📊 Relations", "🔍 Data Preview"])

    with tab1:
        fig_path = os.path.join(FIGURES_DIR, "temporal_event_distribution.png")
        if file_exists(fig_path):
            st.image(fig_path, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(14, 4))
            counts = df.groupby("timestamp").size()
            ax.bar(counts.index, counts.values, width=1.0, color="#1976D2", alpha=0.7)
            ax.axvline(x=249.5, color="#F44336", ls="--", lw=1.5, label="Train|Calib")
            ax.axvline(x=299.5, color="#FF9800", ls="--", lw=1.5, label="Calib|Test")
            ax.set_xlabel("Day"); ax.set_ylabel("Events")
            ax.set_title("Temporal Event Distribution"); ax.legend()
            st.pyplot(fig)
            plt.close(fig)

    with tab2:
        fig_path = os.path.join(FIGURES_DIR, "relation_distribution.png")
        if file_exists(fig_path):
            st.image(fig_path, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            counts = df["relation"].value_counts().sort_values()
            ax.barh(counts.index, counts.values, color="#42A5F5")
            ax.set_xlabel("Count"); ax.set_title("Relation Distribution")
            st.pyplot(fig); plt.close(fig)

    with tab3:
        rel_filter = st.multiselect("Filter by relation", df["relation"].unique().tolist())
        filtered = df[df["relation"].isin(rel_filter)] if rel_filter else df
        st.dataframe(filtered.head(200), use_container_width=True, hide_index=True)

# ===================================================================
# PAGE: Temporal Graph
# ===================================================================
elif page == "🕸️ Temporal Graph":
    st.title("🕸️ Temporal Graph Explorer")

    if not status["dataset"]:
        st.error("Dataset not found. Run the pipeline first.")
        st.stop()

    df = pd.read_csv(DATA_PATH)

    from temporal_graph import build_temporal_graph, get_historical_subgraph

    @st.cache_resource
    def get_full_graph():
        return build_temporal_graph(pd.read_csv(DATA_PATH))

    G = get_full_graph()

    col1, col2, col3 = st.columns(3)
    col1.metric("Nodes", G.number_of_nodes())
    col2.metric("Edges", G.number_of_edges())
    col3.metric("Density", f"{nx.density(G):.4f}")

    st.markdown("---")

    entity = st.selectbox("Select Entity", sorted(G.nodes()))
    timestamp = st.slider("Timestamp", 0, 364, 250)
    window = st.slider("Look-back Window (days)", 5, 60, 30)

    sub = get_historical_subgraph(G, entity, timestamp, window)
    st.info(f"Subgraph for **{entity}** at day {timestamp} (window={window}): "
            f"**{sub.number_of_nodes()}** nodes, **{sub.number_of_edges()}** edges")

    if sub.number_of_edges() > 0:
        fig, ax = plt.subplots(figsize=(10, 7))
        pos = nx.spring_layout(sub, seed=42, k=2.0)
        colors = ["#2196F3" if n == entity else "#90CAF9" for n in sub.nodes()]
        nx.draw_networkx_nodes(sub, pos, ax=ax, node_color=colors, node_size=700,
                               edgecolors="#1565C0", linewidths=2)
        nx.draw_networkx_labels(sub, pos, ax=ax, font_size=8, font_weight="bold")
        nx.draw_networkx_edges(sub, pos, ax=ax, edge_color="#666", alpha=0.5,
                               arrows=True, arrowsize=12)
        ax.set_title(f"Temporal Neighborhood of {entity} (days {timestamp-window}–{timestamp})")
        ax.set_axis_off()
        st.pyplot(fig)
        plt.close(fig)

        # Edge details
        edge_data = []
        for u, v, data in sub.edges(data=True):
            edge_data.append({"from": u, "to": v, "relation": data.get("relation",""),
                              "day": data.get("timestamp",""), "intensity": round(data.get("intensity",0),3)})
        st.dataframe(pd.DataFrame(edge_data), use_container_width=True, hide_index=True)
    else:
        st.warning("No edges found in this window.")

# ===================================================================
# PAGE: Forecasting Results
# ===================================================================
elif page == "🎯 Forecasting Results":
    st.title("🎯 Forecasting Results")

    if not status["predictions"]:
        st.error("No predictions found. Run the pipeline first.")
        st.stop()

    preds = pd.read_csv(os.path.join(RESULTS_DIR, "forecast_predictions.csv"))
    metrics = load_json(os.path.join(RESULTS_DIR, "forecast_metrics.json"))

    st.warning("⚠️ Results are from **synthetic** data. Not benchmark validation.")

    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        col2.metric("Macro-F1", f"{metrics['macro_f1']:.4f}")
        col3.metric("Top-3 Accuracy", f"{metrics['top3_accuracy']:.4f}")
        col4.metric("Test Samples", f"{metrics['n_test_samples']:,}")

    st.markdown("---")
    tab1, tab2 = st.tabs(["📊 Confusion Matrix", "🔍 Predictions"])

    with tab1:
        cm_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
        if file_exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            st.info("Confusion matrix figure not generated yet.")

    with tab2:
        show_correct = st.checkbox("Show only correct predictions", value=False)
        display = preds[preds["correct"]] if show_correct else preds
        st.dataframe(display.head(100), use_container_width=True, hide_index=True)

# ===================================================================
# PAGE: Conformal Prediction
# ===================================================================
elif page == "📐 Conformal Prediction":
    st.title("📐 Conformal Prediction Sets")

    if not status["conformal"]:
        st.error("No conformal results found. Run the pipeline first.")
        st.stop()

    conf = load_json(os.path.join(RESULTS_DIR, "conformal_results.json"))

    st.warning("⚠️ Results from **synthetic** data. Coverage guarantee is approximate under temporal dependence.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Target Coverage", f"{conf['target_coverage']:.0%}")
    col2.metric("Empirical Coverage", f"{conf['empirical_coverage']:.1%}",
                delta=f"{(conf['empirical_coverage'] - conf['target_coverage'])*100:+.1f}pp")
    col3.metric("Avg Set Size", f"{conf['avg_set_size']:.2f}")
    col4.metric("Quantile (q̂)", f"{conf['q_hat']:.4f}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 Coverage by Time", "📊 Set Sizes", "⚠️ Assumptions"])

    with tab1:
        cov_path = os.path.join(FIGURES_DIR, "coverage_by_time.png")
        if file_exists(cov_path):
            st.image(cov_path, use_container_width=True)
        else:
            st.info("Coverage plot not generated yet.")

    with tab2:
        sz_path = os.path.join(FIGURES_DIR, "prediction_set_sizes.png")
        if file_exists(sz_path):
            st.image(sz_path, use_container_width=True)
        else:
            st.info("Set size plot not generated yet.")

    with tab3:
        st.markdown(f"""
        ### Exchangeability Caveat

        {conf.get('exchangeability_caveat', 'N/A')}

        ### What this means
        - The split-conformal method computes a threshold on a **calibration set** (days 250–299).
        - Prediction sets are constructed on the **test set** (days 300–364) by including all
          classes with predicted probability ≥ 1 − q̂.
        - If the data distribution shifts between calibration and test, empirical coverage may
          differ from the target.
        - This is a well-known limitation of conformal prediction under non-exchangeability
          (Barber et al., 2023).
        """)

# ===================================================================
# PAGE: Explanation
# ===================================================================
elif page == "🔍 Explanation":
    st.title("🔍 Graph-Grounded Explanation")

    if not status["explanations"]:
        st.error("No explanations found. Run the pipeline first.")
        st.stop()

    explanations = load_json(os.path.join(RESULTS_DIR, "explanations.json"))

    st.info("🏷️ **Evidence-based prototype explanation** — Relevance scores are heuristic "
            "(entity overlap + relation match + temporal recency), not model-gradient-based attribution.")

    if not explanations:
        st.warning("No explanations available.")
        st.stop()

    options = [
        f"{e['prediction']['head']} --[{e['prediction']['relation']}]--> "
        f"{e['prediction']['tail']} @ day {e['prediction']['timestamp']}"
        for e in explanations
    ]
    idx = st.selectbox("Select prediction to explain", range(len(options)), format_func=lambda i: options[i])
    expl = explanations[idx]

    st.markdown(f"### Prediction")
    pred = expl["prediction"]
    st.markdown(f"**{pred['head']}** —[{pred['relation']}]→ **{pred['tail']}** at day **{pred['timestamp']}**")

    st.markdown("### Explanation")
    st.code(expl["explanation_text"])

    st.markdown("### Evidence Events")
    if expl["evidence_events"]:
        ev_df = pd.DataFrame(expl["evidence_events"])
        ev_df = ev_df.sort_values("relevance_score", ascending=False)
        st.dataframe(ev_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No evidence events found.")

    # Show subgraph figure if available
    fig_path = os.path.join(FIGURES_DIR, "sample_explanation_subgraph.png")
    if idx == 0 and file_exists(fig_path):
        st.markdown("### Evidence Subgraph")
        st.image(fig_path, use_container_width=True)

# ===================================================================
# PAGE: Noise Robustness
# ===================================================================
elif page == "🔊 Noise Robustness":
    st.title("🔊 Noise Robustness Analysis")

    if not status["noise"]:
        st.warning("Noise results not found. Run the pipeline **without** `--skip-noise`.")
        st.stop()

    noise = load_json(os.path.join(RESULTS_DIR, "noise_results.json"))

    st.info("Comparing clean baseline vs. noisy conditions on synthetic data.")

    fig_path = os.path.join(FIGURES_DIR, "noise_robustness.png")
    if file_exists(fig_path):
        st.image(fig_path, use_container_width=True)

    tab1, tab2 = st.tabs(["📊 Intensity Noise", "🏷️ Label Noise"])

    with tab1:
        st.markdown("**Intensity Noise** — Gaussian perturbation of `event_intensity` values.")
        int_data = noise.get("intensity_noise", [])
        if int_data:
            st.dataframe(pd.DataFrame(int_data), use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("**Label Noise** — Random flipping of relation labels in training data.")
        lbl_data = noise.get("label_noise", [])
        if lbl_data:
            st.dataframe(pd.DataFrame(lbl_data), use_container_width=True, hide_index=True)

# ===================================================================
# PAGE: Run Pipeline
# ===================================================================
elif page == "⚙️ Run Pipeline":
    st.title("⚙️ Run Pipeline")

    st.markdown("""
    Run the CET-KG pipeline to generate all results.

    **Commands:**
    ```bash
    # Quick run (skip noise experiments):
    MPLCONFIGDIR=/tmp/mpl_cache python3 code/run_pipeline.py --seed 42 --target-coverage 0.90 --skip-noise

    # Full run (with noise experiments):
    MPLCONFIGDIR=/tmp/mpl_cache python3 code/run_pipeline.py --seed 42 --target-coverage 0.90
    ```
    """)

    col1, col2 = st.columns(2)
    seed = col1.number_input("Seed", value=42, min_value=0)
    coverage = col2.number_input("Target Coverage", value=0.90, min_value=0.5, max_value=0.99, step=0.05)
    skip_noise = st.checkbox("Skip noise experiments (faster)", value=True)

    if st.button("🚀 Run Pipeline", type="primary"):
        cmd = [
            sys.executable, os.path.join(SCRIPT_DIR, "run_pipeline.py"),
            "--seed", str(seed),
            "--target-coverage", str(coverage),
        ]
        if skip_noise:
            cmd.append("--skip-noise")

        env = os.environ.copy()
        env["MPLCONFIGDIR"] = "/tmp/mpl_cache"

        with st.spinner("Running pipeline... This may take 1–3 minutes."):
            result = subprocess.run(cmd, capture_output=True, text=True, env=env,
                                    cwd=PROJECT_ROOT, timeout=600)

        if result.returncode == 0:
            st.success("✅ Pipeline completed successfully!")
            st.code(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
        else:
            st.error("❌ Pipeline failed.")
            st.code(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)

    st.markdown("---")
    st.markdown("### Current Output Files")
    for d, label in [(RESULTS_DIR, "results/"), (FIGURES_DIR, "figures/")]:
        if os.path.isdir(d):
            files = os.listdir(d)
            if files:
                st.markdown(f"**{label}**: {', '.join(sorted(files))}")
            else:
                st.markdown(f"**{label}**: *(empty)*")
