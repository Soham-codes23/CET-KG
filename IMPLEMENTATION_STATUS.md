# CET-KG Implementation Status

**Last Updated:** 2026-11-15

> ⚠️ **All results below are from SYNTHETIC data.**
> This is a proof-of-concept implementation, not benchmark validation
> on real-world temporal knowledge graph datasets.

---

## Component Status

| Component | Status | Implementation |
|-----------|--------|---------------|
| Data Layer | ✅ IMPLEMENTED | Synthetic geopolitical TKG events (3,953 events, 15 countries, 10 relations, 365 days) |
| Temporal Graph | ✅ IMPLEMENTED | NetworkX MultiDiGraph with temporal filtering, subgraph extraction, feature computation |
| Forecasting | ✅ IMPLEMENTED | RandomForest multi-class classification (26 temporal + pair features), no data leakage |
| Conformal Calibration | ✅ IMPLEMENTED | Split-conformal prediction sets, configurable coverage, exchangeability caveat documented |
| Graph-Grounded Explanation | ✅ IMPLEMENTED | Evidence-based prototype (heuristic relevance scoring, not gradient-based attribution) |
| Noise Robustness | ✅ IMPLEMENTED | Historical intensity noise + label noise, configurable levels |
| Interactive Dashboard | ✅ IMPLEMENTED | Streamlit app with 7 pages |
| Tests | ✅ IMPLEMENTED | 21 tests, all passing |
| Real Benchmark Validation | 📋 PLANNED | Requires ICEWS/GDELT/YAGO temporal KG datasets |

---

## Pipeline Results (seed=42, target_coverage=0.90)

### Forecasting Performance

| Metric | Value |
|--------|-------|
| Accuracy | 0.1662 |
| Macro-F1 | 0.1522 |
| Top-3 Accuracy | 0.5149 |
| Test Samples | 740 |
| Classes | 10 |

> **Note:** For a 10-class problem with lightweight features on synthetic data,
> random baseline would achieve ~10% accuracy. The model achieves 16.6%
> (66% above random) with 51.5% top-3 accuracy. Performance is expected to
> improve significantly with real benchmark data and richer features.

### Conformal Calibration

| Metric | Value |
|--------|-------|
| Target Coverage | 90.0% |
| Empirical Coverage | 92.2% |
| Average Set Size | 7.77 |
| Quantile (q̂) | 0.9495 |
| Coverage (early test) | 93.9% |
| Coverage (mid test) | 92.7% |
| Coverage (late test) | 89.9% |

> The empirical coverage (92.2%) exceeds the target (90%), demonstrating
> that the split-conformal method provides approximately valid coverage
> even under temporal dependence. The slight overcoverage is typical of
> finite-sample conformal methods.
>
> **Caveat:** Coverage is approximate, not guaranteed, because temporal
> KG events violate the exchangeability assumption.

### Explanations

5 sample predictions explained with evidence-based prototype method.
Evidence events scored by entity overlap, relation match, and temporal recency.

### Noise Robustness

Results in `results/noise_results.json` (run full pipeline without `--skip-noise`).

---

## File Structure

```
CET-KG (PBL PROJECT)/
├── code/
│   ├── data_layer.py          # Data generation, loading, validation, splits
│   ├── temporal_graph.py      # NetworkX graph construction and features
│   ├── forecasting.py         # RandomForest multi-class forecasting
│   ├── conformal.py           # Split-conformal calibration
│   ├── explainability.py      # Graph-grounded evidence explanation
│   ├── noise_eval.py          # Noise robustness experiments
│   ├── run_pipeline.py        # Main pipeline orchestrator
│   ├── app.py                 # Streamlit dashboard
│   ├── tests.py               # Test suite (21 tests)
│   └── README.md              # Original code README (preserved)
├── data/
│   └── synthetic_tkg_events.csv   # Generated synthetic dataset
├── results/
│   ├── forecast_predictions.csv   # Test set predictions
│   ├── forecast_metrics.json      # Accuracy, F1, Top-3
│   ├── conformal_results.json     # Coverage, set sizes, q_hat
│   ├── explanations.json          # 5 sample explanations
│   └── noise_results.json         # Noise experiment results
├── figures/
│   ├── temporal_event_distribution.png
│   ├── relation_distribution.png
│   ├── confusion_matrix.png
│   ├── coverage_by_time.png
│   ├── prediction_set_sizes.png
│   ├── sample_explanation_subgraph.png
│   └── noise_robustness.png
├── requirements.txt
├── IMPLEMENTATION_AUDIT.md
├── IMPLEMENTATION_STATUS.md        # This file
├── DEMO_SCRIPT.md
├── README_NEW.md
├── README.md                       # Original (preserved)
├── CODE_WALKTHROUGH.md             # Original (preserved)
├── DEMO_MAP.md                     # Original (preserved)
├── MTE_STATUS.md                   # Original (preserved)
├── CET-KG Presentation.pptx       # Original (preserved)
└── CET-KG Research Idea.docx      # Original (preserved)
```

---

## Research Integrity

- All results are from **synthetic data** and are labeled as such.
- No real benchmark metrics (MRR, Hits@K on ICEWS/GDELT) are claimed.
- The conformal coverage guarantee is documented as **approximate** under
  temporal dependence.
- Explanations are labeled as **evidence-based prototypes**, not formal
  gradient-based attributions.
- The forecasting model is a lightweight proof-of-concept, not a state-of-the-art
  temporal KG embedding method.
