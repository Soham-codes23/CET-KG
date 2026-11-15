# CET-KG — Faculty Demo Script

## Quick Start (2 minutes)

```bash
# 1. Navigate to project
cd "CET-KG (PBL PROJECT)"

# 2. Install dependencies (one-time)
pip3 install -r requirements.txt

# 3. Run the pipeline (generates all results, ~8 seconds without noise)
python3 code/run_pipeline.py --seed 42 --target-coverage 0.90 --skip-noise

# 4. Launch the dashboard
streamlit run code/app.py
```

---

## Demo Flow (15–20 minutes)

### Slide 1: Overview Page (~3 min)

**Open:** Dashboard → 🏠 Overview

**Show:**
- Pipeline architecture diagram
- Implementation status table
- Point out the **SYNTHETIC PROOF-OF-CONCEPT** label

**Say:**
> "CET-KG integrates four components into a temporal knowledge graph
> forecasting pipeline: forecasting, conformal calibration, graph-grounded
> explanation, and noise robustness. This is a working proof-of-concept
> on synthetic data — not final benchmark validation."

---

### Slide 2: Dataset Explorer (~2 min)

**Open:** Dashboard → 📊 Dataset Explorer

**Show:**
- Dataset statistics: ~3,953 events, 15 countries, 10 relations
- Temporal distribution chart — point out burst periods and train/calib/test splits
- Relation frequency chart

**Say:**
> "We use a synthetic geopolitical event dataset modeled on ICEWS/GDELT.
> Events represent interactions between 15 countries — cooperation, conflict,
> negotiations, sanctions — over 365 simulated days."

---

### Slide 3: Temporal Graph (~2 min)

**Open:** Dashboard → 🕸️ Temporal Graph

**Show:**
- Select "USA" as entity, timestamp around 250, window 30
- Show the neighborhood subgraph visualization
- Point out the edge details table

**Say:**
> "We represent events as edges in a temporal multigraph using NetworkX.
> For any entity and time point, we can extract the historical neighborhood
> to compute temporal features — degree, relation diversity, intensity."

---

### Slide 4: Forecasting Results (~3 min)

**Open:** Dashboard → 🎯 Forecasting Results

**Show:**
- Key metrics: Accuracy 16.6%, Macro-F1 15.2%, Top-3 Accuracy 51.5%
- Confusion matrix
- Sample predictions table

**Say:**
> "Our baseline is a RandomForest classifier predicting the relation type
> between two entities. With 10 classes, random guessing gives 10%.
> Our model reaches 16.6% accuracy and 51.5% top-3 accuracy using only
> temporal graph features. This is a proof-of-concept baseline — production
> systems would use temporal graph neural networks."

**If asked "why is accuracy low?":**
> "This is a 10-class classification problem on synthetic data with
> lightweight features. The purpose is demonstrating the full pipeline,
> not maximizing accuracy. Top-3 accuracy of 51% shows the model captures
> meaningful signal."

---

### Slide 5: Conformal Prediction (~4 min)

**Open:** Dashboard → 📐 Conformal Prediction

**Show:**
- Target coverage: 90%, Empirical coverage: 92.2%
- Average set size: 7.77
- Coverage by time bucket chart
- Assumptions tab

**Say:**
> "Split-conformal prediction constructs prediction sets with a statistical
> coverage guarantee. We target 90% coverage and achieve 92.2% empirically.
> The prediction sets average 7.77 classes out of 10 — this reflects the
> difficulty of the synthetic task."

**Key point on caveat:**
> "Conformal prediction assumes exchangeability, which temporal data violates.
> Our coverage is *approximate*, not guaranteed. Notice the late-period
> coverage drops to 89.9%, consistent with temporal distribution shift."

---

### Slide 6: Explanation (~3 min)

**Open:** Dashboard → 🔍 Explanation

**Show:**
- Select any prediction from the dropdown
- Show the evidence events table with relevance scores
- Show the explanation text
- Show the subgraph visualization (if first prediction selected)

**Say:**
> "For each prediction, we retrieve historical events involving the same
> entities, score their relevance based on entity overlap, relation match,
> and temporal recency, and present the top evidence."

**Important clarification:**
> "This is an evidence-based prototype explanation — not SHAP or gradient-based
> attribution. The relevance scores are heuristic. A full implementation
> would use GNNExplainer or attention-based attribution."

---

### Slide 7: Noise Robustness (~2 min)

**Open:** Dashboard → 🔊 Noise Robustness

*(Only available if pipeline was run without `--skip-noise`)*

**Show:**
- Accuracy vs noise level
- Coverage vs noise level
- Comparison tables

**Say:**
> "We evaluate robustness under two noise types: intensity perturbation
> and label corruption. As noise increases, accuracy degrades and
> prediction sets grow larger, but conformal coverage remains approximately
> preserved."

---

## Common Faculty Questions

**Q: Is this real data?**
> "No, this is synthetic data designed to demonstrate the pipeline. Real
> benchmark validation on ICEWS/GDELT is the planned next phase."

**Q: Why not use a graph neural network?**
> "GNNs require GPU training and more complex dependencies. The
> RandomForest baseline demonstrates the pipeline architecture on a
> Mac CPU. The conformal calibration and explanation modules are
> model-agnostic and will work with any backbone."

**Q: Does conformal prediction really guarantee coverage?**
> "Under exchangeability, yes. For temporal data, the guarantee is
> approximate. We document this limitation explicitly. Our empirical
> results show the approximation works reasonably well."

**Q: What is novel here?**
> "The integration of conformal calibration and graph-grounded explanation
> into a temporal KG forecasting pipeline. Each component exists in
> isolation in the literature, but combining them for trustworthy
> temporal forecasting is the research contribution."

---

## What NOT to claim

- ❌ Do not claim this is final benchmark performance
- ❌ Do not claim the coverage guarantee is exact under temporal dependence
- ❌ Do not claim the explanation is SHAP or formal attribution
- ❌ Do not claim state-of-the-art forecasting accuracy
- ❌ Do not present synthetic results as real-world validation
