# CET-KG — MTE Presentation Package

## Project Title

**CET-KG: Calibrated & Explainable Temporal Knowledge Graph Forecasting**

### Subtitle

**Trustworthy Early-Warning Prediction over Evolving Relational Systems**

---

## 1. Project Overview

CET-KG is a research project focused on forecasting future events
over evolving relational systems using Temporal Knowledge Graphs.

The project investigates how temporal graph forecasting can be made
more trustworthy by combining:

- Temporal Knowledge Graph forecasting
- Uncertainty quantification
- Conformal calibration
- Graph-grounded explainability
- Robustness to historical and supervision noise

The intended outcome is an early-warning forecasting system that
does not only predict a future event, but also provides information
about the reliability and supporting temporal evidence of that
prediction.

---

## 2. Research Problem

Temporal Knowledge Graphs can represent facts and events that evolve
over time and can be used to forecast future events.

However, a raw forecasting score does not directly answer two
important questions:

1. How reliable is the prediction?
2. Why was this prediction made?

CET-KG investigates these issues by introducing calibrated
uncertainty and graph-grounded explanations around temporal
forecasting.

---

## 3. Proposed Research Pipeline

```text
Historical Events
        ↓
Event Representation
        ↓
Temporal Knowledge Graph
        ↓
Temporal Reasoning
        ↓
Future Event Forecasting
        ↓
Uncertainty Quantification
        ↓
Conformal Calibration
        ↓
Graph-Grounded Explanation
        ↓
Trustworthy Early Warning