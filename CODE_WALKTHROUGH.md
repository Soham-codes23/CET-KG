`CODE_WALKTHROUGH.md`

Ye Ma'am ke **“code mein exactly kya kiya?”** wale question ke liye hai.

Yahan **fake function names mat likhna** jab tak actual code mein woh functions hain. Isliye deliberately generic rakhenge.

```markdown
# CET-KG Proof-of-Concept — Code Walkthrough

## 1. Purpose

The `cet_kg_poc.py` script implements a synthetic proof-of-concept
for the CET-KG research direction.

The purpose of the prototype is to demonstrate the interaction
between temporal forecasting, uncertainty, calibration and
graph-grounded explanation.

---

# 2. Input Data

The prototype operates on synthetic temporal data.

The dataset represents observations/events occurring at different
points in time.

Conceptually:

```text
Timestamp
    +
Observed Event / Features
    +
Historical Context