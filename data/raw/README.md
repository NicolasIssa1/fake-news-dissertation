# Fake News Detection Dissertation

COMP3931 Individual Project  
Student: Nicolas Issa  

This repository contains notebooks, artefacts, and report material for a **research-led (Method 3)** investigation into fake news detection using NLP on the **LIAR dataset (Wang, 2017)**.  
The focus is on **controlled experiments** (baselines → justified variants → transformer models) and **evaluation + analysis** (macro-F1, confusion patterns, error analysis), rather than building a large software system.

---

## Dataset: LIAR

Source (Kaggle mirror): https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset  
Original paper: Wang (2017), *LIAR: A Benchmark Dataset for Fake News Detection*.

Description:  
LIAR is a benchmark dataset of short political statements labeled across six truthfulness categories.

Files:
- `train.tsv`
- `validation.tsv`
- `test.tsv`

Labels (6-way):
- `pants-fire`, `false`, `barely-true`, `half-true`, `mostly-true`, `true`

---

## Repository Structure

- `notebooks/`
  - `01_data_exploration.ipynb` — dataset exploration and justification of preprocessing decisions
  - `02_baseline_models.ipynb` — classical ML baselines (TF-IDF + Logistic Regression / Naive Bayes)
  - `03_transformer_models.ipynb` — transformer baseline(s) (planned next)
  - `04_evaluation.ipynb` — evaluation + error analysis (loads predictions/metrics and produces report-ready artefacts)

- `results/` (**canonical artefacts folder; commit outputs here**)
  - model metrics + predictions + confusion matrices + error analysis artefacts (see below)

- `report/`
  - dissertation report files / drafts (Chapters 1–4 etc.)

- `src/`
  - (optional) reusable helper code if needed later

---

## Notebook 01 — Data Exploration (Week 1 complete)

**Goal:** understand the dataset well enough to justify every later decision in the dissertation.

Completed:
- Loaded `train.tsv`, `validation.tsv`, `test.tsv`
- Mapped LIAR columns to their meanings (schema understanding)
- Checked label distribution and basic data quality
- Confirmed dataset split is preserved (train/valid/test)

Output:
- No saved artefacts required (analysis notebook)

---

## Notebook 02 — Baseline Models (Week 2 complete)

**Goal:** establish reproducible classical baselines for comparison.

Baselines implemented:
- TF-IDF + Logistic Regression (evaluated on validation)
- TF-IDF + Naive Bayes (evaluated on validation)

Artefacts saved to `/results/`:
- `baseline_metrics.json` — baseline metrics (includes macro-F1)
- `valid_predictions.csv` — validation predictions used for reproducibility + error analysis
- `confusion_matrix.csv` — confusion matrix (baseline)

Notes / small TODO:
- Add `random_state=42` to LogisticRegression for reproducibility
- Add a short markdown summary cell at the top of Notebook 02 explaining what it does

---

## Notebook 04 — Evaluation & Error Analysis (Week 3 Tuesday complete)

**Goal:** convert baseline runs into dissertation-quality evidence:
- reproducible evaluation (macro-F1 focus)
- confusion patterns
- qualitative error analysis

Generated artefacts saved to `/results/`:
- `confusion_matrix_final.csv` — regenerated from predictions (canonical)
- `per_class_metrics.csv` — per-class precision/recall/F1 + support
- `top_confusions.csv` — most frequent misclassification pairs
- `error_examples_top3.csv` — sampled examples from top confusion pairs
- (optional/legacy) `baseline_metrics_recomputed.json`, `confusion_matrix_recomputed.csv`

Key finding (baseline error pattern):
- Most frequent confusions are between adjacent labels:
  - `barely-true -> half-true`
  - `mostly-true -> half-true`
  - `half-true -> false`

Interpretation:
- TF-IDF relies on surface lexical cues and struggles with fine-grained veracity boundaries, motivating controlled variants and transformer models.

---

## How to Run (Suggested Order)

1) `01_data_exploration.ipynb`
2) `02_baseline_models.ipynb`
3) `04_evaluation.ipynb`
4) `03_transformer_models.ipynb` (next)

---

## Evaluation Metric Choice

Main comparison metric: **macro-F1**  
Reason: LIAR is multi-class and imbalanced; macro-F1 treats each class equally and reveals poor performance on minority/difficult labels.

---

## Current Week Focus

Week 3 goal:
- controlled baseline variants (e.g., TF-IDF n-grams, class weighting)
- evaluation + error analysis for each variant using the same artefact pipeline

Week 4 goal:
- run at least one transformer baseline (e.g., DistilBERT)
- compare against classical baselines using macro-F1 and error patterns

## Supervisor / Assessor Meeting Notes

### Current status (as of Week 3 Tuesday)
- Baseline models complete (TF-IDF + Logistic Regression / Naive Bayes)
- Evaluation pipeline complete (reproducible metrics + confusion + error analysis artefacts)
- Key baseline failure pattern: adjacent-label confusion (half-true acts as a “magnet” class)

### Key results (validation)
- Baseline TF-IDF + Logistic Regression:
  - Accuracy ≈ 0.215
  - Macro-F1 ≈ 0.196

### Questions for feedback
1) Which controlled variants are most appropriate before moving to transformers?
   - Proposed: TF-IDF (1,2) n-grams; class_weight="balanced"
2) For transformer baseline, is DistilBERT appropriate as the first deep model?
3) Should I include LIAR metadata features in a separate controlled experiment (text-only vs text+metadata)?

### Next planned steps
- Week 3: run controlled classical variants and compare using macro-F1 + error patterns
- Week 4: run a transformer baseline and compare against the strongest classical model
