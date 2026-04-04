# Early Detection of Generalised Anxiety Disorder in Preschool Children
### Interpretable Machine Learning with Explainable AI

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Kaggle%20%7C%20Local-orange)](https://www.kaggle.com/)
[![Dataset](https://img.shields.io/badge/Dataset-Harvard%20Dataverse-red)](https://doi.org/10.7910/DVN/N42LWG)

---

## Overview

This repository contains the full machine learning pipeline for the paper:

> **Early Detection of Generalised Anxiety Disorder in Preschool Children Using Interpretable Machine Learning with Explainable AI**
> Hazal Karakuş, Selahattin Barış Çelebi, Şeyma Uğur
> Batman University, Türkiye

We apply four interpretable classifiers to parent-reported
**Preschool Age Psychiatric Assessment (PAPA)** data from 917 children
to predict binary Generalised Anxiety Disorder (GAD) status.
Class imbalance (24.6% positive prevalence) is addressed with **SMOTE applied
strictly within each cross-validation fold** to prevent data leakage.
Model decisions are explained using **SHAP (LinearExplainer and TreeExplainer)**.

### Key Results

| Model | AUC-ROC | F1 | Recall | MCC | Holdout Accuracy |
|---|---|---|---|---|---|
| Decision Tree | **0.9884** | **0.8913** | 0.9071 | **0.8552** | **95.65%** |
| Logistic Regression | 0.9740 | 0.8847 | **0.9336** | 0.8464 | 92.39% |
| KNN | 0.8978 | 0.7081 | 0.7566 | 0.6066 | 83.70% |
| Naive Bayes | 0.8724 | 0.6290 | 0.5664 | 0.5302 | 83.15% |

**Top predictors (SHAP + Gini + standardised coefficients):**
`Situational anxious affect` and `Uncontrollable cross-domain worries`
— both aligned with DSM-5 GAD diagnostic criteria.

---

## Project Structure

```
gad_preschool/
├── config.py           # All constants, paths, colour palette, PAPA feature labels
├── data_loader.py      # Load and clean Training_Data.xlsx
├── pipelines.py        # Four ImbPipelines (NB, LR, DT, KNN)
├── evaluation.py       # 10-fold CV and holdout evaluation
├── visualisation.py    # Figures 1–6 (class dist, metrics, ROC, PR, CM, radar)
├── xai.py              # Figures 7–12 (feature importance, DT rules, SHAP)
├── excel_export.py     # Excel workbooks (metrics + actual-vs-predicted)
└── main.py             # Entry point — runs full pipeline
requirements.txt
README.md
```

---

## Dataset

- **Source:** Harvard Dataverse — [doi:10.7910/DVN/N42LWG](https://doi.org/10.7910/DVN/N42LWG)
- **Reference:** Carpenter et al. (2016). *Quantifying Risk for Anxiety Disorders in Preschool Children: A Machine Learning Approach.* PLoS ONE.
- **Instrument:** Preschool Age Psychiatric Assessment (PAPA) — Egger & Angold (2004)
- **Samples:** 917 preschool children (ages 2–5)
- **Features:** 56 behavioural and affective items
- **Target:** Binary GAD diagnosis (0 = No GAD, 1 = GAD)
- **Class ratio:** 3.1 : 1 (691 negative / 226 positive)

> **Data policy:** Only `Training_Data.xlsx` is used.
> `Testing_Data.xlsx` is never loaded or referenced at any stage.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/gad-preschool-ml.git
cd gad-preschool-ml
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Requires Python 3.10 or higher.

### 3. Set the data path

Edit `config.py`:

```python
TRAIN_PATH = "/path/to/your/Training Data.xlsx"
OUTPUT_DIR = "/path/to/output/"
```

On **Kaggle**, the defaults work without modification if the dataset is
added under `bariscelebi/anskiyetedata`.

### 4. Run the pipeline

```bash
cd gad_preschool
python main.py
```

---

## Running on Kaggle

1. Upload this repository as a Kaggle Dataset or paste `main.py` into a notebook.
2. Add the Harvard Dataverse dataset: `bariscelebi/anskiyetedata`.
3. Set accelerator to **None** (CPU is sufficient).
4. Run all cells. All outputs save to `/kaggle/working/`.

---

## Module Guide

| Module | What it does | Key functions |
|---|---|---|
| `config.py` | Constants, paths, palette, PAPA labels | `to_en()` |
| `data_loader.py` | Load Excel, drop onset cols, fix NaN | `load_data()` |
| `pipelines.py` | Build 4 ImbPipelines with SMOTE inside | `build_pipelines()` |
| `evaluation.py` | 10-fold CV + holdout | `run_cv()`, `run_holdout()` |
| `visualisation.py` | Figs 1–6 | `plot_*()` |
| `xai.py` | Figs 7–12 (SHAP, DT rules, importance) | `run_shap()`, `plot_feature_importance()`, `plot_decision_tree_rules()` |
| `excel_export.py` | Excel workbooks | `save_metrics_excel()`, `save_actual_pred_excel()` |
| `main.py` | Orchestrates all steps | `main()` |

---

## Outputs

After a full run, the output directory contains:

| File | Description |
|---|---|
| `fig01_class_distribution.png/svg` | Class distribution bar chart |
| `fig02_metric_comparison.png/svg` | 6-metric grouped bar chart |
| `fig03_roc_curves.png/svg` | ROC curves (10-fold CV) |
| `fig04_pr_curves.png/svg` | Precision-Recall curves |
| `fig05_confusion_matrices.png/svg` | Confusion matrices (aggregated) |
| `fig06_radar.png/svg` | Multi-metric radar chart |
| `fig07_feature_importance.png/svg` | LR coefficients + DT Gini (top 15) |
| `fig08_decision_tree_rules.png/svg` | Decision Tree at depth 4 |
| `fig09_shap_beeswarm_logistic_regression.png/svg` | LR SHAP beeswarm |
| `fig10_shap_bar_logistic_regression.png/svg` | LR SHAP bar (mean \|SHAP\|) |
| `fig11_shap_bar_decision_tree.png/svg` | DT SHAP bar (mean \|SHAP\|) |
| `fig12_shap_dot_decision_tree.png/svg` | DT SHAP dot plot |
| `GAD_model_metrics.xlsx` | CV metrics for all 4 models |
| `GAD_actual_vs_predicted.xlsx` | Holdout predictions per sample |

---

## Methodology Summary

### Pipeline Design

```
Training Data (n=733)
    └── For each CV fold:
            ├── Median imputation (inside fold)
            ├── StandardScaler (LR and KNN only)
            ├── SMOTE k=5 (inside fold — no leakage)
            └── Classifier fit + out-of-fold prediction

Holdout Set (n=184, 20% stratified split)
    └── Final model fit on full training partition
        └── Predict + SHAP attribution
```

### Why SMOTE inside the fold?

Applying oversampling before cross-validation leaks information from
validation folds into training, producing overoptimistic metrics
(Blagus & Lusa, 2013; Yi et al., 2023). All SMOTE steps in this
pipeline are applied exclusively within each training fold.

### Evaluation metrics

Seven metrics are reported: AUC-ROC, Average Precision (AP), F1,
Precision, Recall, Accuracy, and MCC. Matthews Correlation Coefficient
(MCC) is included as the primary robustness metric because it
incorporates all four cells of the confusion matrix and is unaffected
by class imbalance (Chicco & Jurman, 2020).

---

## References

- Carpenter et al. (2016). PLoS ONE. https://doi.org/10.1371/journal.pone.0165524
- Egger & Angold (2004). Handbook of Infant, Toddler, and Preschool Mental Health Assessment.
- Chawla et al. (2002). JAIR. https://doi.org/10.1613/jair.953
- Blagus & Lusa (2013). BMC Bioinformatics. https://doi.org/10.1186/1471-2105-14-106
- Lundberg & Lee (2017). NeurIPS. https://doi.org/10.48550/ARXIV.1705.07874
- Chicco & Jurman (2020). BMC Genomics. https://doi.org/10.1186/s12864-019-6413-7
- Ponce-Bobadilla et al. (2024). CTS. https://doi.org/10.1111/cts.70056
- Yang et al. (2021). Epidemiology and Psychiatric Sciences. https://doi.org/10.1017/S2045796021000275

---

## License

This project is released under the [MIT License](LICENSE).

---

## Contact

**Selahattin Barış Çelebi**
Batman University, Department of Management Information Systems
Batman, Türkiye
