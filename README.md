# Concurrent Classification of Preschool Generalised Anxiety Disorder

## Explainable machine learning with parent-reported PAPA symptoms

This repository contains the analysis code, documented outputs, figures, and manuscript associated with a concurrent classification study of generalised anxiety disorder (GAD) in preschool children.

The analysis uses parent-reported symptom items from the Preschool Age Psychiatric Assessment (PAPA). It does **not** establish prospective early detection, an independently validated screening instrument, or clinical utility.

## Study design

- Source dataset: Duke Preschool Anxiety Study training data, available from the [Harvard Dataverse](https://doi.org/10.7910/DVN/N42LWG)
- Participants: 917 children; 226 met the study definition of GAD
- Development partition: 733 participants
- Same-source internal holdout: 184 participants
- Primary feature set: 54 symptom variables
- Excluded from the primary feature set: onset variables and two variables directly involved in the diagnostic definition
- Primary models: L1-regularised logistic regression, decision tree, random forest, and XGBoost
- Primary evaluation: sampling-design-weighted nested stratified cross-validation
- Model selection: highest sensitivity among models satisfying the prespecified specificity requirement of at least 0.80
- Explainability: model-specific feature importance and SHAP analyses

The same-source holdout is an internal evaluation set, not an external validation cohort.

## Primary results

| Model | AUC-ROC | AP | F1 | Precision | Sensitivity | Specificity | Accuracy | MCC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 logistic regression | 0.7003 | 0.3838 | 0.3981 | 0.3198 | 0.5271 | 0.8053 | 0.7642 | 0.2749 | 0.1230 |
| Decision tree | 0.6838 | 0.3446 | 0.2831 | 0.2185 | 0.4022 | 0.7503 | 0.6988 | 0.1216 | 0.1446 |
| **Random forest** | **0.7704** | **0.4292** | **0.4357** | **0.3558** | **0.5619** | **0.8234** | **0.7847** | **0.3233** | **0.1071** |
| XGBoost | 0.7435 | 0.4149 | 0.3854 | 0.3109 | 0.5069 | 0.8050 | 0.7609 | 0.2588 | 0.1140 |

These values come from sampling-design-weighted nested-CV out-of-fold predictions in the development partition. The Kish effective sample size was 213.65.

Random forest was selected under the prespecified specificity constraint. In the same-source internal holdout it achieved AUC-ROC 0.8747, AP 0.6194, sensitivity 0.6484, specificity 0.8997, accuracy 0.8579, MCC 0.5189, and Brier score 0.1047. The holdout Kish effective sample size was 50.52, so these estimates require cautious interpretation.

For nested-CV out-of-fold predictions, random forest had an expected calibration error of 0.0213, calibration intercept of 0.0930, and calibration slope of 1.0751.

## Diagnostic circularity analysis

The two direct diagnostic-definition variables were excluded from the primary analysis. Separate full-feature and diagnostic-definition-only models are retained only as secondary diagnostic-circularity sensitivity analyses. Their substantially higher AUC values must not be presented as primary model performance or evidence of clinical screening validity.

## Repository structure

```text
.
├── src/
│   └── gad_preschool/
│       ├── settings.py             # Configuration, constants, and shared utilities
│       ├── data.py                 # Data loading and feature-set construction
│       ├── models.py               # Pipelines, searches, metrics, and model specifications
│       ├── evaluation.py           # Nested CV and internal-holdout evaluation
│       ├── figures.py              # Performance and diagnostic figures
│       ├── explainability.py       # Feature importance, tree rules, and SHAP
│       ├── reporting.py            # Tables, manifests, checks, and model selection
│       └── main.py                 # End-to-end workflow orchestration
├── data/
│   └── README.md                   # Data access and local path instructions
├── results/
│   ├── README.md                   # Output roles and interpretation
│   ├── manifest.json               # File hashes and provenance metadata
│   └── full_analysis/              # Verified full-analysis workbooks and figures
├── docs/
│   ├── methodology.md
│   ├── architecture.md
│   ├── reproducibility.md
│   └── results.md
├── manuscript/
│   └── manuscript.docx
├── CITATION.cff
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## Data

Raw participant data are not committed to this repository. Download the training workbook from the Harvard Dataverse record and follow [data/README.md](data/README.md).

## Installation

Python 3.10 or later is required. The final recorded notebook environment used Python 3.12.12; exact historical package versions were not embedded in the archived output files.

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

## Running the analysis

Set the input workbook and output directory through environment variables.

PowerShell:

```powershell
$env:GAD_TRAIN_PATH = "C:\path\to\Training Data.xlsx"
$env:GAD_OUTPUT_DIR = "C:\path\to\outputs"
python -m gad_preschool
```

Bash:

```bash
export GAD_TRAIN_PATH="/path/to/Training Data.xlsx"
export GAD_OUTPUT_DIR="/path/to/outputs"
python -m gad_preschool
```

The committed `results/full_analysis` directory contains archived outputs from the completed full analysis. The repository-refresh operation did not rerun the computational pipeline.

## Interpretation limits

- Predictors and the diagnostic outcome originate from the same PAPA instrument.
- The holdout comes from the same source cohort and is not external validation.
- SHAP describes how fitted models generated predictions; it does not identify causal risk factors.
- Missingness reached 51.1% for duration awake per night and 19.1% for hours taken to fall asleep. A dedicated missing-data sensitivity analysis was not completed.
- Independent prospective external validation, external calibration or recalibration, and clinical-utility assessment are required before implementation.

## Licence

The source code is provided under the MIT License. Dataset use remains subject to the terms of the Harvard Dataverse record and the original data providers.
