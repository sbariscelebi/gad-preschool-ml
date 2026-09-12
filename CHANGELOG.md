# Changelog

## 1.1.0 — 2026-09-12

- Reorganised the single analysis script as an installable `gad_preschool` package.
- Separated configuration, data preparation, modelling, evaluation, figures, explainability, reporting, and workflow orchestration.
- Preserved the analysis functions and command-line entry point without changing the archived numerical results.
- Added an architecture guide and updated the reproducibility instructions.

## 1.0.0 — 2026-09-12

- Replaced the legacy early-detection/SMOTE repository with the completed concurrent-classification analysis.
- Added random forest and XGBoost to the primary model comparison.
- Added sampling-design-weighted nested cross-validation and same-source holdout reporting.
- Added calibration, threshold, feature-stability, and diagnostic-circularity outputs.
- Added the revised manuscript, figures, result workbooks, citation metadata, and reproducibility documentation.
- Archived the previous repository state on `archive/legacy-smote-2026-09-12`.
