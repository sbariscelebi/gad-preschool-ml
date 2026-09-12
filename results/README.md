# Results directory

The `full_analysis` directory contains outputs retained from the completed full analysis.

## Primary evidence

- `GAD_primary_design_weight_metrics.xlsx`: primary weighted nested-CV and same-source holdout metrics, model selection, feature audit, and acceptance checks.
- `GAD_calibration_and_CI.xlsx`: calibration summaries, reliability bins, and bootstrap confidence intervals.
- `GAD_threshold_audit.xlsx`: fold-specific decision-threshold audit.
- `GAD_feature_stability.xlsx`: feature-ranking stability summaries.
- `GAD_internal_holdout_predictions.xlsx`: model predictions for the same-source holdout.

## Secondary sensitivity evidence

- `GAD_primary_class_weight_metrics.xlsx`
- `GAD_smote_sensitivity_analysis.xlsx`
- `GAD_circularity_paired_differences.xlsx`
- `GAD_internal_holdout_predictions_class_weight_sensitivity.xlsx`
- `GAD_missingness_sensitivity.xlsx`, whose audit status records that the dedicated analysis was not run

Files whose names contain `smote`, `class_weight`, `circularity`, `diagnostic`, or `sensitivity` must not be cited as primary model performance.

Some retained figure filenames contain historical generator numbers such as `figure03` or `fig12`. These filenames identify pipeline outputs and do not determine the final manuscript figure numbering; consult `manuscript/manuscript.docx` for the published figure order.
