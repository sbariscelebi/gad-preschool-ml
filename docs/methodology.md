# Methodology

## Primary analysis

The primary feature matrix contains 54 PAPA symptom variables. Onset-related variables, the subject identifier, the GAD outcome, the separation anxiety disorder variable, the sampling weight as a predictor, and two direct diagnostic-definition variables are excluded from the predictor matrix.

The 917 observations are split deterministically into a 733-participant development partition and a 184-participant same-source internal holdout using seed 42. Primary performance is estimated in the development partition using 10 outer folds and five inner folds. Hyperparameters are selected using sampling-design-weighted average precision. An additional five-fold cross-fitting procedure within each outer-training partition supplies probabilities for decision-threshold selection.

The threshold maximises sampling-design-weighted sensitivity subject to specificity of at least 0.80. Median imputation is fitted only within the corresponding training data. Standard scaling is applied only to L1-regularised logistic regression.

The primary pipelines use sampling-design weights without SMOTE or class weighting. L1-regularised logistic regression, decision tree, random forest, and XGBoost are compared. Random forest is selected under the prespecified specificity constraint.

## Secondary analyses

Class-weight, SMOTE, native/unweighted, full-feature, and diagnostic-definition-only results are sensitivity analyses. They are not interchangeable with the primary design-weighted results. Full-feature and diagnostic-definition-only analyses evaluate diagnostic circularity.

## Explainability

Absolute standardised logistic-regression coefficients, tree importance measures, and model-specific SHAP values describe fitted-model behaviour. SHAP magnitudes are interpreted within each classifier rather than compared directly across classifiers.

