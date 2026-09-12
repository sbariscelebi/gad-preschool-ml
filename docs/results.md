# Verified results

## Dataset

| Partition | Nominal N | GAD negative | GAD positive | Design-weighted prevalence | Kish effective N |
|---|---:|---:|---:|---:|---:|
| Full dataset | 917 | 691 | 226 | 0.1514 | 264.09 |
| Development | 733 | 552 | 181 | 0.1479 | 213.65 |
| Same-source internal holdout | 184 | 139 | 45 | 0.1665 | 50.52 |

## Selected model

Random forest was selected because it had the highest nested-CV sensitivity among models meeting the prespecified specificity requirement of at least 0.80. Its primary nested-CV metrics were AUC-ROC 0.7704, AP 0.4292, sensitivity 0.5619, specificity 0.8234, accuracy 0.7847, MCC 0.3233, and Brier score 0.1071.

The same-source internal holdout metrics were AUC-ROC 0.8747, AP 0.6194, sensitivity 0.6484, specificity 0.8997, accuracy 0.8579, MCC 0.5189, and Brier score 0.1047. These values do not constitute external validation.

## Calibration

For nested-CV out-of-fold predictions, random forest had expected calibration error 0.0213, calibration intercept 0.0930, and calibration slope 1.0751.

## Circularity sensitivity analysis

Within the separate class-weight sensitivity regime, retaining the two diagnostic-definition variables increased AUC-ROC from 0.7754 to 0.9688 for logistic regression, from 0.7668 to 0.9723 for decision tree, and from 0.7633 to 0.9697 for random forest. These estimates are secondary diagnostic-circularity results and were not used to select the primary model.

