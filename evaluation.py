"""
evaluation.py
=============
Cross-validation and holdout evaluation for all classifiers.

Functions
---------
run_cv       — 10-fold stratified CV; returns metrics dict and probability dict.
run_holdout  — 80/20 stratified split; returns actual-vs-predicted DataFrame.
"""

import numpy as np
import pandas as pd
from sklearn.base         import clone
from sklearn.metrics      import (
    accuracy_score, average_precision_score, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from config import CV_FOLDS, RANDOM_STATE, TEST_SIZE


def run_cv(pipelines: dict, X: pd.DataFrame, y: np.ndarray) -> tuple:
    """
    Ten-fold stratified cross-validation.

    SMOTE is applied inside each pipeline, so synthetic samples never
    leak into the validation fold.

    Parameters
    ----------
    pipelines : dict   {name: ImbPipeline}
    X         : DataFrame
    y         : ndarray  binary labels

    Returns
    -------
    results : dict  {name: {metric: value}}
    probas  : dict  {name: {y_true, y_proba, y_pred}}
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    results, probas = {}, {}

    for name, pipe in pipelines.items():
        y_proba = cross_val_predict(
            pipe, X, y, cv=cv, method="predict_proba"
        )[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        results[name] = {
            "AUC-ROC":   round(roc_auc_score(y, y_proba),            4),
            "AP":        round(average_precision_score(y, y_proba),   4),
            "F1":        round(f1_score(y, y_pred),                   4),
            "Precision": round(precision_score(y, y_pred),            4),
            "Recall":    round(recall_score(y, y_pred),               4),
            "Accuracy":  round(accuracy_score(y, y_pred),             4),
            "MCC":       round(matthews_corrcoef(y, y_pred),          4),
        }
        probas[name] = {
            "y_true":  y,
            "y_proba": y_proba,
            "y_pred":  y_pred,
        }

        r = results[name]
        print(f"  {name:<22}  AUC={r['AUC-ROC']:.4f}  "
              f"F1={r['F1']:.4f}  MCC={r['MCC']:.4f}")

    return results, probas


def run_holdout(pipelines: dict, X: pd.DataFrame,
                y: np.ndarray) -> pd.DataFrame:
    """
    Independent 80/20 stratified holdout evaluation.

    Parameters
    ----------
    pipelines : dict
    X         : DataFrame
    y         : ndarray

    Returns
    -------
    pd.DataFrame with columns:
        Model, Sample_Index, Actual, Predicted, Probability_GAD, Correct
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE,
        stratify=y, random_state=RANDOM_STATE
    )

    rows = []
    for name, pipe in pipelines.items():
        fitted = clone(pipe)
        fitted.fit(X_train, y_train)
        y_proba = fitted.predict_proba(X_test)[:, 1]
        y_pred  = (y_proba >= 0.5).astype(int)

        for i, (actual, pred, prob) in enumerate(zip(y_test, y_pred, y_proba)):
            rows.append({
                "Model":           name,
                "Sample_Index":    i,
                "Actual":          int(actual),
                "Predicted":       int(pred),
                "Probability_GAD": round(float(prob), 4),
                "Correct":         int(actual) == int(pred),
            })

    return pd.DataFrame(rows)
