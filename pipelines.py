"""
pipelines.py
============
Builds one ImbPipeline per classifier.

SMOTE is placed inside every pipeline to prevent data leakage.
Logistic Regression and KNN include a StandardScaler step.
Naive Bayes and Decision Tree do not require feature scaling.

Hyperparameter choices are described in Section 3.3 of the manuscript.
"""

from imblearn.over_sampling import SMOTE
from imblearn.pipeline      import Pipeline as ImbPipeline
from sklearn.impute         import SimpleImputer
from sklearn.linear_model   import LogisticRegression
from sklearn.naive_bayes    import GaussianNB
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.preprocessing  import StandardScaler
from sklearn.tree           import DecisionTreeClassifier

from config import RANDOM_STATE


def build_pipelines() -> dict:
    """
    Return a dict of {model_name: ImbPipeline}.

    Models
    ------
    Naive Bayes         — probabilistic baseline (no scaling needed)
    Logistic Regression — L-BFGS, balanced class weights, max_iter=1000
    Decision Tree       — max_depth=6, balanced weights, min_samples_leaf=10
    KNN                 — 7 neighbours, distance weighting, Euclidean metric
    """
    smote_cfg = dict(random_state=RANDOM_STATE, k_neighbors=5)

    naive_bayes = ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("smote",   SMOTE(**smote_cfg)),
        ("clf",     GaussianNB()),
    ])

    logistic_regression = ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("smote",   SMOTE(**smote_cfg)),
        ("clf",     LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=RANDOM_STATE)),
    ])

    decision_tree = ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("smote",   SMOTE(**smote_cfg)),
        ("clf",     DecisionTreeClassifier(
                        max_depth=6,
                        class_weight="balanced",
                        min_samples_leaf=10,
                        random_state=RANDOM_STATE)),
    ])

    knn = ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("smote",   SMOTE(**smote_cfg)),
        ("clf",     KNeighborsClassifier(
                        n_neighbors=7,
                        weights="distance",
                        metric="euclidean")),
    ])

    return {
        "Naive Bayes":         naive_bayes,
        "Logistic Regression": logistic_regression,
        "Decision Tree":       decision_tree,
        "KNN":                 knn,
    }
