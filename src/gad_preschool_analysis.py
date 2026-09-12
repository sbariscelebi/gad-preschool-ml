"""
==============================================================================
Concurrent Classification of Generalised Anxiety Disorder in Preschool Children:
An Explainable Machine Learning Analysis of Parent-Reported Symptoms
------------------------------------------------------------------------------
Dataset  : Harvard Dataverse  doi:10.7910/DVN/N42LWG
           Carpenter et al. (2016)  PLOS ONE
Target   : GAD (Generalised Anxiety Disorder)  binary classification
Primary models : L1 Logistic Regression | Decision Tree | Random Forest | XGBoost
Baselines      : Naive Bayes and KNN are excluded from the primary analysis
XAI            : SHAP summary + Decision Tree visualisation + Feature importance
Platform : Kaggle  Python 3.10+
------------------------------------------------------------------------------
REVIEWER-1 REVISION POLICY:
  The internal holdout is created before model development.
  Nested cross-validation and hyperparameter selection use training data only.
  Direct diagnostic-definition features are excluded from the primary analysis.
  Full-feature and diagnostic-only analyses are reported as sensitivity analyses.
  Sampling-design weighting without class rebalancing is the primary strategy.
  Native, class-weight, and SMOTE regimes are sensitivity analyses.
  Screening thresholds are learned inside outer-training data only and must
  satisfy specificity >= 0.80 before sensitivity is maximised.
  The internal holdout is never described as external/independent validation.
==============================================================================
"""

# ── 0. LOCALE LOCK  (must be first — forces English in all figure text) ────
import locale
try:
    locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
except Exception:
    try:
        locale.setlocale(locale.LC_ALL, "C.UTF-8")
    except Exception:
        locale.setlocale(locale.LC_ALL, "C")

# ── 1. IMPORTS ─────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import hashlib
import json
import os
import platform
import sys
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.formatter.use_locale"] = False
matplotlib.rcParams["axes.unicode_minus"]        = False

import matplotlib.pyplot as plt

import shap
import sklearn
import imblearn
import openpyxl
from sklearn.linear_model      import LogisticRegression
from sklearn.tree              import DecisionTreeClassifier, plot_tree
from sklearn.ensemble          import RandomForestClassifier
from sklearn.impute            import SimpleImputer
from sklearn.preprocessing     import StandardScaler
from sklearn.model_selection   import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split,
    ParameterGrid, ParameterSampler
)
from sklearn.metrics           import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, balanced_accuracy_score, brier_score_loss,
    matthews_corrcoef, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.base              import clone
from sklearn.calibration       import calibration_curve
from imblearn.over_sampling    import SMOTE
from imblearn.pipeline         import Pipeline as ImbPipeline

try:
    import xgboost
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    xgboost = None
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

from openpyxl                  import Workbook
from openpyxl.styles           import Font, PatternFill, Alignment
from openpyxl.utils            import get_column_letter


# ── 2. CONFIGURATION ───────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20
OUTER_FOLDS  = 10
INNER_FOLDS  = 5
DPI          = 300






PRIMARY_SCORING = "average_precision"

# ============================================================
# TEST MODE
# True  = hızlı test; sonuçlar makalede kullanılmaz
# False = tam bilimsel analiz
# ============================================================
QUICK_TEST = False

RUN_NATIVE_SENSITIVITY = False
RUN_MISSINGNESS_AUDIT = False

SKIP_SHAP = False
SHOW_INLINE_OUTPUTS = True
REQUIRE_XGBOOST = True




SPECIFICITY_FLOOR = 0.80

# Normal analiz değerleri.
BOOTSTRAP_ITERATIONS = 2000
RANDOM_SEARCH_ITERATIONS = 40
THRESHOLD_GRID_SIZE = 1001
CALIBRATION_BINS = 10

# ============================================================
# QUICK TEST SETTINGS
# ============================================================
if QUICK_TEST:
    OUTER_FOLDS = 2
    INNER_FOLDS = 2
    BOOTSTRAP_ITERATIONS = 10
    RANDOM_SEARCH_ITERATIONS = 1
    THRESHOLD_GRID_SIZE = 31
    CALIBRATION_BINS = 5
    DPI = 100

    print("=" * 70)
    print("QUICK TEST MODE IS ACTIVE")
    print("Results are for code/figure validation only.")
    print("DO NOT use these numerical results in the manuscript.")
    print("=" * 70)











TRAIN_PATH = os.getenv(
    "GAD_TRAIN_PATH",
    "/kaggle/input/datasets/bariscelebi/anskiyetedata/Training Data.xlsx",
)
OUTPUT_DIR = os.getenv("GAD_OUTPUT_DIR", "/kaggle/working/")

DIRECT_DIAGNOSTIC_FEATURES = [
    "Anxious affect that occurs in certain situations/environments",
    "Worries that cannot be stopped voluntarily and occur across more than one acctivity",
]

# Flagged for clinician review, but deliberately retained until clinical
# adjudication; this list does not alter the prespecified 54-feature matrix.
SUSPICIOUS_HIGH_LEVEL_FEATURES = [
    "Fear/anxiety about daycare/school attendance screen positive",
    "Anxiety not associated with any particular situation",
]

FONT_TITLE   = 20   # retained for rare axes-level annotations; no in-figure titles
FONT_AXIS    = 20   # axis labels
FONT_TICK    = 13   # tick labels
FONT_LEGEND  = 20   # legend text

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 3. COLOUR PALETTE ──────────────────────────────────────────────────────
PALETTE = {
    "L1 Logistic Regression": "#1D9E75",
    "Decision Tree":        "#EF9F27",
    "Random Forest":        "#3B8BD4",
    "XGBoost":             "#8F64B8",
    "grid":                 "#E0DED8",
    "bg":                   "#F8F8F6",
}

MODEL_SHORT = {
    "L1 Logistic Regression": "LR_L1",
    "Decision Tree":        "DT",
    "Random Forest":        "RF",
    "XGBoost":             "XGB",
}

# ── English labels for every PAPA feature that appears on figures ──────────
# Keys must exactly match the Excel column names.
FEATURE_LABELS_EN = {
    "Irritability":
        "Irritability",
    "Increased unnecessary whole body movements in specific situations":
        "Unnecessary body movements",
    "Difficulty concentrating on tasks or play activity independently":
        "Difficulty concentrating (independent)",
    "Difficulty concentrating on adult-directed tasks or play activities":
        "Difficulty concentrating (adult-directed)",
    "Inattention":
        "Inattention",
    "Fear about possible harm befalling major attachment figures":
        "Fear of harm to attachment figures",
    "Fear about calamitous separation":
        "Fear of calamitous separation",
    "Avoidance of being alone":
        "Avoidance of being alone",
    "Anticipatory distress/resistance to separation":
        "Anticipatory distress at separation",
    "Withdrawal when attachement figure absent":
        "Withdrawal when attachment figure absent",
    "Actual distress when attachment figure absent":
        "Distress when attachment figure absent",
    "Complaints of physical symptoms when separation from major attachment figure is anticipated":
        "Physical symptoms at anticipated separation",
    "Parent's plan disrupted due to child's distress at separation":
        "Parent plan disrupted by separation distress",
    "Complaints of physical symptoms when attendance at school/daycare is anticipated or occurs":
        "Physical symptoms at school/daycare attendance",
    "Fear/anxiety about daycare/school attendance screen positive":
        "Fear/anxiety about school attendance",
    "Fear/Anxiety about leaving home for daycare/school":
        "Fear of leaving home for school",
    "Anticipatory fear of daycare/school":
        "Anticipatory fear of daycare/school",
    "Daycare/school non-attendance due to anxiety":
        "School non-attendance due to anxiety",
    "Has to be taken to daycare/school":
        "Must be taken to school",
    "Has to be taken to daycare/school because of separation anxiety":
        "Must be taken to school (separation anxiety)",
    "Picked up early from daycare/school due to anxiety":
        "Picked up early from school",
    "Child tries unsuccessfully to leave daycare/school due to anxiety":
        "Tries to leave school (unsuccessful)",
    "Child leaves daycare/school due to anxiety":
        "Leaves school due to anxiety",
    "Frequency of reluctance to go to sleep":
        "Reluctance to go to sleep",
    "Frequency of sleeping with family member due to a reluctance to sleep alone":
        "Sleeps with family member",
    "Sleep resistence":
        "Sleep resistance",
    "Hours taken to fall asleep":
        "Hours to fall asleep",
    "Freqency of nights child wakes up during the night":
        "Night waking frequency",
    "How long awak per night":
        "Duration awake per night",
    "Rising at night to check on family members":
        "Rises at night to check family",
    "Increased need for sleep":
        "Increased need for sleep",
    "Restless sleep":
        "Restless sleep",
    "Inadequately rested by sleep":
        "Inadequately rested by sleep",
    "Falls asleep in carseat for unscheduled nap":
        "Falls asleep in car (unscheduled)",
    "Tiredness":
        "Tiredness",
    "Child becomes tired or \"worn out\" more easily than normal":
        "Easily worn out",
    "Separation dreams":
        "Separation dreams",
    "Nervous tension":
        "Nervous tension",
    "Anxious affect that occurs in certain situations/environments":
        "Situational anxious affect",
    "Anxiety not associated with any particular situation":
        "Non-situational anxiety",
    "Exaggerated tartle response":
        "Exaggerated startle response",
    "Concentration difficulties":
        "Concentration difficulties",
    "Easy fatigability":
        "Easy fatigability",
    "Muscle Tension":
        "Muscle tension",
    "Restlessness":
        "Restlessness",
    "Worries that cannot be stopped voluntarily and occur across more than one acctivity":
        "Uncontrollable cross-domain worries",
    "Frequency of worries":
        "Frequency of worries",
    "Hypochondriasis":
        "Hypochondriasis",
    "Worry that family members will become ill":
        "Worry about family illness",
    "Worry about the future":
        "Worry about the future",
    "Worries about natural calamity":
        "Worry about natural disaster",
    "Worries about past behavior":
        "Worry about past behaviour",
    "Worries about competence or performance":
        "Worry about competence/performance",
    "Worries about appearance":
        "Worry about appearance",
    "Worries about money/food":
        "Worry about money/food",
    "Other worries":
        "Other worries",
}

def to_en(name: str, max_len: int = 48) -> str:
    """Return the English short label for a feature name."""
    label = FEATURE_LABELS_EN.get(name, name)
    return label[:max_len]


# ── 4. GLOBAL STYLE ────────────────────────────────────────────────────────
def apply_style() -> None:
    """Apply a clean, publication-ready Matplotlib style."""
    plt.rcParams.update({
        "font.family":                  "serif",
        "font.serif":                   ["DejaVu Serif", "Liberation Serif"],
        "font.size":                    14,
        "axes.titlesize":               FONT_TITLE,
        "axes.titleweight":             "normal",
        "axes.labelsize":               FONT_AXIS,
        "xtick.labelsize":              FONT_TICK,
        "ytick.labelsize":              FONT_TICK,
        "legend.fontsize":              FONT_LEGEND,
        "figure.dpi":                   DPI,
        "savefig.dpi":                  DPI,
        "svg.fonttype":                 "none",
        "figure.facecolor":             "white",
        "axes.facecolor":               "white",
        "axes.grid":                    True,
        "grid.color":                   PALETTE["grid"],
        "grid.linewidth":               0.6,
        "grid.alpha":                   0.55,
        "axes.spines.top":              False,
        "axes.spines.right":            False,
        "axes.formatter.use_locale":    False,
        "axes.unicode_minus":           False,
    })

apply_style()


# ── 5. DATA LOADING AND PRESPECIFIED FEATURE SETS ──────────────────────────
def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data(path: str):
    """Load PAPA data and preserve outcome, IDs, and sampling-design weights."""
    df = pd.read_excel(path)
    required = {"Subject", "GAD", "Sampling Weight",
                *DIRECT_DIAGNOSTIC_FEATURES}
    missing_required = sorted(required.difference(df.columns))
    if missing_required:
        raise ValueError(f"Required columns are missing: {missing_required}")
    if df["Subject"].duplicated().any():
        raise ValueError("Subject IDs are not unique; grouped splitting is required.")
    if not set(df["GAD"].dropna().unique()).issubset({0, 1}):
        raise ValueError("GAD must be binary and encoded as 0/1.")

    drop_fixed = ["Subject", "SAD", "GAD", "Sampling Weight"]
    onset_cols = [c for c in df.columns if "Onset" in c]
    feature_cols = [c for c in df.columns
                    if c not in drop_fixed + onset_cols]

    X = df[feature_cols].copy()
    for col in feature_cols:
        X[col] = pd.to_numeric(
            X[col].astype(str).str.strip().replace({".": np.nan, "": np.nan}),
            errors="coerce",
        )

    y = df["GAD"].astype(int).to_numpy()
    sampling_weight = pd.to_numeric(
        df["Sampling Weight"], errors="coerce").to_numpy(dtype=float)
    if (not np.isfinite(sampling_weight).all() or
            np.any(sampling_weight <= 0)):
        raise ValueError("Sampling Weight must contain finite positive values.")
    print(f"Dataset loaded: {len(df)} samples, {len(feature_cols)} predictors")
    print(f"GAD prevalence: {y.mean():.3f} ({int(y.sum())}/{len(y)})")
    weighted_prevalence = float(np.average(y, weights=sampling_weight))
    kish_n = float(sampling_weight.sum() ** 2 /
                   np.square(sampling_weight).sum())
    print(f"Design-weighted GAD prevalence: {weighted_prevalence:.3f}")
    print(f"Approximate Kish effective sample size: {kish_n:.1f}")
    print(f"Onset columns excluded: {len(onset_cols)}")
    return df, X, y, sampling_weight


def build_feature_sets(X: pd.DataFrame) -> dict:
    """Create the reviewer-requested primary and sensitivity feature sets."""
    missing = [c for c in DIRECT_DIAGNOSTIC_FEATURES if c not in X.columns]
    if missing:
        raise ValueError(f"Diagnostic-definition features missing: {missing}")
    reduced = X.drop(columns=DIRECT_DIAGNOSTIC_FEATURES)
    if set(reduced.columns).intersection(DIRECT_DIAGNOSTIC_FEATURES):
        raise AssertionError("Diagnostic-definition features remain in primary set.")
    return {
        "Primary_reduced": reduced,
        "Full_features_sensitivity": X.copy(),
        "Diagnostic_only_sensitivity": X[DIRECT_DIAGNOSTIC_FEATURES].copy(),
    }


def make_train_holdout_indices(y: np.ndarray) -> tuple:
    """Create the untouched internal holdout before all model development."""
    all_idx = np.arange(len(y))
    train_idx, holdout_idx = train_test_split(
        all_idx, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    if set(train_idx).intersection(holdout_idx):
        raise AssertionError("Training/holdout index overlap detected.")
    if len(train_idx) + len(holdout_idx) != len(y):
        raise AssertionError("Train/holdout split does not cover every sample.")
    return np.sort(train_idx), np.sort(holdout_idx)


# ── 6. REVIEWER-ALIGNED MODEL SPECIFICATIONS ──────────────────────────────
def build_model_specs(resampling_mode: str,
                      imputation_strategy: str = "median") -> dict:
    """Build design-weight, native, class-weight, or SMOTE model regimes."""
    allowed_modes = {"design_weight", "native", "class_weight", "smote"}
    if resampling_mode not in allowed_modes:
        raise ValueError(f"resampling_mode must be one of {allowed_modes}.")
    if imputation_strategy not in {"median", "zero"}:
        raise ValueError("imputation_strategy must be 'median' or 'zero'.")

    use_smote = resampling_mode == "smote"
    smote_step = [("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5))]
    if not use_smote:
        smote_step = []

    use_class_weight = resampling_mode == "class_weight"
    lr_weight = "balanced" if use_class_weight else None
    dt_weight = "balanced" if use_class_weight else None
    rf_weight = "balanced_subsample" if use_class_weight else None
    rf_trees = [20] if QUICK_TEST else [300, 600, 1000]
    imputer = (SimpleImputer(strategy="median")
               if imputation_strategy == "median" else
               SimpleImputer(strategy="constant", fill_value=0))

    specs = {
        "L1 Logistic Regression": {
            "pipeline": ImbPipeline([
                ("imputer", clone(imputer)),
                ("scaler", StandardScaler()),
                *smote_step,
                ("clf", LogisticRegression(
                    penalty="l1", solver="liblinear", max_iter=5000,
                    class_weight=lr_weight, random_state=RANDOM_STATE)),
            ]),
            "search": "grid",
            "params": {"clf__C": [1.0] if QUICK_TEST else
                   list(np.logspace(-3, 2, 11))},
        },
        "Decision Tree": {
            "pipeline": ImbPipeline([
                ("imputer", clone(imputer)),
                *smote_step,
                ("clf", DecisionTreeClassifier(
                    class_weight=dt_weight, random_state=RANDOM_STATE)),
            ]),
            "search": "random",
            "params": {
                "clf__max_depth": [3, 6] if QUICK_TEST else
                                  [2, 3, 4, 5, 6, 8, 10, None],
                "clf__min_samples_leaf": [5, 10] if QUICK_TEST else
                                          [2, 5, 10, 15, 20, 30],
                "clf__criterion": ["gini"] if QUICK_TEST else
                                  ["gini", "entropy", "log_loss"],
                "clf__ccp_alpha": [0.0] if QUICK_TEST else
                                  [0.0, 0.0001, 0.001, 0.005, 0.01, 0.02],
            },
        },
        "Random Forest": {
            "pipeline": ImbPipeline([
                ("imputer", clone(imputer)),
                *smote_step,
                ("clf", RandomForestClassifier(
                    class_weight=rf_weight, random_state=RANDOM_STATE,
                    n_jobs=1)),
            ]),
            "search": "random",
            "params": {
                "clf__n_estimators": rf_trees,
                "clf__max_depth": [4, None] if QUICK_TEST else
                                  [4, 6, 8, 10, 14, None],
                "clf__min_samples_leaf": [5, 10] if QUICK_TEST else
                                          [1, 2, 5, 10, 20],
                "clf__max_features": ["sqrt"] if QUICK_TEST else
                                      ["sqrt", "log2", 0.35, 0.5, 0.75],
                "clf__ccp_alpha": [0.0] if QUICK_TEST else
                                  [0.0, 0.0001, 0.001, 0.005],
            },
        },
    }

    if XGBOOST_AVAILABLE:
        specs["XGBoost"] = {
            "pipeline": ImbPipeline([
                ("imputer", clone(imputer)),
                *smote_step,
                ("clf", XGBClassifier(
                    objective="binary:logistic", eval_metric="logloss",
                    tree_method="hist", random_state=RANDOM_STATE,
                    n_jobs=1, verbosity=0)),
            ]),
            "search": "random",
            "dynamic_scale_pos_weight": use_class_weight,
            "params": {
                "clf__n_estimators": [30] if QUICK_TEST else
                                     [150, 300, 500, 800, 1200],
                "clf__max_depth": [3] if QUICK_TEST else [2, 3, 4, 5, 6],
                "clf__learning_rate": [0.1] if QUICK_TEST else
                                      [0.01, 0.03, 0.05, 0.08, 0.12, 0.20],
                "clf__subsample": [0.8] if QUICK_TEST else [0.6, 0.75, 0.9, 1.0],
                "clf__colsample_bytree": [0.8] if QUICK_TEST else
                                         [0.5, 0.7, 0.9, 1.0],
                "clf__min_child_weight": [1] if QUICK_TEST else [1, 3, 5, 10],
                "clf__gamma": [0.0] if QUICK_TEST else [0.0, 0.1, 0.5, 1.0],
                "clf__reg_alpha": [0.0] if QUICK_TEST else [0.0, 0.01, 0.1, 1.0],
                "clf__reg_lambda": [1.0] if QUICK_TEST else [0.5, 1.0, 2.0, 5.0, 10.0],
            },
        }
    return specs


def _round_metric(value) -> float:
    return round(float(value), 4) if np.isfinite(value) else np.nan


def _validated_weights(sample_weight, n_samples: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(n_samples, dtype=float)
    weights = np.asarray(sample_weight, dtype=float)
    if weights.shape != (n_samples,):
        raise ValueError("sample_weight must match the number of observations.")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("sample_weight must contain finite positive values.")
    return weights


def calculate_metrics(y_true: np.ndarray, y_proba: np.ndarray,
                      threshold=0.50, sample_weight=None) -> dict:
    """Calculate unweighted or sampling-design-weighted positive-class metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    weights = _validated_weights(sample_weight, len(y_true))
    threshold_array = np.asarray(threshold, dtype=float)
    if threshold_array.ndim == 0:
        y_pred = (y_proba >= float(threshold_array)).astype(int)
        threshold_label = round(float(threshold_array), 6)
    else:
        if threshold_array.shape != y_proba.shape:
            raise ValueError("Per-sample thresholds must match y_proba shape.")
        y_pred = (y_proba >= threshold_array).astype(int)
        threshold_label = "fold_specific"

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1], sample_weight=weights).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    f1 = (2 * tp / (2 * tp + fp + fn)
          if (2 * tp + fp + fn) else 0.0)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    balanced_accuracy = (sensitivity + specificity) / 2
    denominator = np.sqrt((tp + fp) * (tp + fn) *
                          (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denominator
           if denominator else 0.0)
    weighted = sample_weight is not None
    count = (lambda value: round(float(value), 4)) if weighted else int
    kish_n = float(weights.sum() ** 2 / np.square(weights).sum())
    return {
        "Decision_Threshold": threshold_label,
        "Evaluation_Weighting": (
            "sampling_design_weighted" if weighted else "unweighted"),
        "Weight_Sum": round(float(weights.sum()), 4),
        "Kish_Effective_N": round(kish_n, 2),
        "AUC-ROC": _round_metric(roc_auc_score(
            y_true, y_proba, sample_weight=weights)),
        "AP": _round_metric(average_precision_score(
            y_true, y_proba, sample_weight=weights)),
        "F1": _round_metric(f1),
        "Precision": _round_metric(precision),
        "Sensitivity": _round_metric(sensitivity),
        "Specificity": _round_metric(specificity),
        "NPV": _round_metric(npv),
        "Accuracy": _round_metric(accuracy),
        "Balanced Accuracy": _round_metric(balanced_accuracy),
        "MCC": _round_metric(mcc),
        "Brier": _round_metric(np.average(
            np.square(y_proba - y_true), weights=weights)),
        "TN": count(tn), "FP": count(fp),
        "FN": count(fn), "TP": count(tp),
    }


def derive_screening_threshold(y_true: np.ndarray, y_proba: np.ndarray,
                               specificity_floor: float = SPECIFICITY_FLOOR,
                               sample_weight=None
                               ) -> dict:
    """Maximise weighted sensitivity subject to the specificity floor."""
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    weights = _validated_weights(sample_weight, len(y_true))
    if len(np.unique(y_true)) < 2:
        return {"threshold": 0.50, "sensitivity": np.nan,
                "specificity": np.nan, "status": "constraint_not_met"}

    candidates = np.unique(np.concatenate([
        np.linspace(0.0, 1.0, THRESHOLD_GRID_SIZE),
        y_proba,
        [np.nextafter(float(np.max(y_proba)), np.inf)],
    ]))
    rows = []
    for value in candidates:
        pred = (y_proba >= value).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true, pred, labels=[0, 1], sample_weight=weights).ravel()
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        if np.isfinite(spec) and spec + 1e-12 >= specificity_floor:
            rows.append((sens, spec, float(value)))
    if not rows:
        return {"threshold": 0.50, "sensitivity": np.nan,
                "specificity": np.nan, "status": "constraint_not_met"}

    # Highest sensitivity, then highest specificity, then highest threshold.
    sens, spec, value = max(rows, key=lambda row: (row[0], row[1], row[2]))
    return {"threshold": value, "sensitivity": sens,
            "specificity": spec, "status": "constraint_met"}


METRIC_DISPLAY_ORDER = [
    "Decision_Threshold", "Evaluation_Weighting", "Weight_Sum",
    "Kish_Effective_N", "AUC-ROC", "AP", "F1", "Precision",
    "Sensitivity", "Specificity",
    "NPV", "Accuracy", "Balanced Accuracy", "MCC", "Brier",
    "TN", "FP", "FN", "TP",
]


def metrics_to_dataframe(results: dict) -> pd.DataFrame:
    """Convert a model -> metrics mapping into a readable result table."""
    rows = [{"Model": model_name, **metrics}
            for model_name, metrics in results.items()]
    return pd.DataFrame(rows)[["Model", *METRIC_DISPLAY_ORDER]]


def show_dataframe(df: pd.DataFrame, title: str) -> None:
    """Show a result table in notebooks, with a plain-text fallback."""
    print(f"\n{title}")
    print("-" * len(title))
    if not SHOW_INLINE_OUTPUTS:
        print(df.to_string(index=False))
        return
    try:
        from IPython.display import display
        display(df.reset_index(drop=True))
    except Exception:
        print(df.to_string(index=False))


def _parameter_space_size(params: dict) -> int:
    size = 1
    for values in params.values():
        size *= len(values)
    return size


def _fit_estimator(estimator, X, y, sample_weight=None):
    """Fit a pipeline and route design weights only to its final classifier."""
    if sample_weight is None:
        return estimator.fit(X, y)
    return estimator.fit(X, y,
                         clf__sample_weight=np.asarray(sample_weight,
                                                       dtype=float))


def _weighted_candidate_score(base_estimator, params, X, y,
                              sample_weight, splits) -> float:
    scores = []
    for fit_idx, val_idx in splits:
        estimator = clone(base_estimator).set_params(**params)
        _fit_estimator(
            estimator, X.iloc[fit_idx], y[fit_idx],
            sample_weight[fit_idx])
        proba = estimator.predict_proba(X.iloc[val_idx])[:, 1]
        scores.append(average_precision_score(
            y[val_idx], proba, sample_weight=sample_weight[val_idx]))
    return float(np.mean(scores))


class ManualWeightedSearchCV:
    """Small Grid/Random search with sampling weights in fit and AP scoring.

    GridSearchCV cannot portably pass fold-specific design weights to scorers
    across the scikit-learn versions used by Kaggle.  This deterministic
    implementation makes the weighting explicit and exposes the familiar
    ``best_*`` interface used by the rest of the analysis.
    """

    def __init__(self, estimator, params, search_kind, cv, n_iter,
                 random_state):
        self.estimator = estimator
        self.params = params
        self.search_kind = search_kind
        self.cv = cv
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y, sample_weight):
        y = np.asarray(y, dtype=int)
        weights = _validated_weights(sample_weight, len(y))
        if self.search_kind == "random":
            candidates = list(ParameterSampler(
                self.params,
                n_iter=min(self.n_iter, _parameter_space_size(self.params)),
                random_state=self.random_state))
        else:
            candidates = list(ParameterGrid(self.params))
        splits = list(self.cv.split(X, y))
        scores = Parallel(n_jobs=-1)(
            delayed(_weighted_candidate_score)(
                self.estimator, params, X, y, weights, splits)
            for params in candidates)
        best_index = int(np.argmax(scores))
        self.best_params_ = candidates[best_index]
        self.best_score_ = float(scores[best_index])
        self.best_estimator_ = clone(self.estimator).set_params(
            **self.best_params_)
        _fit_estimator(self.best_estimator_, X, y, weights)
        self.cv_results_ = {
            "params": candidates,
            "mean_test_score": np.asarray(scores, dtype=float),
        }
        return self

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)


def make_search(spec: dict, y_fit: np.ndarray,
                seed_offset: int = 0,
                weighted_scoring: bool = False):
    inner_cv = StratifiedKFold(
        n_splits=INNER_FOLDS, shuffle=True,
        random_state=RANDOM_STATE + seed_offset,
    )
    estimator = clone(spec["pipeline"])
    if spec.get("dynamic_scale_pos_weight", False):
        n_pos = int(np.sum(np.asarray(y_fit) == 1))
        n_neg = int(np.sum(np.asarray(y_fit) == 0))
        estimator.set_params(clf__scale_pos_weight=n_neg / max(n_pos, 1))
    if weighted_scoring:
        return ManualWeightedSearchCV(
            estimator=estimator,
            params=spec["params"],
            search_kind=spec.get("search", "grid"),
            cv=inner_cv,
            n_iter=RANDOM_SEARCH_ITERATIONS,
            random_state=RANDOM_STATE + seed_offset,
        )
    common = dict(estimator=estimator, scoring=PRIMARY_SCORING, cv=inner_cv,
                  n_jobs=-1, refit=True, return_train_score=False)
    if spec.get("search") == "random":
        return RandomizedSearchCV(
            param_distributions=spec["params"],
            n_iter=min(RANDOM_SEARCH_ITERATIONS,
                       _parameter_space_size(spec["params"])),
            random_state=RANDOM_STATE + seed_offset, **common)
    return GridSearchCV(param_grid=spec["params"], **common)


def _fit_search(search, X, y, sample_weight=None):
    if isinstance(search, ManualWeightedSearchCV):
        if sample_weight is None:
            raise ValueError("Weighted search requires sampling weights.")
        return search.fit(X, y, sample_weight=sample_weight)
    return search.fit(X, y)


def cross_fitted_predict_proba(estimator, X: pd.DataFrame, y: np.ndarray,
                               cv, sample_weight=None) -> np.ndarray:
    """Generate OOF probabilities while fitting design weights fold by fold."""
    output = np.full(len(y), np.nan, dtype=float)
    weights = (None if sample_weight is None else
               _validated_weights(sample_weight, len(y)))
    for fit_idx, val_idx in cv.split(X, y):
        fitted = clone(estimator)
        _fit_estimator(
            fitted, X.iloc[fit_idx], y[fit_idx],
            None if weights is None else weights[fit_idx])
        output[val_idx] = fitted.predict_proba(X.iloc[val_idx])[:, 1]
    if np.isnan(output).any():
        raise AssertionError("Cross-fitting left observations without predictions.")
    return output


def extract_fold_feature_importance(fitted_pipeline, feature_names: list,
                                    regime: str, feature_set: str,
                                    model_name: str, fold: int) -> list:
    """Return fold-level coefficient/importance ranks for stability auditing."""
    classifier = fitted_pipeline.named_steps["clf"]
    if hasattr(classifier, "coef_"):
        raw = np.asarray(classifier.coef_).reshape(-1)
        importance = np.abs(raw)
        nonzero = np.abs(raw) > 1e-12
        source = "absolute_standardized_coefficient"
    elif hasattr(classifier, "feature_importances_"):
        raw = np.asarray(classifier.feature_importances_, dtype=float)
        importance = raw.copy()
        nonzero = importance > 0
        source = "model_feature_importance"
    else:
        return []
    order = np.argsort(-importance, kind="mergesort")
    ranks = np.empty(len(importance), dtype=int)
    ranks[order] = np.arange(1, len(importance) + 1)
    return [{
        "Regime": regime,
        "Feature_Set": feature_set,
        "Model": model_name,
        "Outer_Fold": fold,
        "Feature": feature,
        "Importance_Source": source,
        "Raw_Effect_or_Importance": float(raw[idx]),
        "Absolute_Importance": float(importance[idx]),
        "Rank": int(ranks[idx]),
        "Top_10": bool(ranks[idx] <= 10),
        "Nonzero": bool(nonzero[idx]),
    } for idx, feature in enumerate(feature_names)]


# ── 7. NESTED CROSS-VALIDATION ON TRAINING DATA ONLY ──────────────────────
def run_nested_cv(model_specs: dict, X_train: pd.DataFrame,
                  y_train: np.ndarray, feature_set_name: str,
                  resampling_mode: str,
                  original_indices: np.ndarray = None,
                  sample_weight=None,
                  evaluation_weight=None) -> tuple:
    outer_cv = StratifiedKFold(
        n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    outer_splits = list(outer_cv.split(X_train, y_train))
    results, probas, tuning_rows, threshold_rows = {}, {}, [], []
    stability_rows = []
    if original_indices is None:
        original_indices = np.asarray(X_train.index)
    fit_weights = (None if sample_weight is None else
                   _validated_weights(sample_weight, len(y_train)))
    metric_weights = (fit_weights if evaluation_weight is None else
                      _validated_weights(evaluation_weight, len(y_train)))

    for model_name, spec in model_specs.items():
        started = time.time()
        oof_proba = np.full(len(y_train), np.nan, dtype=float)
        oof_threshold = np.full(len(y_train), np.nan, dtype=float)
        fold_statuses = []
        for fold, (fit_idx, val_idx) in enumerate(outer_splits, start=1):
            search = make_search(
                spec, y_train[fit_idx], seed_offset=fold,
                weighted_scoring=fit_weights is not None)
            _fit_search(
                search, X_train.iloc[fit_idx], y_train[fit_idx],
                None if fit_weights is None else fit_weights[fit_idx])
            oof_proba[val_idx] = search.predict_proba(
                X_train.iloc[val_idx]
            )[:, 1]
            threshold_cv = StratifiedKFold(
                n_splits=INNER_FOLDS, shuffle=True,
                random_state=RANDOM_STATE + 10000 + fold)
            inner_oof_proba = cross_fitted_predict_proba(
                clone(search.best_estimator_), X_train.iloc[fit_idx],
                y_train[fit_idx], cv=threshold_cv,
                sample_weight=(None if fit_weights is None else
                               fit_weights[fit_idx]))
            threshold_info = derive_screening_threshold(
                y_train[fit_idx], inner_oof_proba,
                sample_weight=(None if metric_weights is None else
                               metric_weights[fit_idx]))
            oof_threshold[val_idx] = threshold_info["threshold"]
            fold_statuses.append(threshold_info["status"])
            tuning_rows.append({
                "Resampling_Mode": resampling_mode,
                "Feature_Set": feature_set_name,
                "Model": model_name,
                "Outer_Fold": fold,
                "Inner_Best_AP": round(float(search.best_score_), 6),
                "Inner_AP_Weighting": (
                    "sampling_design_weighted" if fit_weights is not None
                    else "unweighted"),
                "Best_Params": json.dumps(search.best_params_, sort_keys=True),
            })
            threshold_rows.append({
                "Resampling_Mode": resampling_mode,
                "Feature_Set": feature_set_name,
                "Model": model_name,
                "Outer_Fold": fold,
                "Decision_Threshold": round(
                    float(threshold_info["threshold"]), 8),
                "Inner_Threshold_Sensitivity": _round_metric(
                    threshold_info["sensitivity"]),
                "Inner_Threshold_Specificity": _round_metric(
                    threshold_info["specificity"]),
                "Specificity_Floor": SPECIFICITY_FLOOR,
                "Constraint_Status": threshold_info["status"],
                "Threshold_Source": "outer_training_only_inner_CV",
                "Threshold_Metric_Weighting": (
                    "sampling_design_weighted" if metric_weights is not None
                    else "unweighted"),
                "Outer_Training_Indices": json.dumps(
                    [int(i) for i in original_indices[fit_idx]]),
                "Outer_Validation_Indices": json.dumps(
                    [int(i) for i in original_indices[val_idx]]),
            })
            stability_rows.extend(extract_fold_feature_importance(
                search.best_estimator_, list(X_train.columns),
                resampling_mode, feature_set_name, model_name, fold))
            elapsed = time.time() - started
            eta = elapsed / fold * (OUTER_FOLDS - fold)
            print(f"  {feature_set_name} | {model_name} | "
                  f"outer fold {fold}/{OUTER_FOLDS} | "
                  f"inner best AP={search.best_score_:.4f} | "
                  f"threshold={threshold_info['threshold']:.4f} | "
                  f"constraint={threshold_info['status']} | ETA={eta:.1f}s | "
                  f"params={search.best_params_}")

        if np.isnan(oof_proba).any():
            raise AssertionError("Nested CV left samples without OOF predictions.")
        if np.isnan(oof_threshold).any():
            raise AssertionError("Nested CV left samples without thresholds.")
        y_pred = (oof_proba >= oof_threshold).astype(int)
        results[model_name] = calculate_metrics(
            y_train, oof_proba, oof_threshold,
            sample_weight=metric_weights)
        default_metrics = calculate_metrics(
            y_train, oof_proba, 0.50, sample_weight=metric_weights)
        unweighted_metrics = calculate_metrics(
            y_train, oof_proba, oof_threshold)
        default_unweighted_metrics = calculate_metrics(
            y_train, oof_proba, 0.50)
        probas[model_name] = {
            "y_true": y_train.copy(), "y_proba": oof_proba,
            "y_pred": y_pred, "sample_thresholds": oof_threshold,
            "default_metrics": default_metrics,
            "weighted_metrics": results[model_name],
            "unweighted_metrics": unweighted_metrics,
            "default_unweighted_metrics": default_unweighted_metrics,
            "sample_weight": (None if metric_weights is None else
                              metric_weights.copy()),
            "constraint_status": (
                "constraint_met" if all(s == "constraint_met"
                                        for s in fold_statuses)
                else "constraint_not_met"),
        }
        elapsed = time.time() - started
        print(f"  Completed {model_name} in {elapsed:.1f}s: "
              f"AP={results[model_name]['AP']:.4f}, "
              f"MCC={results[model_name]['MCC']:.4f}")

    return (results, probas, pd.DataFrame(tuning_rows),
            pd.DataFrame(threshold_rows), pd.DataFrame(stability_rows))


# ── 8. ONE-TIME INTERNAL HOLDOUT EVALUATION ───────────────────────────────
def fit_and_evaluate_holdout(model_specs: dict, X_train: pd.DataFrame,
                             y_train: np.ndarray, X_holdout: pd.DataFrame,
                             y_holdout: np.ndarray,
                             subject_holdout: pd.Series,
                             resampling_mode: str,
                             sample_weight=None,
                             threshold_weight=None,
                             holdout_weight=None) -> tuple:
    """Tune on training only, then evaluate once on original holdout samples."""
    rows, final_models, holdout_metrics, final_tuning = [], {}, {}, []
    threshold_rows = []
    fit_weights = (None if sample_weight is None else
                   _validated_weights(sample_weight, len(y_train)))
    threshold_weights = (
        fit_weights if threshold_weight is None else
        _validated_weights(threshold_weight, len(y_train)))
    holdout_weights = (None if holdout_weight is None else
                       _validated_weights(holdout_weight, len(y_holdout)))
    for model_name, spec in model_specs.items():
        search = make_search(
            spec, y_train, seed_offset=100,
            weighted_scoring=fit_weights is not None)
        _fit_search(search, X_train, y_train, fit_weights)
        threshold_cv = StratifiedKFold(
            n_splits=INNER_FOLDS, shuffle=True,
            random_state=RANDOM_STATE + 20000)
        training_oof_proba = cross_fitted_predict_proba(
            clone(search.best_estimator_), X_train, y_train,
            cv=threshold_cv, sample_weight=fit_weights)
        threshold_info = derive_screening_threshold(
            y_train, training_oof_proba,
            sample_weight=threshold_weights)
        proba = search.predict_proba(X_holdout)[:, 1]
        pred = (proba >= threshold_info["threshold"]).astype(int)
        final_models[model_name] = search.best_estimator_
        holdout_metrics[model_name] = calculate_metrics(
            y_holdout, proba, threshold_info["threshold"],
            sample_weight=holdout_weights)
        holdout_unweighted = calculate_metrics(
            y_holdout, proba, threshold_info["threshold"])
        final_tuning.append({
            "Model": model_name,
            "Best_Training_AP": round(float(search.best_score_), 6),
            "Best_Params": json.dumps(search.best_params_, sort_keys=True),
            "Training_Derived_Threshold": round(
                float(threshold_info["threshold"]), 8),
            "Threshold_Constraint_Status": threshold_info["status"],
            "Holdout_Default_0.50_Metrics": json.dumps(
                calculate_metrics(
                    y_holdout, proba, 0.50,
                    sample_weight=holdout_weights), sort_keys=True),
            "Holdout_Unweighted_Threshold_Metrics": json.dumps(
                holdout_unweighted, sort_keys=True),
        })
        threshold_rows.append({
            "Resampling_Mode": resampling_mode,
            "Feature_Set": "Primary_reduced",
            "Model": model_name,
            "Outer_Fold": "final_training_to_holdout",
            "Decision_Threshold": round(
                float(threshold_info["threshold"]), 8),
            "Inner_Threshold_Sensitivity": _round_metric(
                threshold_info["sensitivity"]),
            "Inner_Threshold_Specificity": _round_metric(
                threshold_info["specificity"]),
            "Specificity_Floor": SPECIFICITY_FLOOR,
            "Constraint_Status": threshold_info["status"],
            "Threshold_Source": "733_training_samples_only_inner_CV",
            "Threshold_Metric_Weighting": (
                "sampling_design_weighted"
                if threshold_weights is not None else "unweighted"),
        })
        metrics = holdout_metrics[model_name]
        print(f"  Holdout | {model_name} | "
              f"AUC={metrics['AUC-ROC']:.4f} | "
              f"AP={metrics['AP']:.4f} | "
              f"Sensitivity={metrics['Sensitivity']:.4f} | "
              f"Specificity={metrics['Specificity']:.4f} | "
              f"MCC={metrics['MCC']:.4f} | "
              f"params={search.best_params_}")
        for original_index, subject, actual, predicted, probability in zip(
                X_holdout.index, subject_holdout, y_holdout, pred, proba):
            rows.append({
                "Regime": resampling_mode,
                "Model": model_name,
                "Original_Row_Index": int(original_index),
                "Subject_ID": str(subject),
                "Actual": int(actual),
                "Predicted": int(predicted),
                "Probability_GAD": round(float(probability), 6),
                "Decision_Threshold": round(
                    float(threshold_info["threshold"]), 8),
                "Correct": bool(actual == predicted),
            })
    return (pd.DataFrame(rows), final_models, holdout_metrics,
            pd.DataFrame(final_tuning), pd.DataFrame(threshold_rows))


# ── 9. SAVE FIGURE HELPER ──────────────────────────────────────────────────
def _save(fig, name: str, aliases=None) -> None:
    """Save figures efficiently in test mode and fully in final mode."""

    if QUICK_TEST:
        output_names = [name]
        formats = ("png",)
        save_dpi = 100
    else:
        output_names = [name, *(aliases or [])]
        formats = ("svg", "png")
        save_dpi = DPI

    for output_name in output_names:
        for fmt in formats:
            path = os.path.join(
                OUTPUT_DIR,
                f"{output_name}.{fmt}"
            )
            fig.savefig(
                path,
                dpi=save_dpi,
                bbox_inches="tight",
                format=fmt
            )

    if SHOW_INLINE_OUTPUTS:
        try:
            from IPython.display import display
            display(fig)
        except Exception:
            plt.show()

    plt.close(fig)

    saved_names = ", ".join(
        f"{item}.{fmt}"
        for item in output_names
        for fmt in formats
    )

    status = (
        "saved and displayed"
        if SHOW_INLINE_OUTPUTS
        else "saved"
    )

    print(
        f"  {status.capitalize()} : {saved_names}"
    )


# ── 10. FIGURE 1 — CLASS DISTRIBUTION ────────────────────────────────────
def plot_class_distribution(y: np.ndarray) -> None:
    apply_style()
    counts = [int(np.sum(y == 0)), int(np.sum(y == 1))]
    labels = ["Negative (No GAD)", "Positive (GAD)"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts,
                  color=["#B4B2A9", "#3B8BD4"],
                  edgecolor="white", linewidth=0.8, zorder=3)

    for bar, v in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 5,
            f"n = {v}\n({v / len(y) * 100:.1f}%)",
            ha="center", va="bottom",
            fontsize=FONT_TICK, fontweight="bold"
        )

    ax.set_ylabel("Number of children", fontsize=FONT_AXIS)
    ax.set_xlabel("Observed GAD ", fontsize=FONT_AXIS)
    ax.set_ylim(0, max(counts) * 1.25)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save(fig, "fig01_class_distribution")


# ── 11. FIGURE 2 — METRIC COMPARISON ─────────────────────────────────────
def plot_metric_comparison(results: dict) -> None:
    metrics = ["AUC-ROC", "AP", "F1", "Precision",
               "Sensitivity", "Specificity", "MCC"]
    x = np.arange(len(metrics))
    w = 0.80 / max(len(results), 1)

    apply_style()
    fig, ax = plt.subplots(figsize=(15, 6))
    for i, (name, mdata) in enumerate(results.items()):
        vals   = [mdata[m] for m in metrics]
        offset = (i - (len(results) - 1) / 2) * w
        bars   = ax.bar(x + offset, vals, w,
                        label=name, color=PALETTE[name],
                        edgecolor="white", linewidth=0.4, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.004,
                f"{v:.3f}",
                ha="center", va="bottom",
                fontsize=8.5, rotation=90
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=FONT_TICK)
    ax.set_ylim(0.0, 1.15)
    ax.set_ylabel("Score", fontsize=FONT_AXIS)
    ax.set_xlabel("Evaluation Metric", fontsize=FONT_AXIS)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, loc="upper right")
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save(fig, "fig02_metric_comparison")


# ── 12. FIGURE 3 — ROC CURVES ────────────────────────────────────────────
def plot_roc_curves(probas: dict) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 7))

    for name, data in probas.items():
        weights = data.get("sample_weight")
        fpr, tpr, _ = roc_curve(
            data["y_true"], data["y_proba"], sample_weight=weights)
        auc = roc_auc_score(
            data["y_true"], data["y_proba"], sample_weight=weights)
        ax.plot(fpr, tpr, color=PALETTE[name], linewidth=2.2,
                label=f"{name}  (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4,
            label="Random classifier  (AUC = 0.50)")
    ax.fill_between([0, 1], [0, 1], alpha=0.03, color="gray")
    ax.set_xlabel("False positive rate (1 − specificity)", fontsize=FONT_AXIS)
    ax.set_ylabel("True positive rate (sensitivity)", fontsize=FONT_AXIS)
    ax.legend(fontsize=FONT_LEGEND, loc="lower right", framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    _save(fig, "fig03_roc_curves")


# ── 13. FIGURE 4 — PRECISION-RECALL CURVES ───────────────────────────────
def plot_pr_curves(probas: dict) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 7))

    for name, data in probas.items():
        weights = data.get("sample_weight")
        prec, rec, _ = precision_recall_curve(
            data["y_true"], data["y_proba"], sample_weight=weights
        )
        ap = average_precision_score(
            data["y_true"], data["y_proba"], sample_weight=weights)
        ax.plot(rec, prec, color=PALETTE[name], linewidth=2.2,
                label=f"{name}  (AP = {ap:.4f})")

    first = probas[list(probas.keys())[0]]
    first_weights = first.get("sample_weight")
    base = (float(np.mean(first["y_true"])) if first_weights is None else
            float(np.average(first["y_true"], weights=first_weights)))
    ax.axhline(base, color="gray", linestyle="--", linewidth=1.2,
               label=f"No-skill baseline  (prevalence = {base:.2f})")
    ax.set_xlabel("Recall (sensitivity)", fontsize=FONT_AXIS)
    ax.set_ylabel("Precision (positive predictive value)", fontsize=FONT_AXIS)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    _save(fig, "fig04_pr_curves")


# ── 14. FIGURE 5 — CONFUSION MATRICES ────────────────────────────────────
def plot_confusion_matrices(probas: dict) -> None:
    """Save one publication-ready confusion matrix per model."""
    apply_style()
    cmap_map = {
        "L1 Logistic Regression": "Greens",
        "Decision Tree": "Oranges",
        "Random Forest": "Blues",
        "XGBoost": "Purples",
    }

    for name, data in probas.items():
        fig, ax = plt.subplots(figsize=(5.0, 4.6))
        weights = data.get("sample_weight")
        cm = confusion_matrix(
            data["y_true"], data["y_pred"], labels=[0, 1],
            sample_weight=weights)
        thresh = cm.max() / 2.0
        image = ax.imshow(
            cm, cmap=cmap_map.get(name, "Blues"), aspect="equal",
            interpolation="nearest")
        #
        # YENİ
        for i in range(2):
            for j in range(2):
                label = (str(int(cm[i, j])) if weights is None else
                         f"{cm[i, j]:.1f}")
                # ax.text(
                #     j, i, label, ha="center", va="center",
                #     fontsize=FONT_TITLE + 4, fontweight="bold",
                #     color="white" if cm[i, j] > thresh else "black")

                # # YENİ
                ax.text(
                    j, i, label, ha="center", va="center",
                    fontsize=FONT_TITLE + 1,
                    fontweight="normal",
                    color="white" if cm[i, j] > thresh else "black")
                        
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        # YENİ
        ax.set_xticklabels(
            ["No GAD", "GAD"],
            fontsize=FONT_TICK + 2,
            fontweight="bold",
        )
        ax.set_yticklabels(
            ["No GAD", "GAD"],
            fontsize=FONT_TICK + 2,
            fontweight="bold",
        )
        
        ax.set_xlabel(
            "Predicted GAD ",
            fontsize=FONT_AXIS,
            labelpad=8,
        )
        ax.set_ylabel(
            "Observed GAD ",
            fontsize=FONT_AXIS,
            labelpad=8,
        )
        
        ax.tick_params(axis="both", which="major", labelsize=FONT_TICK+4, pad=4)
        ax.grid(False)
        
        colour_bar = fig.colorbar(
            image,
            ax=ax,
            fraction=0.050,
            pad=0.05,
        )
        # colour_bar.set_label(
        #     "Weighted count" if weights is not None else "Number of children",
        #     fontsize=FONT_AXIS,
        #     labelpad=8,
        # )
        # YENİ
        colour_bar.set_label(
            "Weighted count" if weights is not None else "Number of children",
            fontsize=FONT_AXIS + 1,
            labelpad=10,
        )
        # colour_bar.ax.tick_params(labelsize=FONT_TICK)
        # YENİ
        colour_bar.ax.tick_params(
            labelsize=FONT_TICK + 2,
            pad=4,
        )
        # plt.tight_layout(pad=0.5)
        # YENİ
        fig.subplots_adjust(
            left=0.34,    # Observed GAD status için sol boşluk
            bottom=0.20,  # Predicted GAD status için alt boşluk
            right=0.86,   # Colorbar için sağ boşluk
            top=0.98,
        )

        tag = MODEL_SHORT[name].lower()
        aliases = (["manuscript_confusion_matrix_selected_model"]
                   if name == "Random Forest" else None)
        _save(fig, f"fig05_confusion_matrix_{tag}", aliases=aliases)


# ── 15. FIGURE 6 — RADAR CHART ────────────────────────────────────────────
def plot_radar(results: dict) -> None:
    metrics = ["AUC-ROC", "F1", "Sensitivity", "Specificity", "MCC"]
    N       = len(metrics)
    angles  = [n / N * 2 * np.pi for n in range(N)] + [0.0]

    apply_style()
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for name, mdata in results.items():
        vals = [mdata[m] for m in metrics] + [mdata[metrics[0]]]
        ax.plot(angles, vals, "o-", linewidth=2.2,
                label=name, color=PALETTE[name])
        ax.fill(angles, vals, alpha=0.10, color=PALETTE[name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=FONT_TICK)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"],
                       fontsize=9, color="gray")
    ax.legend(fontsize=FONT_LEGEND,
              bbox_to_anchor=(1.35, 1.12), framealpha=0.9)
    ax.grid(color=PALETTE["grid"], linewidth=0.8)
    plt.tight_layout()
    _save(fig, "fig06_radar")


# ── 16. CALIBRATION, UNCERTAINTY, AND CLINICAL-THRESHOLD FIGURES ───────────
def weighted_reliability_points(y_true: np.ndarray, y_proba: np.ndarray,
                                sample_weight=None,
                                n_bins: int = CALIBRATION_BINS) -> tuple:
    """Return equal-width reliability points and bin weight masses."""
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    weights = _validated_weights(sample_weight, len(y_true))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_proba, edges[1:-1], right=True)
    mean_predicted, observed, masses = [], [], []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if np.any(mask):
            bin_weights = weights[mask]
            masses.append(float(bin_weights.sum()))
            mean_predicted.append(float(np.average(
                y_proba[mask], weights=bin_weights)))
            observed.append(float(np.average(
                y_true[mask], weights=bin_weights)))
    return (np.asarray(mean_predicted), np.asarray(observed),
            np.asarray(masses))


def expected_calibration_error(y_true: np.ndarray, y_proba: np.ndarray,
                               sample_weight=None,
                               n_bins: int = CALIBRATION_BINS) -> float:
    """Sampling-design-weighted equal-width ECE."""
    predicted, observed, masses = weighted_reliability_points(
        y_true, y_proba, sample_weight=sample_weight, n_bins=n_bins)
    if not len(masses):
        return np.nan
    ece = np.sum(masses / masses.sum() * np.abs(observed - predicted))
    return float(ece)


def calibration_slope_intercept(y_true: np.ndarray,
                                y_proba: np.ndarray,
                                sample_weight=None) -> tuple:
    """Estimate calibration intercept and slope from the probability logit."""
    clipped = np.clip(np.asarray(y_proba, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    try:
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000)
        model.fit(logit, y_true, sample_weight=sample_weight)
    except (TypeError, ValueError):
        model = LogisticRegression(penalty="none", solver="lbfgs",
                                   max_iter=5000)
        model.fit(logit, y_true, sample_weight=sample_weight)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def bootstrap_confidence_intervals(y_true: np.ndarray,
                                   y_proba: np.ndarray,
                                   threshold,
                                   iterations: int,
                                   seed: int,
                                   sample_weight=None) -> pd.DataFrame:
    """Stratified bootstrap CIs for discrimination and threshold metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    threshold_array = np.asarray(threshold, dtype=float)
    if threshold_array.ndim == 0:
        threshold_array = np.full(len(y_true), float(threshold_array))
    if threshold_array.shape != y_true.shape:
        raise ValueError("Bootstrap thresholds must match y_true.")
    weights = (None if sample_weight is None else
               _validated_weights(sample_weight, len(y_true)))
    neg = np.flatnonzero(y_true == 0)
    pos = np.flatnonzero(y_true == 1)
    if not len(neg) or not len(pos):
        raise ValueError("Bootstrap requires both outcome classes.")
    rng = np.random.default_rng(seed)
    point_estimates = calculate_metrics(
        y_true, y_proba, threshold_array, sample_weight=weights)
    samples = {name: [] for name in [
        "AUC-ROC", "AP", "Sensitivity", "Specificity", "Precision",
        "F1", "Accuracy", "MCC", "Brier"]}
    for _ in range(iterations):
        idx = np.concatenate([
            rng.choice(neg, size=len(neg), replace=True),
            rng.choice(pos, size=len(pos), replace=True),
        ])
        rng.shuffle(idx)
        metrics = calculate_metrics(
            y_true[idx], y_proba[idx], threshold_array[idx],
            sample_weight=(None if weights is None else weights[idx]))
        for metric in samples:
            samples[metric].append(metrics[metric])
    rows = []
    for metric, values in samples.items():
        low, high = np.quantile(values, [0.025, 0.975])
        rows.append({"Metric": metric,
                     "Point_Estimate": float(point_estimates[metric]),
                     "CI_2.5%": float(low),
                     "CI_97.5%": float(high),
                     "Bootstrap_Iterations": iterations,
                     "Bootstrap_Seed": seed})
    return pd.DataFrame(rows)


def build_calibration_and_ci_frames(primary_probas: dict) -> tuple:
    calibration_rows, reliability_rows, ci_frames, seeds = [], [], [], {}
    for model_idx, (model_name, data) in enumerate(primary_probas.items()):
        y_true = np.asarray(data["y_true"])
        y_proba = np.asarray(data["y_proba"])
        weights = data.get("sample_weight")
        intercept, slope = calibration_slope_intercept(
            y_true, y_proba, sample_weight=weights)
        weighted_metrics = calculate_metrics(
            y_true, y_proba, data["sample_thresholds"],
            sample_weight=weights)
        calibration_rows.append({
            "Model": model_name,
            "Evaluation_Set": "Nested-CV OOF development partition",
            "Evaluation_Weighting": weighted_metrics["Evaluation_Weighting"],
            "Brier": weighted_metrics["Brier"],
            "ECE": expected_calibration_error(
                y_true, y_proba, sample_weight=weights),
            "Calibration_Intercept": intercept,
            "Calibration_Slope": slope,
        })
        mean_pred, frac_pos, masses = weighted_reliability_points(
            y_true, y_proba, sample_weight=weights,
            n_bins=CALIBRATION_BINS)
        for bin_number, (predicted, observed, mass) in enumerate(
                zip(mean_pred, frac_pos, masses), start=1):
            reliability_rows.append({
                "Model": model_name, "Nonempty_Bin_Number": bin_number,
                "Mean_Predicted_Probability": float(predicted),
                "Observed_GAD_Rate": float(observed),
                "Bin_Weight_Mass": float(mass),
            })
        seed = RANDOM_STATE + 50000 + model_idx
        seeds[model_name] = seed
        ci = bootstrap_confidence_intervals(
            y_true, y_proba, data["sample_thresholds"],
            BOOTSTRAP_ITERATIONS, seed, sample_weight=weights)
        ci.insert(0, "Model", model_name)
        ci_frames.append(ci)
    return (pd.DataFrame(calibration_rows), pd.DataFrame(reliability_rows),
            pd.concat(ci_frames, ignore_index=True), seeds)


def paired_feature_set_differences(feature_set_probas: dict,
                                   iterations: int = BOOTSTRAP_ITERATIONS,
                                   seed: int = RANDOM_STATE + 61000
                                   ) -> tuple:
    """Paired bootstrap deltas quantify diagnostic-circularity inflation."""
    primary_name = "Primary_reduced"
    comparisons = [
        "Full_features_sensitivity",
        "Diagnostic_only_sensitivity",
    ]
    metric_names = ["AUC-ROC", "AP", "Sensitivity", "Specificity",
                    "MCC", "Brier"]
    primary = feature_set_probas[primary_name]
    rows = []
    seeds = {}
    for comparison_idx, comparison_name in enumerate(comparisons):
        if comparison_name not in feature_set_probas:
            continue
        comparison = feature_set_probas[comparison_name]
        common_models = sorted(set(primary).intersection(comparison))
        for model_idx, model_name in enumerate(common_models):
            reduced = primary[model_name]
            augmented = comparison[model_name]
            y_true = np.asarray(reduced["y_true"], dtype=int)
            if not np.array_equal(y_true, augmented["y_true"]):
                raise AssertionError("Paired ablation outcomes are misaligned.")
            weights = reduced.get("sample_weight")
            if weights is not None:
                weights = _validated_weights(weights, len(y_true))
            point_reduced = calculate_metrics(
                y_true, reduced["y_proba"], reduced["sample_thresholds"],
                sample_weight=weights)
            point_augmented = calculate_metrics(
                y_true, augmented["y_proba"],
                augmented["sample_thresholds"], sample_weight=weights)
            local_seed = seed + comparison_idx * 100 + model_idx
            seeds[f"{comparison_name}|{model_name}"] = local_seed
            rng = np.random.default_rng(local_seed)
            neg = np.flatnonzero(y_true == 0)
            pos = np.flatnonzero(y_true == 1)
            draws = {metric: [] for metric in metric_names}
            for _ in range(iterations):
                idx = np.concatenate([
                    rng.choice(neg, size=len(neg), replace=True),
                    rng.choice(pos, size=len(pos), replace=True),
                ])
                rng.shuffle(idx)
                local_weights = None if weights is None else weights[idx]
                reduced_metrics = calculate_metrics(
                    y_true[idx], reduced["y_proba"][idx],
                    reduced["sample_thresholds"][idx],
                    sample_weight=local_weights)
                augmented_metrics = calculate_metrics(
                    y_true[idx], augmented["y_proba"][idx],
                    augmented["sample_thresholds"][idx],
                    sample_weight=local_weights)
                for metric in metric_names:
                    draws[metric].append(
                        augmented_metrics[metric] - reduced_metrics[metric])
            for metric in metric_names:
                low, high = np.quantile(draws[metric], [0.025, 0.975])
                rows.append({
                    "Model": model_name,
                    "Reference_Feature_Set": primary_name,
                    "Comparison_Feature_Set": comparison_name,
                    "Metric": metric,
                    "Reference_Point_Estimate": point_reduced[metric],
                    "Comparison_Point_Estimate": point_augmented[metric],
                    "Delta_Comparison_Minus_Reference": round(
                        point_augmented[metric] - point_reduced[metric], 6),
                    "Delta_CI_2.5%": float(low),
                    "Delta_CI_97.5%": float(high),
                    "Bootstrap_Iterations": iterations,
                    "Bootstrap_Seed": local_seed,
                    "Interpretation": (
                        "Positive delta is worse" if metric == "Brier" else
                        "Positive delta is better"),
                })
    return pd.DataFrame(rows), seeds


def summarize_feature_stability(stability_rows: pd.DataFrame) -> pd.DataFrame:
    if stability_rows.empty:
        return pd.DataFrame()
    summary = (stability_rows.groupby(
        ["Regime", "Feature_Set", "Model", "Feature"], as_index=False)
        .agg(Folds_Evaluated=("Outer_Fold", "nunique"),
             Mean_Absolute_Importance=("Absolute_Importance", "mean"),
             Median_Rank=("Rank", "median"),
             Top_10_Frequency=("Top_10", "mean"),
             Nonzero_Frequency=("Nonzero", "mean")))
    summary["Stable_Top_10_At_Least_70pct"] = (
        summary["Top_10_Frequency"] >= 0.70)
    return summary.sort_values(
        ["Regime", "Feature_Set", "Model", "Median_Rank",
         "Mean_Absolute_Importance"],
        ascending=[True, True, True, True, False]).reset_index(drop=True)


def build_clinical_impact_table(results: dict,
                                population_size: int = 1000) -> pd.DataFrame:
    """Translate weighted confusion results to expected counts per 1,000."""
    rows = []
    for model_name, metrics in results.items():
        total = sum(float(metrics[key]) for key in ["TN", "FP", "FN", "TP"])
        scale = population_size / total
        rows.append({
            "Model": model_name,
            "Population_Basis": population_size,
            "Expected_True_Negatives": round(float(metrics["TN"]) * scale, 1),
            "Expected_False_Positives": round(float(metrics["FP"]) * scale, 1),
            "Expected_Missed_GAD_Positive": round(float(metrics["FN"]) * scale, 1),
            "Expected_Detected_GAD_Positive": round(float(metrics["TP"]) * scale, 1),
            "Expected_Referrals": round(
                (float(metrics["TP"]) + float(metrics["FP"])) * scale, 1),
            "Decision_Threshold": metrics["Decision_Threshold"],
            "Evaluation_Weighting": metrics["Evaluation_Weighting"],
        })
    return pd.DataFrame(rows)


def plot_resampling_comparison(regime_results: dict) -> None:
    """Save one imbalance-strategy comparison chart per metric."""
    metrics = ["Sensitivity", "Specificity", "AP", "MCC"]
    rows = []
    for regime, result_map in regime_results.items():
        for model, values in result_map.items():
            for metric in metrics:
                rows.append({"Regime": regime, "Model": model,
                             "Metric": metric, "Value": values[metric]})
    frame = pd.DataFrame(rows)
    apply_style()
    regimes = list(regime_results)
    models = list(next(iter(regime_results.values())))
    width = 0.82 / max(len(regimes), 1)
    colors = ["#2C7FB8", "#8C8C8C", "#1D9E75", "#C76D3A"]
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(9.5, 6.0))
        subset = frame[frame["Metric"] == metric]
        x = np.arange(len(models))
        for regime_idx, regime in enumerate(regimes):
            offset = (regime_idx - (len(regimes) - 1) / 2) * width
            values = [float(subset[(subset["Regime"] == regime) &
                                   (subset["Model"] == model)]["Value"].iloc[0])
                      for model in models]
            bars = ax.bar(
                x + offset, values, width=width, label=regime,
                color=colors[regime_idx % len(colors)],
                edgecolor="white", linewidth=0.5, zorder=3)
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.015, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=10, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_SHORT[m] for m in models],
                           fontsize=FONT_TICK)
        ax.set_ylim(-0.05 if metric == "MCC" else 0.0, 1.05)
        ax.set_xlabel("Model", fontsize=FONT_AXIS)
        ax.set_ylabel(f"Nested-CV {metric}", fontsize=FONT_AXIS)
        ax.legend(fontsize=FONT_LEGEND, loc="best", framealpha=0.9)
        ax.set_axisbelow(True)
        plt.tight_layout()
        _save(fig, f"fig09_imbalance_strategy_{metric.lower()}")


def plot_threshold_tradeoff(primary_probas: dict) -> None:
    """Save one sensitivity-specificity threshold chart per model."""
    apply_style()
    for model_name, data in primary_probas.items():
        fig, ax = plt.subplots(figsize=(8.2, 6.2))
        thresholds = np.linspace(0.0, 1.0, 201)
        sensitivities, specificities = [], []
        for threshold in thresholds:
            metrics = calculate_metrics(data["y_true"], data["y_proba"],
                                        threshold,
                                        sample_weight=data.get("sample_weight"))
            sensitivities.append(metrics["Sensitivity"])
            specificities.append(metrics["Specificity"])
        median_threshold = float(np.median(data["sample_thresholds"]))
        ax.plot(thresholds, sensitivities, label="Sensitivity", color="#C73E1D")
        ax.plot(thresholds, specificities, label="Specificity", color="#1D6FA5")
        ax.axhline(SPECIFICITY_FLOOR, color="black", linestyle="--",
                   linewidth=1, label="Specificity floor")
        ax.axvline(median_threshold, color=PALETTE[model_name],
                   linestyle=":", linewidth=2,
                   label=f"Median fold threshold={median_threshold:.3f}")
        ax.set_xlabel("Decision threshold", fontsize=FONT_AXIS)
        ax.set_ylabel("Classification score", fontsize=FONT_AXIS)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.03)
        ax.legend(fontsize=FONT_LEGEND, loc="best", framealpha=0.9)
        ax.set_axisbelow(True)
        plt.tight_layout()
        tag = MODEL_SHORT[model_name].lower()
        _save(fig, f"fig10_threshold_tradeoff_{tag}")


def plot_reliability(primary_probas: dict) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2,
            label="Perfect calibration")
    for model_name, data in primary_probas.items():
        mean_pred, frac_pos, _ = weighted_reliability_points(
            data["y_true"], data["y_proba"],
            sample_weight=data.get("sample_weight"),
            n_bins=CALIBRATION_BINS)
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=2,
                color=PALETTE[model_name], label=model_name)
    ax.set_xlabel("Mean predicted probability of GAD", fontsize=FONT_AXIS)
    ax.set_ylabel("Observed weighted GAD proportion", fontsize=FONT_AXIS)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=FONT_LEGEND)
    plt.tight_layout()
    _save(fig, "fig11_calibration_reliability")


# ── 16. XAI — FITTED PRIMARY MODELS ONLY ──────────────────────────────────
def transformed_matrix(fitted_pipeline, X: pd.DataFrame) -> np.ndarray:
    """Apply fitted preprocessing without applying SMOTE to evaluation data."""
    values = fitted_pipeline.named_steps["imputer"].transform(X)
    if "scaler" in fitted_pipeline.named_steps:
        values = fitted_pipeline.named_steps["scaler"].transform(values)
    return values

def _save_importance_figure(
        series: pd.Series,
        model_name: str,
        x_label: str,
        output_name: str,
        aliases=None) -> None:
    """Save one clean Top-15 feature-importance chart."""
    apply_style()

    fig, ax = plt.subplots(figsize=(10.0, 7.4))
    labels = [to_en(name) for name in series.index]
    positions = np.arange(len(labels))

    bars = ax.barh(
        positions,
        series.values,
        height=0.72,
        color=PALETTE[model_name],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(
        labels,
        fontsize=22
    )
    ax.invert_yaxis()
    
    maximum = float(series.max())
    
    # Sağdaki sayılar büyüdüğü için biraz daha fazla boşluk bırak.
    ax.set_xlim(0, maximum * 1.22)
    
    for bar, value in zip(bars, series.values):
        ax.text(
            value + maximum * 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            ha="left",
            va="center",
            fontsize=22,
            color="#242424",
        )
    
    ax.set_xlabel(
        x_label,
        fontsize=22,
        labelpad=10
    )
    
    ax.set_ylabel("")
    
    ax.tick_params(
        axis="x",
        labelsize=22,
        pad=5
    )
    
    ax.tick_params(
        axis="y",
        labelsize=22,
        pad=6
    )

    ax.grid(axis="x", color="#D9D7D0", linewidth=0.7, alpha=0.70)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    fig.subplots_adjust(
        left=0.42,   # Uzun PAPA özellik adları için alan
        right=0.93,  # Çubuk sonu değerleri için alan
        bottom=0.15,
        top=0.98,
    )
    _save(fig, output_name, aliases=aliases)


def plot_feature_importance(final_models: dict, feature_names: list) -> None:
    """Save separate LR and Decision Tree Top-15 importance figures."""
    for model_name in ("L1 Logistic Regression", "Decision Tree"):
        clf = final_models[model_name].named_steps["clf"]

        values = (
            np.abs(clf.coef_[0])
            if model_name == "L1 Logistic Regression"
            else clf.feature_importances_
        )

        importance = pd.Series(
            values,
            index=feature_names,
        ).sort_values(ascending=False).head(15)

        x_label = (
            "Absolute standardized coefficient"
            if model_name == "L1 Logistic Regression"
            else "Impurity-based feature importance"
        )

        tag = MODEL_SHORT[model_name].lower()
        _save_importance_figure(
            importance,
            model_name,
            x_label,
            f"fig07_feature_importance_{tag}_primary_reduced",
            aliases=[f"manuscript_figure02_{tag}_top15_primary_reduced"],
        )


def plot_ensemble_feature_importance_supplement(
        final_models: dict,
        feature_names: list) -> None:
    """Save separate RF and XGBoost Top-15 supplementary figures."""
    for model_name in ("Random Forest", "XGBoost"):
        if model_name not in final_models:
            continue

        clf = final_models[model_name].named_steps["clf"]
        importance = pd.Series(
            clf.feature_importances_,
            index=feature_names,
        ).sort_values(ascending=False).head(15)

        tag = MODEL_SHORT[model_name].lower()
        _save_importance_figure(
            importance,
            model_name,
            "Impurity-based feature importance",
            f"figS01_feature_importance_{tag}_primary_reduced",
        )


# ── 17. XAI — TUNED DECISION TREE RULES ───────────────────────────────────
def plot_decision_tree_rules(final_models: dict,
                             feature_names: list) -> None:
    pipeline = final_models["Decision Tree"]
    clf = pipeline.named_steps["clf"]
    short_names = [to_en(name, max_len=32) for name in feature_names]
    apply_style()
    fig, ax = plt.subplots(figsize=(28, 11))
    plot_tree(
        clf, feature_names=short_names, class_names=["No GAD", "GAD"],
        filled=True, rounded=True, max_depth=4, fontsize=8, ax=ax,
        impurity=False, precision=2,
    )
    plt.tight_layout()
    _save(fig, "fig08_decision_tree_rules_primary_reduced")


def positive_class_shap_array(raw_values) -> np.ndarray:
    """Normalize SHAP output across supported SHAP versions."""
    if isinstance(raw_values, list):
        return np.asarray(raw_values[1])
    values = np.asarray(raw_values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def _draw_shap_dot_panel(
    ax,
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list,
    random_seed: int,
    ytick_fontsize: int = 20,
    xlabel_fontsize: int = 20,
    ylabel_fontsize: int = 20,
    xtick_fontsize: int = 16,
    point_size: int = 60,
    point_alpha: float = 0.80,
    custom_style: bool = False
) -> None:
    """Draw a deterministic SHAP dot/beeswarm figure on one axis."""
    mean_abs = np.abs(shap_values).mean(axis=0)

    top_indices = np.argsort(mean_abs)[::-1][:15]

    if custom_style:
        jitter_std = 0.11
        edgecolors = "white"
        linewidths = 0.35
        zero_line_color = "#666666"
        zero_line_width = 1.0
        axis_facecolor = "#FCFCFC"
        ytick_weight = "normal"
        label_weight = "normal"
        show_x_grid = True
    else:
        jitter_std = 0.095
        edgecolors = "none"
        linewidths = 0.0
        zero_line_color = "#777777"
        zero_line_width = 0.8
        axis_facecolor = "white"
        ytick_weight = "normal"
        label_weight = "normal"
        show_x_grid = False

    rng = np.random.default_rng(random_seed)
    cmap = plt.get_cmap("coolwarm")

    ax.set_facecolor(axis_facecolor)

    for y_position, feature_index in enumerate(top_indices):
        x_values = np.asarray(shap_values[:, feature_index], dtype=float)
        colour_values = np.asarray(feature_values[:, feature_index], dtype=float)
        finite = np.isfinite(colour_values)

        if finite.any():
            low, high = np.nanpercentile(colour_values[finite], [5, 95])
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                normalized = np.full(len(colour_values), 0.5)
            else:
                normalized = np.clip(
                    (colour_values - low) / (high - low),
                    0.0,
                    1.0
                )
                normalized[~finite] = 0.5
        else:
            normalized = np.full(len(colour_values), 0.5)

        jitter = rng.normal(0.0, jitter_std, size=len(x_values))

        ax.scatter(
            x_values,
            y_position + jitter,
            c=cmap(normalized),
            s=point_size,
            alpha=point_alpha,
            edgecolors=edgecolors,
            linewidths=linewidths,
            rasterized=True,
            zorder=3
        )

    ax.axvline(
        0.0,
        color=zero_line_color,
        linewidth=zero_line_width,
        linestyle="--",
        alpha=0.95,
        zorder=2
    )

    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels(
        [to_en(feature_names[index]) for index in top_indices],
        fontsize=ytick_fontsize,
        fontweight=ytick_weight
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "SHAP value (impact on GAD prediction)",
        fontsize=xlabel_fontsize,
        fontweight=label_weight,
        labelpad=10
    )

    ax.set_ylabel(
        "PAPA symptom feature",
        fontsize=ylabel_fontsize,
        fontweight=label_weight,
        labelpad=10
    )

    ax.tick_params(
        axis="x",
        labelsize=xtick_fontsize,
        width=1.0,
        length=5,
        pad=5
    )

    ax.tick_params(
        axis="y",
        length=0,
        pad=6
    )

    if show_x_grid:
        ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.25)
        ax.grid(axis="y", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#9A9A9A")

    ax.set_axisbelow(True)
def _add_fixed_shap_colorbar(
    fig,
    cbar_tick_fontsize: int = 16,
    cbar_label_fontsize: int = 20,
    label_weight: str = "normal"
) -> None:
    """Add a fixed-position colorbar so all SHAP beeswarm plots match."""
    cmap = plt.get_cmap("coolwarm")

    scalar_map = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=0.0, vmax=1.0),
        cmap=cmap
    )
    scalar_map.set_array([])

    cax = fig.add_axes([0.90, 0.16, 0.018, 0.68])

    colour_bar = fig.colorbar(
        scalar_map,
        cax=cax
    )

    colour_bar.set_ticks([0.0, 1.0])
    colour_bar.set_ticklabels(["Low", "High"])

    colour_bar.ax.tick_params(
        labelsize=cbar_tick_fontsize,
        width=0.8,
        length=4
    )

    colour_bar.set_label(
        "Feature value",
        fontsize=cbar_label_fontsize,
        fontweight=label_weight
    )

    colour_bar.outline.set_linewidth(0.8)


def _save_shap_beeswarm_figure(
    shap_values: np.ndarray,
    holdout_values: np.ndarray,
    feature_names: list,
    model_name: str,
    random_seed: int,
    filename: str,
    aliases: list | None = None
) -> None:
    """Create and save one SHAP beeswarm figure with a fixed layout."""
    apply_style()

    fig, ax = plt.subplots(figsize=(24, 14))

    fig.subplots_adjust(
        left=0.40,
        right=0.88,
        bottom=0.12,
        top=0.97
    )

    _draw_shap_dot_panel(
        ax,
        shap_values,
        holdout_values,
        feature_names,
        random_seed,
        ytick_fontsize=44,
        xlabel_fontsize=44,
        ylabel_fontsize=44,
        xtick_fontsize=44,
        point_size=190,
        point_alpha=0.90,
        custom_style=True
    )

    _add_fixed_shap_colorbar(
        fig,
        cbar_tick_fontsize=44,
        cbar_label_fontsize=44,
        label_weight="normal"
    )

    _save(
        fig,
        filename,
        aliases=aliases if aliases is not None else []
    )


def _draw_shap_bar_panel(ax, shap_values: np.ndarray,
                         feature_names: list,
                         colour: str) -> None:
    """Draw a Top-15 mean absolute SHAP figure on one axis."""
    importance = pd.Series(
        np.abs(shap_values).mean(axis=0), index=feature_names
    ).sort_values(ascending=False).head(15)

    labels = [to_en(name) for name in importance.index]

    ax.barh(
        range(len(labels)),
        importance.values,
        color=colour,
        edgecolor="white",
        linewidth=0.8,
        zorder=3
    )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(
        labels,
        fontsize=28,
        fontweight="bold"
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Mean |SHAP value|",
        fontsize=28,
        fontweight="bold"
    )

    ax.set_ylabel(
        "PAPA symptom feature",
        fontsize=28,
        fontweight="bold"
    )

    ax.tick_params(
        axis="x",
        labelsize=24
    )

    ax.tick_params(
        axis="y",
        labelsize=24
    )

    ax.set_axisbelow(True)


# ── 18. XAI — SHAP ON THE UNTOUCHED INTERNAL HOLDOUT ─────────────────────
def run_shap(final_models: dict, X_train: pd.DataFrame,
             X_holdout: pd.DataFrame, feature_names: list) -> None:

    if SKIP_SHAP:
        print("  SHAP skipped because GAD_SKIP_SHAP=1")
        return

    shap_by_model = {}
    holdout_by_model = {}

    # ============================================================
    # 1. CALCULATE SHAP VALUES FOR ALL MODELS
    # ============================================================
    for model_name, pipeline in final_models.items():

        print(f"  SHAP — {model_name}")

        clf = pipeline.named_steps["clf"]

        train_values = transformed_matrix(
            pipeline,
            X_train
        )

        holdout_values = transformed_matrix(
            pipeline,
            X_holdout
        )

        background = train_values[
            :min(300, len(train_values))
        ]

        if model_name == "L1 Logistic Regression":

            explainer = shap.LinearExplainer(
                clf,
                background
            )

            shap_values = positive_class_shap_array(
                explainer.shap_values(
                    holdout_values
                )
            )

        else:

            explainer = shap.TreeExplainer(
                clf
            )

            shap_values = positive_class_shap_array(
                explainer.shap_values(
                    holdout_values
                )
            )

        shap_values = np.asarray(
            shap_values,
            dtype=float
        )

        if shap_values.shape != holdout_values.shape:
            raise ValueError(
                f"Unexpected SHAP shape for {model_name}: "
                f"{shap_values.shape} versus "
                f"{holdout_values.shape}"
            )

        shap_by_model[model_name] = shap_values
        holdout_by_model[model_name] = holdout_values

    # ============================================================
    # 2. CHECK REQUIRED MAIN-MANUSCRIPT MODELS
    # ============================================================
    required_main_models = [
        "L1 Logistic Regression",
        "Decision Tree"
    ]

    missing = [
        name
        for name in required_main_models
        if name not in shap_by_model
    ]

    if missing:
        raise KeyError(
            f"Missing models for manuscript SHAP figure: {missing}"
        )

    # ============================================================
    # 3. MAIN MANUSCRIPT SHAP FIGURES
    #
    # a = Logistic Regression beeswarm
    # b = Logistic Regression bar
    # c = Decision Tree beeswarm
    # d = Decision Tree bar
    # ============================================================
   
        # ============================================================
    # 3. MAIN MANUSCRIPT SHAP FIGURES
    #
    # a = Logistic Regression beeswarm
    # b = Logistic Regression bar
    # c = Decision Tree beeswarm
    # d = Decision Tree bar
    # ============================================================
    manuscript_beeswarm_specs = [
        (
            "L1 Logistic Regression",
            RANDOM_STATE + 101,
            "fig12a_shap_beeswarm_lr_l1_primary_reduced",
            [
                "manuscript_figure03a_beeswarm_lr_l1"
            ]
        ),
        (
            "Decision Tree",
            RANDOM_STATE + 202,
            "fig12c_shap_beeswarm_dt_primary_reduced",
            [
                "manuscript_figure03c_beeswarm_dt"
            ]
        ),
    ]

    for model_name, plot_seed, filename, aliases in manuscript_beeswarm_specs:
        _save_shap_beeswarm_figure(
            shap_values=shap_by_model[model_name],
            holdout_values=holdout_by_model[model_name],
            feature_names=feature_names,
            model_name=model_name,
            random_seed=plot_seed,
            filename=filename,
            aliases=aliases
        )

    manuscript_bar_specs = [
        (
            "L1 Logistic Regression",
            "b"
        ),
        (
            "Decision Tree",
            "d"
        ),
    ]

    for model_name, panel_tag in manuscript_bar_specs:
        apply_style()

        fig, ax = plt.subplots(
            figsize=(18, 12)
        )

        _draw_shap_bar_panel(
            ax,
            shap_by_model[model_name],
            feature_names,
            PALETTE[model_name]
        )

        fig.tight_layout()

        model_tag = MODEL_SHORT[model_name].lower()

        filename = (
            f"fig12{panel_tag}_shap_bar_"
            f"{model_tag}_primary_reduced"
        )

        _save(
            fig,
            filename,
            aliases=[
                f"manuscript_figure03{panel_tag}_bar_{model_tag}"
            ]
        )

    # ============================================================
    # 4. SUPPLEMENTARY ENSEMBLE SHAP FIGURES
    #
    # Random Forest beeswarm
    # XGBoost beeswarm
    # Random Forest bar
    # ============================================================
    en_names = [
        to_en(name)
        for name in feature_names
    ]

    supplementary_beeswarm_specs = [
        (
            "Random Forest",
            RANDOM_STATE + 303,
            "figS_shap_beeswarm_rf_primary_reduced"
        ),
        (
            "XGBoost",
            RANDOM_STATE + 404,
            "figS_shap_beeswarm_xgb_primary_reduced"
        ),
    ]

    for model_name, plot_seed, filename in supplementary_beeswarm_specs:
        if model_name in shap_by_model:
            _save_shap_beeswarm_figure(
                shap_values=shap_by_model[model_name],
                holdout_values=holdout_by_model[model_name],
                feature_names=feature_names,
                model_name=model_name,
                random_seed=plot_seed,
                filename=filename,
                aliases=[]
            )

    if "Random Forest" in shap_by_model:
        rf_mean_abs = pd.Series(
            np.abs(shap_by_model["Random Forest"]).mean(axis=0),
            index=en_names
        ).sort_values(
            ascending=False
        ).head(15)

        apply_style()

        fig2, ax2 = plt.subplots(
            figsize=(15, 10)
        )

        ax2.barh(
            range(len(rf_mean_abs)),
            rf_mean_abs.values,
            color=PALETTE["Random Forest"],
            edgecolor="white",
            linewidth=0.8,
            zorder=3
        )

        ax2.set_yticks(
            range(len(rf_mean_abs))
        )

        ax2.set_yticklabels(
            rf_mean_abs.index,
            fontsize=22,
            fontweight="normal"
        )

        ax2.invert_yaxis()

        ax2.set_xlabel(
            "Mean |SHAP value|",
            fontsize=FONT_AXIS,
            fontweight="normal"
        )

        ax2.set_ylabel(
            "PAPA symptom feature",
            fontsize=FONT_AXIS,
            fontweight="normal"
        )

        ax2.tick_params(
            axis="x",
            labelsize=18
        )

        ax2.tick_params(
            axis="y",
            labelsize=22
        )

        ax2.set_axisbelow(True)

        #
        fig2.tight_layout()

        _save(
            fig2,
            "figS_shap_bar_rf_primary_reduced"
        )

    # ============================================================
    # 4. SUPPLEMENTARY ENSEMBLE SHAP FIGURES

    # ============================================================
    # 4. SUPPLEMENTARY ENSEMBLE SHAP FIGURES
    #
    # Random Forest
    # XGBoost
    # ============================================================
    en_names = [
        to_en(name)
        for name in feature_names
    ]

    # ============================================================
    # RANDOM FOREST SHAP BEESWARM
    # Independent styling block
    # ============================================================
    
    if "Random Forest" in shap_by_model:
    
        model_name = "Random Forest"
    
        rf_shap_values = shap_by_model[
            model_name
        ]
    
        rf_holdout_values = holdout_by_model[
            model_name
        ]
    
        apply_style()
    
        fig, ax = plt.subplots(
            figsize=(24, 14)
        )
        #
        _draw_shap_dot_panel(
            ax,
            rf_shap_values,
            rf_holdout_values,
            feature_names,
            RANDOM_STATE + 303,
            ytick_fontsize=44,
            xlabel_fontsize=44,
            ylabel_fontsize=44,
            xtick_fontsize=44,
            point_size=190,
            point_alpha=0.90,
            custom_style=True
        )

        plt.subplots_adjust(
            left=0.38,
            right=0.88,
            bottom=0.12,
            top=0.97
        )

        _add_fixed_shap_colorbar(
            fig,
            cbar_tick_fontsize=44,
            cbar_label_fontsize=44,
            label_weight="normal"
        )

        _save(
            fig,
            "figS_shap_beeswarm_rf_primary_reduced"
        )

        # ============================================================
        # RANDOM FOREST SHAP BAR
        # Independent styling block
        # ============================================================

        rf_mean_abs = pd.Series(
            np.abs(rf_shap_values).mean(axis=0),
            index=en_names
        ).sort_values(
            ascending=False
        ).head(15)

        apply_style()

        fig2, ax2 = plt.subplots(
            figsize=(15, 10)
        )

        ax2.barh(
            range(len(rf_mean_abs)),
            rf_mean_abs.values,
            color=PALETTE["Random Forest"],
            edgecolor="white",
            linewidth=0.8,
            zorder=3
        )

        ax2.set_yticks(
            range(len(rf_mean_abs))
        )

        ax2.set_yticklabels(
            rf_mean_abs.index,
            fontsize=22,
            fontweight="normal"
        )

        ax2.invert_yaxis()

        ax2.set_xlabel(
            "Mean |SHAP value|",
            fontsize=FONT_AXIS,
            fontweight="normal"
        )

        ax2.set_ylabel(
            "PAPA symptom feature",
            fontsize=FONT_AXIS,
            fontweight="normal"
        )

        ax2.tick_params(
            axis="x",
            labelsize=18
        )

        ax2.tick_params(
            axis="y",
            labelsize=22
        )

        ax2.set_axisbelow(True)

        plt.tight_layout()

        _save(
            fig2,
            "figS_shap_bar_rf_primary_reduced"
        )

# ── 19. EXCEL — METRICS ───────────────────────────────────────────────────
_HF  = Font(bold=True, size=12, color="FFFFFF")
_HFI = PatternFill("solid", fgColor="1D3A5F")
_HA  = Alignment(horizontal="center", vertical="center", wrap_text=True)
_BF  = PatternFill("solid", fgColor="D4EDDA")
_EF  = PatternFill("solid", fgColor="F1EFE8")
_WF  = PatternFill("solid", fgColor="FFFFFF")


def _hdr(ws, row, col, val):
    c = ws.cell(row=row, column=col, value=val)
    c.font      = _HF
    c.fill      = _HFI
    c.alignment = _HA


def _dat(ws, row, col, val, even=False, best=False, fmt=None):
    c           = ws.cell(row=row, column=col, value=val)
    c.font      = Font(bold=best, size=11)
    c.fill      = _BF if best else (_EF if even else _WF)
    c.alignment = Alignment(horizontal="center", vertical="center")
    if fmt:
        c.number_format = fmt


def save_metrics_excel(results: dict, path: str) -> None:
    """Save primary reduced-feature nested-CV metrics."""
    wb      = Workbook()
    metrics = ["AUC-ROC", "AP", "F1", "Precision", "Sensitivity",
               "Specificity", "NPV", "Accuracy", "Balanced Accuracy",
               "MCC", "Brier", "TN", "FP", "FN", "TP"]
    header  = ["Model"] + metrics
    ws      = wb.create_sheet("GAD")

    ws.merge_cells(f"A1:{get_column_letter(len(header))}1")
    c           = ws["A1"]
    c.value     = (f"GAD Prediction — Primary Reduced Feature Set "
                   f"(Nested {OUTER_FOLDS}-Fold CV on Training Only)")
    c.font      = Font(bold=True, size=14, color="FFFFFF")
    c.fill      = PatternFill("solid", fgColor="0A2540")
    c.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 26

    for j, h in enumerate(header, 1):
        _hdr(ws, 2, j, h)
    ws.row_dimensions[2].height = 22

    best_model = max(results, key=lambda k: results[k]["AUC-ROC"])

    for ri, (name, mdata) in enumerate(results.items(), 3):
        is_best = (name == best_model)
        c       = ws.cell(row=ri, column=1, value=name)
        c.font  = Font(bold=True, size=11)
        c.fill  = _BF if is_best else (_EF if ri % 2 == 0 else _WF)
        c.alignment = Alignment(horizontal="left", vertical="center")

        for j, metric in enumerate(metrics, 2):
            _dat(ws, ri, j, mdata[metric],
                 even=(ri % 2 == 0), best=is_best, fmt="0.0000")
        ws.row_dimensions[ri].height = 20

    ws.column_dimensions["A"].width = 26
    for j in range(2, len(header) + 1):
        ws.column_dimensions[get_column_letter(j)].width = 14
    ws.freeze_panes = "A3"

    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]
    wb.save(path)
    print(f"  Saved : {path}")


# ── 20. EXCEL — ACTUAL VS PREDICTED ──────────────────────────────────────
def save_actual_pred_excel(df: pd.DataFrame, holdout_metrics: dict,
                           path: str) -> None:
    """Save traceable original-sample predictions for the internal holdout."""
    wb = Workbook()

    # Summary sheet
    ws0 = wb.create_sheet("Summary", 0)
    summary_metrics = ["AUC-ROC", "AP", "F1", "Sensitivity",
                       "Specificity", "MCC", "Accuracy"]
    summary_columns = ["Model", "Total Samples"] + summary_metrics
    ws0.merge_cells(
        f"A1:{get_column_letter(len(summary_columns))}1"
    )
    c           = ws0["A1"]
    c.value     = ("GAD Prediction — Internal Holdout Test (20%) | "
                   "Primary Reduced Feature Set")
    c.font      = Font(bold=True, size=14, color="FFFFFF")
    c.fill      = PatternFill("solid", fgColor="0A2540")
    c.alignment = Alignment(horizontal="center", vertical="center")
    ws0.row_dimensions[1].height = 26

    for j, h in enumerate(summary_columns, 1):
        _hdr(ws0, 2, j, h)
    ws0.row_dimensions[2].height = 20

    for ri, (name, grp) in enumerate(df.groupby("Model"), 3):
        total   = len(grp)
        values = [name, total] + [holdout_metrics[name][metric]
                                  for metric in summary_metrics]
        for j, v in enumerate(values, 1):
            _dat(ws0, ri, j, v, even=(ri % 2 == 0))
        ws0.row_dimensions[ri].height = 18

    for j, w in enumerate([26, 16] + [14] * len(summary_metrics), 1):
        ws0.column_dimensions[get_column_letter(j)].width = w
    ws0.freeze_panes = "A3"

    # Per-model detail sheets
    cols       = ["Original_Row_Index", "Subject_ID", "Actual", "Predicted",
                  "Probability_GAD", "Decision_Threshold", "Correct"]
    wrong_fill = PatternFill("solid", fgColor="FDECEA")

    for name, grp in df.groupby("Model"):
        ws = wb.create_sheet(title=f"GAD_{MODEL_SHORT[name]}"[:31])

        ws.merge_cells(f"A1:{get_column_letter(len(cols))}1")
        c           = ws["A1"]
        c.value     = f"GAD  —  {name}  (Holdout Validation Set)"
        c.font      = Font(bold=True, size=13, color="FFFFFF")
        c.fill      = PatternFill("solid", fgColor="1D3A5F")
        c.alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 22

        for j, h in enumerate(cols, 1):
            _hdr(ws, 2, j, h)
        ws.row_dimensions[2].height = 20

        for ri, (_, row) in enumerate(grp.iterrows(), 3):
            for j, col in enumerate(cols, 1):
                c           = ws.cell(row=ri, column=j, value=row[col])
                c.font      = Font(size=11)
                c.alignment = Alignment(horizontal="center",
                                        vertical="center")
                c.fill      = (wrong_fill if not row["Correct"]
                               else (_EF if ri % 2 == 0 else _WF))
                if col in {"Probability_GAD", "Decision_Threshold"}:
                    c.number_format = "0.0000"
            ws.row_dimensions[ri].height = 18

        for j in range(1, len(cols) + 1):
            ws.column_dimensions[get_column_letter(j)].width = 20
        ws.freeze_panes = "A3"

    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]
    wb.save(path)
    print(f"  Saved : {path}")


def write_dataframe_sheet(wb: Workbook, name: str,
                          frame: pd.DataFrame) -> None:
    ws = wb.create_sheet(name[:31])
    for col_idx, column in enumerate(frame.columns, 1):
        _hdr(ws, 1, col_idx, str(column))
    for row_idx, (_, row) in enumerate(frame.iterrows(), 2):
        for col_idx, value in enumerate(row, 1):
            if pd.isna(value):
                value = None
            elif isinstance(value, (np.integer, np.floating)):
                value = value.item()
            ws.cell(row=row_idx, column=col_idx, value=value)
    for col_idx, column in enumerate(frame.columns, 1):
        width = min(60, max(12, len(str(column)) + 2))
        ws.column_dimensions[get_column_letter(col_idx)].width = width
    ws.freeze_panes = "A2"


def save_frames_excel(path: str, frames: dict) -> None:
    """Save named audit frames to one styled workbook."""
    wb = Workbook()
    del wb["Sheet"]
    for sheet_name, frame in frames.items():
        write_dataframe_sheet(wb, sheet_name, frame.reset_index(drop=True))
    wb.save(path)
    print(f"  Saved : {path}")


def default_metrics_to_dataframe(probas: dict) -> pd.DataFrame:
    return metrics_to_dataframe({
        model: data["default_metrics"] for model, data in probas.items()
    })


def unweighted_metrics_to_dataframe(probas: dict,
                                    default_threshold: bool = False
                                    ) -> pd.DataFrame:
    key = ("default_unweighted_metrics" if default_threshold else
           "unweighted_metrics")
    return metrics_to_dataframe({model: data[key]
                                 for model, data in probas.items()})


def select_primary_model(results: dict, probas: dict) -> tuple:
    """Apply the prespecified clinical hierarchy without consulting holdout."""
    interpretability_rank = {
        "L1 Logistic Regression": 0, "Decision Tree": 1,
        "Random Forest": 2, "XGBoost": 3,
    }
    rows = []
    for model, metrics in results.items():
        eligible = (metrics["Specificity"] >= SPECIFICITY_FLOOR and
                    probas[model]["constraint_status"] == "constraint_met")
        rows.append({
            "Model": model,
            "Eligible_Specificity_At_Least_0.80": bool(eligible),
            "Nested_CV_Sensitivity": metrics["Sensitivity"],
            "Nested_CV_Specificity": metrics["Specificity"],
            "Nested_CV_AP": metrics["AP"],
            "Nested_CV_MCC": metrics["MCC"],
            "Interpretability_Rank": interpretability_rank[model],
        })
    eligible_models = [row for row in rows
                       if row["Eligible_Specificity_At_Least_0.80"]]
    if eligible_models:
        best_row = max(
            eligible_models,
            key=lambda row: (row["Nested_CV_Sensitivity"],
                             row["Nested_CV_AP"], row["Nested_CV_MCC"],
                             -row["Interpretability_Rank"]))
        selected = best_row["Model"]
        status = "selected_under_prespecified_constraint"
    else:
        selected = None
        status = "no_model_met_prespecified_specificity_constraint"
    for row in rows:
        row["Selection_Status"] = status
        row["Selected_Model"] = selected or "NONE"
    return selected, pd.DataFrame(rows)


def build_feature_audit(df: pd.DataFrame,
                        primary_features: list) -> pd.DataFrame:
    primary_set = set(primary_features)
    rows = []
    for column in df.columns:
        is_onset = "Onset" in str(column)
        is_direct = column in DIRECT_DIAGNOSTIC_FEATURES
        is_suspicious = column in SUSPICIOUS_HIGH_LEVEL_FEATURES
        if column == "GAD":
            role = "Outcome"
        elif column in {"Subject", "SAD", "Sampling Weight"}:
            role = "Administrative_or_nonpredictor"
        elif is_onset:
            role = "Excluded_onset"
        elif is_direct:
            role = "Excluded_direct_diagnostic_definition"
        elif is_suspicious:
            role = "Retained_pending_clinical_adjudication"
        else:
            role = "Retained_candidate_requires_clinical_role_confirmation"
        missing_count = int(df[column].isna().sum())
        rows.append({
            "Original_Column": column,
            "Audit_Role": role,
            "Is_Onset": is_onset,
            "Is_Direct_Diagnostic_Definition": is_direct,
            "Flagged_High_Level_Clinical_Review": is_suspicious,
            "Included_in_Primary_54": column in primary_set,
            "Missing_Count": missing_count,
            "Missing_Percent": round(100 * missing_count / len(df), 3),
            "Unique_Nonmissing_Values": int(df[column].nunique(dropna=True)),
            "Clinician_Assigned_Role": "",
            "Clinician_Adjudication_Required": bool(
                column in primary_set and not is_direct and not is_onset),
            "Automatic_Exclusion_Reason": (
                "onset" if is_onset else
                "direct_diagnostic_definition" if is_direct else "none"),
        })
    return pd.DataFrame(rows)


def run_acceptance_checks(df: pd.DataFrame, feature_sets: dict,
                          primary_specs: dict, train_idx: np.ndarray,
                          holdout_idx: np.ndarray, primary_probas: dict,
                          threshold_audit: pd.DataFrame,
                          holdout_predictions: pd.DataFrame) -> pd.DataFrame:
    """Executable checks corresponding to the prespecified acceptance criteria."""
    checks = []

    def record(name, condition, evidence):
        checks.append({"Check": name, "Passed": bool(condition),
                       "Evidence": str(evidence)})
        if not condition:
            raise AssertionError(f"Acceptance check failed: {name}: {evidence}")

    primary_columns = list(feature_sets["Primary_reduced"].columns)
    onset_in_primary = [c for c in primary_columns if "Onset" in c]
    direct_in_primary = sorted(set(primary_columns) &
                               set(DIRECT_DIAGNOSTIC_FEATURES))
    record("Primary matrix has exactly 54 predictors",
           len(primary_columns) == 54, len(primary_columns))
    record("No onset predictor in primary matrix",
           not onset_in_primary, onset_in_primary)
    record("No direct diagnostic feature in primary matrix",
           not direct_in_primary, direct_in_primary)
    record("Sampling Weight is analysis metadata, not a predictor",
           "Sampling Weight" not in primary_columns,
           "Sampling Weight excluded from X and retained for weighting")
    primary_smote_steps = {
        model: [name for name, _ in spec["pipeline"].steps if name == "smote"]
        for model, spec in primary_specs.items()
    }
    record("No SMOTE object in primary pipelines",
           all(not value for value in primary_smote_steps.values()),
           primary_smote_steps)
    primary_class_weights = {
        model: spec["pipeline"].named_steps["clf"].get_params().get(
            "class_weight")
        for model, spec in primary_specs.items()
    }
    record("No class balancing in design-weight primary pipelines",
           all(value is None for value in primary_class_weights.values()),
           primary_class_weights)
    record("Development and holdout indices are disjoint",
           not set(train_idx).intersection(holdout_idx),
           f"train={len(train_idx)}, holdout={len(holdout_idx)}")
    train_idx_2, holdout_idx_2 = make_train_holdout_indices(
        df["GAD"].astype(int).to_numpy())
    record("Top-level split is deterministic under the locked seed",
           np.array_equal(train_idx, train_idx_2) and
           np.array_equal(holdout_idx, holdout_idx_2), RANDOM_STATE)

    primary_threshold_rows = threshold_audit[
        (threshold_audit["Resampling_Mode"] == "design_weight") &
        (threshold_audit["Feature_Set"] == "Primary_reduced") &
        (threshold_audit["Outer_Fold"].astype(str) !=
         "final_training_to_holdout")]
    leakage_free = True
    for _, row in primary_threshold_rows.iterrows():
        fit = set(json.loads(row["Outer_Training_Indices"]))
        val = set(json.loads(row["Outer_Validation_Indices"]))
        leakage_free &= not fit.intersection(val)
        leakage_free &= fit.issubset(set(train_idx))
        leakage_free &= val.issubset(set(train_idx))
        leakage_free &= not val.intersection(set(holdout_idx))
    record("Every outer-fold threshold is training-derived and leakage-free",
           leakage_free and len(primary_threshold_rows) ==
           OUTER_FOLDS * len(primary_specs),
           f"rows={len(primary_threshold_rows)}")

    expected_holdout_rows = len(holdout_idx) * len(primary_specs)
    holdout_once = (len(holdout_predictions) == expected_holdout_rows and
                    all(group["Original_Row_Index"].nunique() == len(holdout_idx)
                        for _, group in holdout_predictions.groupby("Model")))
    record("Each model predicts all 184 holdout children exactly once",
           holdout_once and len(holdout_idx) == 184,
           f"rows={len(holdout_predictions)}, expected={expected_holdout_rows}")

    confusion_consistent = True
    for model, data in primary_probas.items():
        recomputed = calculate_metrics(data["y_true"], data["y_proba"],
                                       data["sample_thresholds"],
                                       sample_weight=data.get("sample_weight"))
        stored = data["weighted_metrics"]
        confusion_consistent &= all(
            np.isclose(float(recomputed[key]), float(stored[key]),
                       atol=1e-12)
            for key in ["Sensitivity", "Specificity", "Accuracy", "MCC",
                        "TN", "FP", "FN", "TP"])
    record("Confusion-derived metrics independently reconcile",
           confusion_consistent, "TN/FP/FN/TP matched")
    record("Smoke outputs are explicitly non-manuscript when enabled",
           (not QUICK_TEST) or BOOTSTRAP_ITERATIONS < 2000,
           "SMOKE_TEST_NOT_FOR_MANUSCRIPT" if QUICK_TEST else "FULL_ANALYSIS")
    return pd.DataFrame(checks)


def save_revision_audit_excel(all_cv_results: dict,
                              nested_tuning: pd.DataFrame,
                              final_tuning: pd.DataFrame,
                              feature_sets: dict, path: str) -> None:
    """Save feature-set ablation and tuning provenance for Reviewer 1."""
    metric_rows = []
    for feature_set_name, model_results in all_cv_results.items():
        for model_name, metrics in model_results.items():
            metric_rows.append({
                "Feature_Set": feature_set_name,
                "Model": model_name,
                **metrics,
            })
    feature_rows = []
    all_features = list(feature_sets["Full_features_sensitivity"].columns)
    primary_features = set(feature_sets["Primary_reduced"].columns)
    for feature in all_features:
        feature_rows.append({
            "Original_Feature": feature,
            "Display_Label": to_en(feature),
            "Direct_Diagnostic_Definition":
                feature in DIRECT_DIAGNOSTIC_FEATURES,
            "Included_in_Primary_Analysis": feature in primary_features,
        })

    wb = Workbook()
    del wb["Sheet"]
    write_dataframe_sheet(wb, "Feature_Set_Comparison",
                          pd.DataFrame(metric_rows))
    write_dataframe_sheet(wb, "Nested_CV_Tuning", nested_tuning)
    write_dataframe_sheet(wb, "Final_Training_Tuning", final_tuning)
    write_dataframe_sheet(wb, "Feature_Audit", pd.DataFrame(feature_rows))
    wb.save(path)
    print(f"  Saved : {path}")


def resolve_code_provenance() -> dict:
    """Return code identity without assuming execution from a .py file.

    Kaggle/Jupyter notebook cells do not define ``__file__``.  An explicit
    GAD_CODE_PATH may be supplied when the exact exported script/notebook file
    is available; otherwise the manifest records the interactive runtime
    honestly instead of failing after the analysis has completed.
    """
    candidates = [
        ("GAD_CODE_PATH", os.environ.get("GAD_CODE_PATH")),
        ("__file__", globals().get("__file__")),
    ]
    for source, candidate in candidates:
        if candidate:
            resolved = os.path.abspath(os.fspath(candidate))
            if os.path.isfile(resolved):
                return {
                    "code_path": resolved,
                    "code_sha256": file_sha256(resolved),
                    "code_provenance": f"file:{source}",
                }
    return {
        "code_path": None,
        "code_sha256": None,
        "code_provenance": (
            "interactive_notebook_cell; set GAD_CODE_PATH to hash an "
            "exported source file"
        ),
    }


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_run_manifest(path: str, data_path: str, train_idx: np.ndarray,
                      holdout_idx: np.ndarray, feature_sets: dict,
                      sampling_weight: np.ndarray,
                      regime_results: dict,
                      holdout_results: dict, tuning_audit: pd.DataFrame,
                      threshold_audit: pd.DataFrame,
                      bootstrap_seeds: dict,
                      selection_audit: pd.DataFrame,
                      acceptance_checks: pd.DataFrame) -> None:
    code_provenance = resolve_code_provenance()
    manifest = {
        "analysis_status": ("SMOKE_TEST_NOT_FOR_MANUSCRIPT" if QUICK_TEST
                            else "FULL_ANALYSIS"),
        "data_path": os.path.abspath(data_path),
        "data_sha256": file_sha256(data_path),
        **code_provenance,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "primary_scoring": PRIMARY_SCORING,
        "specificity_floor": SPECIFICITY_FLOOR,
        "threshold_objective": (
            "maximise_sensitivity_subject_to_specificity_at_least_0.80"),
        "threshold_grid_size": THRESHOLD_GRID_SIZE,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seeds": bootstrap_seeds,
        "random_search_iterations": RANDOM_SEARCH_ITERATIONS,
        "training_indices": [int(i) for i in train_idx],
        "holdout_indices": [int(i) for i in holdout_idx],
        "sampling_weight_summary": {
            "minimum": float(np.min(sampling_weight)),
            "maximum": float(np.max(sampling_weight)),
            "sum": float(np.sum(sampling_weight)),
            "kish_effective_n": float(
                np.sum(sampling_weight) ** 2 /
                np.square(sampling_weight).sum()),
        },
        "direct_diagnostic_features": DIRECT_DIAGNOSTIC_FEATURES,
        "feature_sets": {name: list(frame.columns)
                         for name, frame in feature_sets.items()},
        "nested_cv_results_by_regime": regime_results,
        "internal_holdout_results_by_regime": holdout_results,
        "tuning_audit": tuning_audit.to_dict(orient="records"),
        "threshold_audit": threshold_audit.to_dict(orient="records"),
        "model_selection_audit": selection_audit.to_dict(orient="records"),
        "acceptance_checks": acceptance_checks.to_dict(orient="records"),
        "xgboost_available": XGBOOST_AVAILABLE,
        "require_xgboost": REQUIRE_XGBOOST,
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "imbalanced_learn": imblearn.__version__,
            "shap": shap.__version__,
            "matplotlib": matplotlib.__version__,
            "openpyxl": openpyxl.__version__,
            "xgboost": (xgboost.__version__ if XGBOOST_AVAILABLE else None),
        },
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2,
                  default=_json_default)
    print(f"  Saved : {path}")


# ── 21. MAIN ──────────────────────────────────────────────────────────────
def main():
    div = "=" * 62
    print(div)
    print("Reviewer-Aligned GAD Classification Analysis")
    print("Design-weight primary | Class-weight/native/SMOTE sensitivity")
    print(div)

    if REQUIRE_XGBOOST and not XGBOOST_AVAILABLE:
        raise ImportError(
            "XGBoost is required for the planned full analysis but is not "
            "installed. Install it with `pip install xgboost`, or set "
            "GAD_REQUIRE_XGBOOST=0 only for a clearly labelled smoke test.")

    print("\n[1/11] Loading data and locking the train/holdout split ...")
    df, X, y_gad, sampling_weight = load_data(TRAIN_PATH)
    feature_sets = build_feature_sets(X)
    train_idx, holdout_idx = make_train_holdout_indices(y_gad)
    if len(df) == 917 and (len(train_idx), len(holdout_idx)) != (733, 184):
        raise AssertionError("Expected a 733/184 split for n=917.")
    print(f"Training/model-development samples: {len(train_idx)}")
    print(f"Untouched internal holdout samples: {len(holdout_idx)}")
    print(f"Primary predictors after diagnostic exclusions: "
          f"{feature_sets['Primary_reduced'].shape[1]}")

    def partition_summary(label, indices):
        local_y = y_gad[indices]
        local_w = sampling_weight[indices]
        return {
            "Dataset": label,
            "Samples": len(indices),
            "GAD_negative": int(np.sum(local_y == 0)),
            "GAD_positive": int(np.sum(local_y == 1)),
            "Unweighted_Positive_Rate": round(float(np.mean(local_y)), 4),
            "Design_Weighted_Positive_Rate": round(
                float(np.average(local_y, weights=local_w)), 4),
            "Sampling_Weight_Sum": round(float(local_w.sum()), 4),
            "Kish_Effective_N": round(float(
                local_w.sum() ** 2 / np.square(local_w).sum()), 2),
        }

    dataset_summary = pd.DataFrame([
        partition_summary("All original observations", np.arange(len(df))),
        partition_summary("Model-development partition", train_idx),
        partition_summary("Untouched internal holdout", holdout_idx),
    ])
    show_dataframe(dataset_summary, "Dataset and class-distribution summary")

    feature_set_summary = pd.DataFrame([
        {
            "Feature_Set": feature_set_name,
            "Number_of_predictors": feature_frame.shape[1],
            "Purpose": {
                "Primary_reduced": "Primary analysis without direct diagnostic-definition features",
                "Full_features_sensitivity": "Sensitivity analysis with all candidate predictors",
                "Diagnostic_only_sensitivity": "Circularity/diagnostic-definition sensitivity analysis",
            }[feature_set_name],
        }
        for feature_set_name, feature_frame in feature_sets.items()
    ])
    show_dataframe(feature_set_summary, "Prespecified feature-set summary")

    design_specs = build_model_specs("design_weight")
    class_weight_specs = build_model_specs("class_weight")
    smote_specs = build_model_specs("smote")
    native_specs = build_model_specs("native")
    core_model_names = ["L1 Logistic Regression", "Decision Tree",
                        "Random Forest"]
    primary_X = feature_sets["Primary_reduced"]
    X_train = primary_X.iloc[train_idx].copy()
    X_holdout = primary_X.iloc[holdout_idx].copy()
    y_train = y_gad[train_idx]
    y_holdout = y_gad[holdout_idx]
    weight_train = sampling_weight[train_idx]
    weight_holdout = sampling_weight[holdout_idx]
    tuning_frames, threshold_frames, stability_frames = [], [], []

    print("\n[2/11] Primary sampling-design-weighted nested CV ...")
    (design_results, design_probas, design_tuning,
     design_thresholds, design_stability) = run_nested_cv(
        design_specs, X_train, y_train, "Primary_reduced",
        "design_weight", original_indices=train_idx,
        sample_weight=weight_train, evaluation_weight=weight_train)
    tuning_frames.append(design_tuning)
    threshold_frames.append(design_thresholds)
    stability_frames.append(design_stability)
    show_dataframe(metrics_to_dataframe(design_results),
                   "PRIMARY design-weighted nested-CV threshold metrics")
    show_dataframe(unweighted_metrics_to_dataframe(design_probas),
                   "Unweighted reference metrics for the same OOF predictions")

    print("\n[3/11] Preserved class-weight circularity analyses ...")
    class_weight_results, class_weight_probas = {}, {}
    for feature_set_name, feature_frame in feature_sets.items():
        # XGBoost is the strong ensemble benchmark only for the primary 54-set.
        specs_for_set = (class_weight_specs
                         if feature_set_name == "Primary_reduced"
                         else {name: class_weight_specs[name]
                               for name in core_model_names})
        X_train_set = feature_frame.iloc[train_idx].copy()
        (results_set, probas_set, tuning_set,
         thresholds_set, stability_set) = run_nested_cv(
            specs_for_set, X_train_set, y_train, feature_set_name,
            "class_weight", original_indices=train_idx,
            evaluation_weight=weight_train
        )
        class_weight_results[feature_set_name] = results_set
        class_weight_probas[feature_set_name] = probas_set
        tuning_frames.append(tuning_set)
        threshold_frames.append(thresholds_set)
        stability_frames.append(stability_set)
        show_dataframe(
            metrics_to_dataframe(results_set),
            f"Class-weight nested-CV threshold metrics — {feature_set_name}",
        )

    class_primary_results = class_weight_results["Primary_reduced"]
    class_primary_probas = class_weight_probas["Primary_reduced"]

    print("\n[4/11] Native and SMOTE sensitivity analyses ...")
    (smote_results, smote_probas, smote_tuning,
     smote_thresholds, smote_stability) = run_nested_cv(
        smote_specs, X_train, y_train, "Primary_reduced", "smote",
        original_indices=train_idx, evaluation_weight=weight_train)
    tuning_frames.append(smote_tuning)
    threshold_frames.append(smote_thresholds)
    stability_frames.append(smote_stability)
    show_dataframe(metrics_to_dataframe(smote_results),
                   "SMOTE sensitivity-analysis threshold metrics")
    show_dataframe(default_metrics_to_dataframe(smote_probas),
                   "SMOTE sensitivity-analysis metrics at threshold 0.50")

    native_results, native_probas = {}, {}
    if RUN_NATIVE_SENSITIVITY:
        (native_results, native_probas, native_tuning,
         native_thresholds, native_stability) = run_nested_cv(
            native_specs, X_train, y_train, "Primary_reduced", "native",
            original_indices=train_idx, evaluation_weight=weight_train)
        tuning_frames.append(native_tuning)
        threshold_frames.append(native_thresholds)
        stability_frames.append(native_stability)
        show_dataframe(metrics_to_dataframe(native_results),
                       "Native-prevalence sensitivity-analysis metrics")

    primary_results = design_results
    primary_probas = design_probas
    selected_model, selection_audit = select_primary_model(
        primary_results, primary_probas)
    show_dataframe(selection_audit,
                   "Prespecified training-only model-selection audit")

    print("\n[5/11] One-time internal holdout evaluations (not model selection) ...")
    (df_predictions, final_models, holdout_metrics,
     final_tuning, final_thresholds) = fit_and_evaluate_holdout(
        design_specs, X_train, y_train, X_holdout, y_holdout,
        df.iloc[holdout_idx]["Subject"],
        "design_weight", sample_weight=weight_train,
        threshold_weight=weight_train, holdout_weight=weight_holdout,
    )
    (class_predictions, _, class_holdout_metrics,
     class_final_tuning, class_final_thresholds) = fit_and_evaluate_holdout(
        class_weight_specs, X_train, y_train, X_holdout, y_holdout,
        df.iloc[holdout_idx]["Subject"],
        "class_weight", threshold_weight=weight_train,
        holdout_weight=weight_holdout,
    )
    show_dataframe(
        metrics_to_dataframe(holdout_metrics),
        "PRIMARY design-weighted internal holdout metrics — Primary_reduced",
    )
    show_dataframe(final_tuning, "Final training-only hyperparameter selection")

    print("\n[6/11] Calibration, paired circularity deltas, and confidence intervals ...")
    (calibration_summary, reliability_bins, bootstrap_ci,
     calibration_seeds) = build_calibration_and_ci_frames(primary_probas)
    circularity_differences, circularity_seeds = paired_feature_set_differences(
        class_weight_probas)
    bootstrap_seeds = {
        "calibration_and_metric_CI": calibration_seeds,
        "paired_circularity_differences": circularity_seeds,
    }
    show_dataframe(calibration_summary, "Calibration summary")
    show_dataframe(bootstrap_ci, "Stratified-bootstrap 95% confidence intervals")
    show_dataframe(circularity_differences,
                   "Paired diagnostic-circularity performance differences")

    print("\n[7/11] Missingness and imputation negative-control audit ...")
    missingness_comparison = pd.DataFrame([{
        "Analysis": "Audit disabled by environment",
        "Status": "NOT_RUN",
    }])
    if RUN_MISSINGNESS_AUDIT:
        zero_l1_spec = {
            "L1 Logistic Regression": build_model_specs(
                "design_weight", imputation_strategy="zero"
            )["L1 Logistic Regression"]
        }
        (zero_results, zero_probas, zero_tuning,
         zero_thresholds, zero_stability) = run_nested_cv(
            zero_l1_spec, X_train, y_train,
            "Primary_reduced_zero_imputation",
            "design_weight_zero_imputation", original_indices=train_idx,
            sample_weight=weight_train, evaluation_weight=weight_train)
        missing_only_X = X_train.isna().astype(float)
        missing_l1_spec = {
            "L1 Logistic Regression": design_specs[
                "L1 Logistic Regression"]
        }
        (missing_results, missing_probas, missing_tuning,
         missing_thresholds, missing_stability) = run_nested_cv(
            missing_l1_spec, missing_only_X, y_train,
            "Missingness_indicators_only", "design_weight_missingness_only",
            original_indices=train_idx, sample_weight=weight_train,
            evaluation_weight=weight_train)
        tuning_frames.extend([zero_tuning, missing_tuning])
        threshold_frames.extend([zero_thresholds, missing_thresholds])
        stability_frames.extend([zero_stability, missing_stability])
        missingness_rows = []
        for analysis_name, result_map in [
                ("Median imputation primary", {
                    "L1 Logistic Regression": design_results[
                        "L1 Logistic Regression"]}),
                ("Zero imputation sensitivity", zero_results),
                ("Missingness-only negative control", missing_results)]:
            for model_name, metrics in result_map.items():
                missingness_rows.append({
                    "Analysis": analysis_name, "Model": model_name,
                    **metrics})
        missingness_comparison = pd.DataFrame(missingness_rows)
        show_dataframe(missingness_comparison,
                       "Missingness/imputation audit metrics")

    print("\n[8/11] Preserved figures from primary training-only nested CV ...")
    plot_class_distribution(y_train)
    plot_metric_comparison(primary_results)
    plot_roc_curves(primary_probas)
    plot_pr_curves(primary_probas)
    plot_confusion_matrices(primary_probas)
    plot_radar(primary_results)
    regime_plot_results = {
        "Design weight (primary)": primary_results,
        "Class weight (sensitivity)": class_primary_results,
        "SMOTE (sensitivity)": smote_results,
    }
    if native_results:
        regime_plot_results["Native (sensitivity)"] = native_results
    plot_resampling_comparison(regime_plot_results)
    plot_threshold_tradeoff(primary_probas)
    plot_reliability(primary_probas)

    print("\n[9/11] Explainability for final design-weight primary models ...")
    primary_features = list(primary_X.columns)
    plot_feature_importance(final_models, primary_features)
    plot_ensemble_feature_importance_supplement(
        final_models, primary_features)
    plot_decision_tree_rules(final_models, primary_features)
    run_shap(final_models, X_train, X_holdout, primary_features)

    print("\n[10/11] Acceptance checks and auditable Excel outputs ...")
    nested_tuning = pd.concat(tuning_frames, ignore_index=True)
    threshold_audit = pd.concat(
        [*threshold_frames, final_thresholds, class_final_thresholds],
        ignore_index=True)
    raw_stability = pd.concat(stability_frames, ignore_index=True)
    stability_summary = summarize_feature_stability(raw_stability)
    feature_audit = build_feature_audit(df, primary_features)
    acceptance_checks = run_acceptance_checks(
        df, feature_sets, design_specs, train_idx, holdout_idx,
        primary_probas, threshold_audit, df_predictions)
    show_dataframe(acceptance_checks, "Executable acceptance checks")

    primary_sensitivity_rows = []
    for feature_set_name, model_results in class_weight_results.items():
        for model_name, metrics in model_results.items():
            primary_sensitivity_rows.append({
                "Feature_Set": feature_set_name, "Model": model_name,
                "Result_Role": ("PRIMARY" if feature_set_name ==
                                "Primary_reduced" else
                                "CIRCULARITY_SENSITIVITY_ONLY"),
                **metrics})
    primary_sensitivity_frame = pd.DataFrame(primary_sensitivity_rows)

    clinical_impact = build_clinical_impact_table(primary_results)

    design_frames = {
        "Nested_CV_Weighted": metrics_to_dataframe(primary_results),
        "Nested_CV_Unweighted_Reference":
            unweighted_metrics_to_dataframe(primary_probas),
        "Nested_CV_Default_0.50": default_metrics_to_dataframe(primary_probas),
        "Holdout_Weighted": metrics_to_dataframe(holdout_metrics),
        "Clinical_Impact_per_1000": clinical_impact,
        "Model_Selection": selection_audit,
        "Final_Training_Tuning": final_tuning,
        "Dataset_Summary": dataset_summary,
        "Feature_Audit": feature_audit,
        "Acceptance_Checks": acceptance_checks,
    }
    save_frames_excel(
        os.path.join(OUTPUT_DIR, "GAD_primary_design_weight_metrics.xlsx"),
        design_frames)
    # Backward-compatible result filename from the first analysis.
    save_frames_excel(
        os.path.join(OUTPUT_DIR, "GAD_primary_nested_cv_metrics.xlsx"),
        design_frames)

    save_frames_excel(
        os.path.join(OUTPUT_DIR, "GAD_primary_class_weight_metrics.xlsx"),
        {
            "Nested_CV_Weighted": metrics_to_dataframe(class_primary_results),
            "Nested_CV_Unweighted":
                unweighted_metrics_to_dataframe(class_primary_probas),
            "Nested_CV_Default_0.50":
                default_metrics_to_dataframe(class_primary_probas),
            "Holdout_Weighted": metrics_to_dataframe(class_holdout_metrics),
            "Feature_Set_Ablations": primary_sensitivity_frame,
            "Final_Training_Tuning": class_final_tuning,
            "Feature_Audit": feature_audit,
        })
    save_frames_excel(
        os.path.join(OUTPUT_DIR, "GAD_smote_sensitivity_analysis.xlsx"),
        {
            "Nested_CV_Threshold_Tuned": metrics_to_dataframe(smote_results),
            "Nested_CV_Default_0.50": default_metrics_to_dataframe(smote_probas),
            "Nested_CV_Tuning": smote_tuning,
            "Interpretation": pd.DataFrame([{
                "Analysis_Role": "SECONDARY_SENSITIVITY_ANALYSIS_ONLY",
                "Primary_Analysis": False,
                "Caution": ("SMOTE may alter correlation and calibration; "
                            "results are not the main clinical claim."),
            }]),
        })
    if native_results:
        save_frames_excel(
            os.path.join(OUTPUT_DIR, "GAD_native_sensitivity_analysis.xlsx"),
            {
                "Nested_CV_Weighted": metrics_to_dataframe(native_results),
                "Nested_CV_Unweighted":
                    unweighted_metrics_to_dataframe(native_probas),
                "Interpretation": pd.DataFrame([{
                    "Analysis_Role": "SECONDARY_SENSITIVITY_ANALYSIS_ONLY",
                    "Primary_Analysis": False,
                    "Caution": "No synthetic resampling and no class balancing.",
                }]),
            })
    save_frames_excel(
        os.path.join(OUTPUT_DIR, "GAD_threshold_audit.xlsx"),
        {"Threshold_Audit": threshold_audit,
         "Hyperparameter_Audit": nested_tuning,
         "Model_Selection": selection_audit})
    save_frames_excel(
        os.path.join(OUTPUT_DIR, "GAD_calibration_and_CI.xlsx"),
        {"Calibration_Summary": calibration_summary,
         "Reliability_Bins": reliability_bins,
         "Bootstrap_95CI": bootstrap_ci})
    save_frames_excel(
        os.path.join(OUTPUT_DIR, "GAD_circularity_paired_differences.xlsx"),
        {"Paired_Bootstrap_Deltas": circularity_differences,
         "Feature_Set_Ablations": primary_sensitivity_frame})
    save_frames_excel(
        os.path.join(OUTPUT_DIR, "GAD_feature_stability.xlsx"),
        {"Stability_Summary": stability_summary,
         "Fold_Level_Importance": raw_stability})
    save_frames_excel(
        os.path.join(OUTPUT_DIR, "GAD_missingness_sensitivity.xlsx"),
        {"Missingness_And_Imputation": missingness_comparison})
    save_actual_pred_excel(
        df_predictions, holdout_metrics,
        os.path.join(OUTPUT_DIR, "GAD_internal_holdout_predictions.xlsx")
    )
    save_actual_pred_excel(
        class_predictions, class_holdout_metrics,
        os.path.join(
            OUTPUT_DIR,
            "GAD_internal_holdout_predictions_class_weight_sensitivity.xlsx")
    )
    save_revision_audit_excel(
        class_weight_results, nested_tuning, final_tuning, feature_sets,
        os.path.join(OUTPUT_DIR, "GAD_reviewer1_revision_audit.xlsx"))

    print("\n[11/11] Reproducibility manifest ...")
    regime_results = {
        "design_weight_primary": design_results,
        "class_weight_sensitivity": class_weight_results,
        "smote_sensitivity": smote_results,
        "native_sensitivity": native_results,
    }
    holdout_results = {
        "design_weight_primary": holdout_metrics,
        "class_weight_sensitivity": class_holdout_metrics,
    }
    save_run_manifest(
        os.path.join(OUTPUT_DIR, "GAD_run_manifest.json"),
        TRAIN_PATH, train_idx, holdout_idx, feature_sets,
        sampling_weight, regime_results, holdout_results,
        nested_tuning, threshold_audit, bootstrap_seeds,
        selection_audit, acceptance_checks,
    )

    print(f"\n{div}")
    print("PRIMARY DESIGN-WEIGHTED REDUCED-FEATURE NESTED-CV SUMMARY")
    print(div)
    print(f"  {'Model':<24} {'AUC':>7} {'F1':>7} "
          f"{'Sens.':>8} {'Spec.':>8} {'MCC':>7}")
    print("  " + "-" * 54)
    for name, r in primary_results.items():
        marker = "  <-- selected" if name == selected_model else ""
        print(f"  {name:<24} {r['AUC-ROC']:>7.4f} "
              f"{r['F1']:>7.4f} {r['Sensitivity']:>8.4f} "
              f"{r['Specificity']:>8.4f} "
               f"{r['MCC']:>7.4f}{marker}")

    if selected_model is None:
        print("\n  CONSTRAINT STATUS: No model met nested-CV specificity >= 0.80.")
    else:
        print(f"\n  Selected from development data only: {selected_model}")
    print("  Holdout results were not used for model, parameter, feature, or "
          "threshold selection.")
    print("  Original fig01-fig08 graph types and core result tables were retained.")
    print("  Class-weight, native, SMOTE, Full-56, and diagnostic-only-2 outputs "
          "are sensitivity analyses, not primary success results.")

    n = len([f for f in os.listdir(OUTPUT_DIR)
              if f.endswith((".xlsx", ".svg", ".png", ".json"))])
    print(f"\n  Output directory  :  {OUTPUT_DIR}")
    print(f"  Files generated   :  {n}")
    print(div)


if __name__ == "__main__":
    main()
