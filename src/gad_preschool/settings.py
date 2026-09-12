"""Shared imports, constants, feature labels, and plotting style."""

from __future__ import annotations

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
