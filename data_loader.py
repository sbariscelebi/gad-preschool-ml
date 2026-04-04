"""
data_loader.py
==============
Loads Training_Data.xlsx from the Harvard Dataverse PAPA dataset.

Returns
-------
X            : pd.DataFrame  — 56 behavioural / affective predictors
y_gad        : np.ndarray    — binary GAD label (0 = No GAD, 1 = GAD)
feature_names: list[str]     — original column names (before EN mapping)

Data policy
-----------
Only Training_Data.xlsx is used.
Testing_Data.xlsx is never loaded or referenced at any stage.
"""

import locale

try:
    locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
except Exception:
    try:
        locale.setlocale(locale.LC_ALL, "C.UTF-8")
    except Exception:
        locale.setlocale(locale.LC_ALL, "C")

import numpy as np
import pandas as pd

from config import TRAIN_PATH


def load_data(path: str = TRAIN_PATH):
    """
    Load and clean the PAPA training file.

    Steps
    -----
    1. Read Excel.
    2. Drop onset columns (temporal leakage), Subject, SAD, Sampling Weight.
    3. Convert SPSS dot-encoded missing values (period = NaN).
    4. Cast all feature columns to numeric.

    Parameters
    ----------
    path : str
        Path to Training_Data.xlsx.

    Returns
    -------
    X, y_gad, feature_names
    """
    df = pd.read_excel(path)
    print(f"Dataset loaded  —  {df.shape[0]} samples  |  {df.shape[1]} columns")

    drop_fixed   = ["Subject", "SAD", "GAD", "Sampling Weight"]
    onset_cols   = [c for c in df.columns if "Onset" in c]
    drop_cols    = drop_fixed + onset_cols
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    for col in feature_cols:
        X[col] = (
            X[col].astype(str).str.strip()
                  .replace({"." : np.nan, "" : np.nan})
        )
        X[col] = pd.to_numeric(X[col], errors="coerce")

    y_gad = df["GAD"].values

    print(f"Predictors      :  {len(feature_cols)}")
    print(f"Onset cols dropped  :  {len(onset_cols)}")
    print(f"GAD prevalence  :  {y_gad.mean():.3f}  "
          f"({int(y_gad.sum())} positive  /  {len(y_gad)} total)")

    return X, y_gad, feature_cols
