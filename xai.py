"""
xai.py
======
Explainability module: Figures 7, 8, 9, 10, 11, 12.

Functions
---------
plot_feature_importance   — Fig 7: LR standardised coefficients + DT Gini
plot_decision_tree_rules  — Fig 8: Decision Tree at display depth 4
run_shap                  — Figs 9-12: SHAP beeswarm, bar, dot plots
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.impute        import SimpleImputer
from sklearn.linear_model  import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree          import DecisionTreeClassifier, plot_tree
from imblearn.over_sampling import SMOTE

from config import (
    DPI, FONT_AXIS, FONT_TICK, FONT_TITLE, OUTPUT_DIR,
    PALETTE, RANDOM_STATE, TEST_SIZE, to_en,
)
from visualisation import apply_style


def _save(fig, name: str) -> None:
    for fmt in ("svg", "png"):
        path = os.path.join(OUTPUT_DIR, f"{name}.{fmt}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight", format=fmt)
    plt.close(fig)
    print(f"  Saved : {name}.svg / .png")


def _split_and_resample(X: pd.DataFrame, y: np.ndarray):
    """80/20 split → median impute → SMOTE on train partition."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=TEST_SIZE,
        stratify=y, random_state=RANDOM_STATE,
    )
    imp      = SimpleImputer(strategy="median")
    X_tr_imp = imp.fit_transform(X_tr)
    X_val_imp = imp.transform(X_val)

    sm       = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_tr_imp, y_tr)

    return X_tr_imp, X_val_imp, X_res, y_res, y_tr, y_val


# ── Figure 7 — Feature importance ─────────────────────────────────────────

def plot_feature_importance(X: pd.DataFrame, y: np.ndarray,
                            feature_names: list) -> None:
    """
    Top-15 feature importance for LR (standardised coefficients)
    and Decision Tree (Gini importance).
    """
    print("  Feature importance ...")
    X_tr_imp, _, X_res, y_res, _, _ = _split_and_resample(X, y)

    # LR coefficients
    scl    = StandardScaler()
    X_sc   = scl.fit_transform(X_res)
    lr_clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                solver="lbfgs", random_state=RANDOM_STATE)
    lr_clf.fit(X_sc, y_res)
    lr_imp = pd.Series(np.abs(lr_clf.coef_[0]),
                       index=feature_names).sort_values(ascending=False)

    # Decision Tree Gini
    dt_clf = DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                    min_samples_leaf=10,
                                    random_state=RANDOM_STATE)
    dt_clf.fit(X_res, y_res)
    dt_imp = pd.Series(dt_clf.feature_importances_,
                       index=feature_names).sort_values(ascending=False)

    top_n  = 15
    lr_top = lr_imp.head(top_n)
    dt_top = dt_imp.head(top_n)

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "Feature Importance for GAD Prediction  |  Top 15 Predictors",
        fontsize=FONT_TITLE + 2, fontweight="bold", y=1.01,
    )

    configs = [
        (axes[0], lr_top, PALETTE["Logistic Regression"],
         "Logistic Regression", "Absolute standardised coefficient"),
        (axes[1], dt_top, PALETTE["Decision Tree"],
         "Decision Tree",       "Gini importance"),
    ]

    for ax, series, color, title, xlabel in configs:
        labels = [to_en(n) for n in series.index]
        bars   = ax.barh(
            range(len(labels)), series.values,
            color=color, edgecolor="white", linewidth=0.5, zorder=3,
        )
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=FONT_TICK)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel, fontsize=FONT_AXIS)
        ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", pad=8)
        ax.set_axisbelow(True)
        for bar, val in zip(bars, series.values):
            ax.text(
                val + series.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10,
            )

    plt.tight_layout()
    _save(fig, "fig07_feature_importance")


# ── Figure 8 — Decision Tree rules ────────────────────────────────────────

def plot_decision_tree_rules(X: pd.DataFrame, y: np.ndarray,
                              feature_names: list) -> None:
    """Visualise the fitted Decision Tree at display depth 4."""
    print("  Decision Tree rules ...")
    _, _, X_res, y_res, _, _ = _split_and_resample(X, y)

    dt = DecisionTreeClassifier(
        max_depth=6, class_weight="balanced",
        min_samples_leaf=10, random_state=RANDOM_STATE,
    )
    dt.fit(X_res, y_res)

    short_names = [to_en(n, max_len=32) for n in feature_names]

    apply_style()
    fig, ax = plt.subplots(figsize=(28, 10))
    plot_tree(
        dt,
        feature_names=short_names,
        class_names=["No GAD", "GAD"],
        filled=True, rounded=True,
        max_depth=4, fontsize=8, ax=ax,
        impurity=False, precision=2,
    )
    ax.set_title(
        "Decision Tree  —  GAD Classification Rules  (depth shown: 4)",
        fontsize=FONT_TITLE, fontweight="bold", pad=14,
    )
    plt.tight_layout()
    _save(fig, "fig08_decision_tree_rules")


# ── Figures 9-12 — SHAP ───────────────────────────────────────────────────

def run_shap(X: pd.DataFrame, y: np.ndarray,
             feature_names: list) -> None:
    """
    SHAP explainability for Logistic Regression and Decision Tree.

    LR  → LinearExplainer  → beeswarm (Fig 9) + bar (Fig 10)
    DT  → TreeExplainer    → bar (Fig 11) + dot plot (Fig 12)

    SHAP values are computed on the holdout test set only,
    following Ponce-Bobadilla et al. (2024).
    """
    X_tr, X_val, y_tr, _ = train_test_split(
        X, y, test_size=TEST_SIZE,
        stratify=y, random_state=RANDOM_STATE,
    )
    imp       = SimpleImputer(strategy="median")
    X_tr_imp  = imp.fit_transform(X_tr)
    X_val_imp = imp.transform(X_val)

    en_names = [to_en(n) for n in feature_names]

    for model_name, explainer_type in [
        ("Logistic Regression", "linear"),
        ("Decision Tree",       "tree"),
    ]:
        print(f"  SHAP  —  {model_name} ...")
        tag = model_name.lower().replace(" ", "_")

        if explainer_type == "linear":
            scl      = StandardScaler()
            X_tr_s   = scl.fit_transform(X_tr_imp)
            X_val_s  = scl.transform(X_val_imp)
        else:
            X_tr_s  = X_tr_imp
            X_val_s = X_val_imp

        sm     = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        Xr, yr = sm.fit_resample(X_tr_s, y_tr)

        if explainer_type == "linear":
            clf = LogisticRegression(
                max_iter=1000, class_weight="balanced",
                solver="lbfgs", random_state=RANDOM_STATE,
            )
        else:
            clf = DecisionTreeClassifier(
                max_depth=6, class_weight="balanced",
                min_samples_leaf=10, random_state=RANDOM_STATE,
            )
        clf.fit(Xr, yr)

        X_val_df = pd.DataFrame(X_val_s, columns=en_names)

        if explainer_type == "linear":
            explainer = shap.LinearExplainer(clf, Xr, feature_names=en_names)
            shap_exp  = explainer(X_val_df)
            sv_arr    = shap_exp.values

            # Fig 9 — Beeswarm
            apply_style()
            fig, _ = plt.subplots(figsize=(11, 8))
            shap.plots.beeswarm(shap_exp, max_display=15,
                                show=False, color_bar=True)
            plt.title(
                "SHAP Summary  —  Logistic Regression\n"
                "GAD Prediction  |  Top 15 Predictors",
                fontsize=FONT_TITLE, fontweight="bold", pad=12,
            )
            plt.xlabel("SHAP value  (impact on model output)",
                       fontsize=FONT_AXIS)
            plt.tight_layout()
            _save(fig, f"fig09_shap_beeswarm_{tag}")

            # Fig 10 — Bar
            apply_style()
            fig2, ax2 = plt.subplots(figsize=(11, 7))
            shap.plots.bar(shap_exp, max_display=15, ax=ax2, show=False)
            ax2.set_title(
                "SHAP Feature Importance  —  Logistic Regression\n"
                "Mean |SHAP value|  |  Top 15 Predictors",
                fontsize=FONT_TITLE, fontweight="bold", pad=12,
            )
            ax2.set_xlabel("Mean |SHAP value|", fontsize=FONT_AXIS)
            plt.tight_layout()
            _save(fig2, f"fig10_shap_bar_{tag}")

        else:
            # Decision Tree — legacy API
            explainer = shap.TreeExplainer(clf)
            sv_raw    = explainer.shap_values(X_val_df)

            if isinstance(sv_raw, list):
                sv_arr = sv_raw[1]
            elif sv_raw.ndim == 3:
                sv_arr = sv_raw[:, :, 1]
            else:
                sv_arr = sv_raw

            mean_shap = pd.Series(
                np.abs(sv_arr).mean(axis=0), index=en_names
            ).sort_values(ascending=False).head(15)

            # Fig 11 — Bar (DT)
            apply_style()
            fig, ax = plt.subplots(figsize=(11, 8))
            ax.barh(range(len(mean_shap)), mean_shap.values,
                    color=PALETTE["Decision Tree"],
                    edgecolor="white", zorder=3)
            ax.set_yticks(range(len(mean_shap)))
            ax.set_yticklabels(mean_shap.index.tolist(), fontsize=FONT_TICK)
            ax.invert_yaxis()
            ax.set_xlabel("Mean |SHAP value|", fontsize=FONT_AXIS)
            ax.set_title(
                "SHAP Feature Importance  —  Decision Tree\n"
                "GAD Prediction  |  Top 15 Predictors",
                fontsize=FONT_TITLE, fontweight="bold", pad=12,
            )
            ax.set_axisbelow(True)
            for i, v in enumerate(mean_shap.values):
                ax.text(v + mean_shap.max() * 0.01, i,
                        f"{v:.4f}", va="center", fontsize=10)
            plt.tight_layout()
            _save(fig, f"fig11_shap_bar_{tag}")

            # Fig 12 — Dot plot (DT)
            top_feats = mean_shap.index.tolist()
            top_idx   = [en_names.index(f) for f in top_feats]
            sv_top    = sv_arr[:, top_idx]
            xv_top    = X_val_df.values[:, top_idx]
            xv_norm   = (xv_top - xv_top.min(0)) / (
                         xv_top.max(0) - xv_top.min(0) + 1e-8)

            apply_style()
            fig2, ax2 = plt.subplots(figsize=(11, 7))
            cmap = plt.cm.RdBu_r

            for fi in range(len(top_feats)):
                rng      = np.random.default_rng(fi)
                y_jitter = fi + rng.uniform(-0.3, 0.3, sv_top.shape[0])
                sc = ax2.scatter(
                    sv_top[:, fi], y_jitter,
                    c=xv_norm[:, fi], cmap=cmap,
                    alpha=0.5, s=12, linewidths=0,
                )

            ax2.set_yticks(range(len(top_feats)))
            ax2.set_yticklabels(top_feats, fontsize=FONT_TICK)
            ax2.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax2.set_xlabel("SHAP value  (impact on model output)",
                           fontsize=FONT_AXIS)
            ax2.set_title(
                "SHAP Dot Plot  —  Decision Tree\n"
                "GAD Prediction  |  Top 15 Predictors",
                fontsize=FONT_TITLE, fontweight="bold", pad=12,
            )
            cbar = plt.colorbar(sc, ax=ax2, fraction=0.03)
            cbar.set_label("Feature value (normalised)", fontsize=FONT_LEGEND)
            plt.tight_layout()
            _save(fig2, f"fig12_shap_dot_{tag}")
