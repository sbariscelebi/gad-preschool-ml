"""
visualisation.py
================
Standard performance figures (Figures 1 – 6).

Figures
-------
Fig 1 — Class distribution bar chart
Fig 2 — Multi-metric grouped bar chart (6 metrics)
Fig 3 — ROC curves (out-of-fold)
Fig 4 — Precision-Recall curves (out-of-fold)
Fig 5 — Confusion matrices (aggregated CV folds)
Fig 6 — Radar chart (5 dimensions)
"""

import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.formatter.use_locale"] = False
matplotlib.rcParams["axes.unicode_minus"]        = False

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix,
    precision_recall_curve, roc_auc_score, roc_curve,
)

from config import (
    CV_FOLDS, DPI, FONT_AXIS, FONT_LEGEND, FONT_TICK, FONT_TITLE,
    OUTPUT_DIR, PALETTE,
)


# ── Global style ───────────────────────────────────────────────────────────

def apply_style() -> None:
    """Apply a clean, publication-ready Matplotlib style."""
    plt.rcParams.update({
        "font.family":               "DejaVu Sans",
        "axes.titlesize":            FONT_TITLE,
        "axes.titleweight":          "bold",
        "axes.labelsize":            FONT_AXIS,
        "xtick.labelsize":           FONT_TICK,
        "ytick.labelsize":           FONT_TICK,
        "legend.fontsize":           FONT_LEGEND,
        "figure.facecolor":          "white",
        "axes.facecolor":            PALETTE["bg"],
        "axes.grid":                 True,
        "grid.color":                PALETTE["grid"],
        "grid.linewidth":            0.6,
        "axes.spines.top":           False,
        "axes.spines.right":         False,
        "axes.formatter.use_locale": False,
        "axes.unicode_minus":        False,
    })


apply_style()


# ── Helper ─────────────────────────────────────────────────────────────────

def _save(fig, name: str) -> None:
    """Save figure as SVG (vector) and PNG (300 DPI)."""
    for fmt in ("svg", "png"):
        path = os.path.join(OUTPUT_DIR, f"{name}.{fmt}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight", format=fmt)
    plt.close(fig)
    print(f"  Saved : {name}.svg / .png")


# ── Figure 1 — Class distribution ─────────────────────────────────────────

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
            fontsize=FONT_TICK, fontweight="bold",
        )

    ax.set_title("Class Distribution  —  GAD  (Training Data)",
                 fontsize=FONT_TITLE, fontweight="bold", pad=10)
    ax.set_ylabel("Number of Samples", fontsize=FONT_AXIS)
    ax.set_xlabel("Class", fontsize=FONT_AXIS)
    ax.set_ylim(0, max(counts) * 1.25)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save(fig, "fig01_class_distribution")


# ── Figure 2 — Metric comparison ──────────────────────────────────────────

def plot_metric_comparison(results: dict) -> None:
    metrics = ["AUC-ROC", "AP", "F1", "Precision", "Recall", "MCC"]
    x = np.arange(len(metrics))
    w = 0.20

    apply_style()
    fig, ax = plt.subplots(figsize=(15, 6))
    fig.suptitle(
        f"Model Performance Comparison  —  GAD  "
        f"({CV_FOLDS}-Fold Stratified Cross-Validation)",
        fontsize=FONT_TITLE + 2, fontweight="bold", y=1.02,
    )

    for i, (name, mdata) in enumerate(results.items()):
        vals   = [mdata[m] for m in metrics]
        offset = (i - 1.5) * w
        bars   = ax.bar(x + offset, vals, w,
                        label=name, color=PALETTE[name],
                        edgecolor="white", linewidth=0.4, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.004,
                f"{v:.3f}",
                ha="center", va="bottom",
                fontsize=8.5, rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=FONT_TICK)
    ax.set_ylim(0.35, 1.15)
    ax.set_ylabel("Score", fontsize=FONT_AXIS)
    ax.set_xlabel("Evaluation Metric", fontsize=FONT_AXIS)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, loc="upper right")
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save(fig, "fig02_metric_comparison")


# ── Figure 3 — ROC curves ─────────────────────────────────────────────────

def plot_roc_curves(probas: dict) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title(
        f"ROC Curves  —  GAD  ({CV_FOLDS}-Fold CV)",
        fontsize=FONT_TITLE, fontweight="bold", pad=10,
    )

    for name, data in probas.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_proba"])
        auc = roc_auc_score(data["y_true"], data["y_proba"])
        ax.plot(fpr, tpr, color=PALETTE[name], linewidth=2.2,
                label=f"{name}  (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4,
            label="Random classifier  (AUC = 0.50)")
    ax.fill_between([0, 1], [0, 1], alpha=0.03, color="gray")
    ax.set_xlabel("False Positive Rate", fontsize=FONT_AXIS)
    ax.set_ylabel("True Positive Rate", fontsize=FONT_AXIS)
    ax.legend(fontsize=FONT_LEGEND, loc="lower right", framealpha=0.9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    _save(fig, "fig03_roc_curves")


# ── Figure 4 — Precision-Recall curves ────────────────────────────────────

def plot_pr_curves(probas: dict) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title(
        f"Precision-Recall Curves  —  GAD  ({CV_FOLDS}-Fold CV)",
        fontsize=FONT_TITLE, fontweight="bold", pad=10,
    )

    for name, data in probas.items():
        prec, rec, _ = precision_recall_curve(
            data["y_true"], data["y_proba"]
        )
        ap = average_precision_score(data["y_true"], data["y_proba"])
        ax.plot(rec, prec, color=PALETTE[name], linewidth=2.2,
                label=f"{name}  (AP = {ap:.4f})")

    base = probas[list(probas.keys())[0]]["y_true"].mean()
    ax.axhline(base, color="gray", linestyle="--", linewidth=1.2,
               label=f"No-skill baseline  (prevalence = {base:.2f})")
    ax.set_xlabel("Recall", fontsize=FONT_AXIS)
    ax.set_ylabel("Precision", fontsize=FONT_AXIS)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    plt.tight_layout()
    _save(fig, "fig04_pr_curves")


# ── Figure 5 — Confusion matrices ─────────────────────────────────────────

def plot_confusion_matrices(probas: dict) -> None:
    apply_style()
    n = len(probas)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.suptitle(
        f"Confusion Matrices  —  GAD  ({CV_FOLDS}-Fold CV)",
        fontsize=FONT_TITLE + 2, fontweight="bold", y=1.02,
    )

    cmap_map = {
        "Naive Bayes":         "Blues",
        "Logistic Regression": "Greens",
        "Decision Tree":       "Oranges",
        "KNN":                 "Purples",
    }

    for ax, (name, data) in zip(axes, probas.items()):
        cm     = confusion_matrix(data["y_true"], data["y_pred"])
        thresh = cm.max() / 2.0
        ax.imshow(cm, cmap=cmap_map[name], aspect="auto",
                  interpolation="nearest")

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        fontsize=FONT_TITLE + 2, fontweight="bold",
                        color="white" if cm[i, j] > thresh else "black")

        ax.set_title(name, fontsize=FONT_TITLE, fontweight="bold", pad=8)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted: 0", "Predicted: 1"],
                           fontsize=FONT_TICK)
        ax.set_yticklabels(["Actual: 0", "Actual: 1"],
                           fontsize=FONT_TICK)
        if ax is axes[0]:
            ax.set_ylabel("Actual Class", fontsize=FONT_AXIS)

    plt.tight_layout()
    _save(fig, "fig05_confusion_matrices")


# ── Figure 6 — Radar chart ────────────────────────────────────────────────

def plot_radar(results: dict) -> None:
    metrics = ["AUC-ROC", "F1", "Precision", "Recall", "MCC"]
    N       = len(metrics)
    angles  = [n / N * 2 * np.pi for n in range(N)] + [0.0]

    apply_style()
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    fig.suptitle(
        "Model Comparison Radar Chart  —  GAD",
        fontsize=FONT_TITLE + 2, fontweight="bold", y=1.03,
    )

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
