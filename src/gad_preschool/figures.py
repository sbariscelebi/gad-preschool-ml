"""Performance, calibration, uncertainty, threshold, and clinical-impact figures."""

from __future__ import annotations

from .settings import *  # noqa: F403
from .models import _validated_weights, calculate_metrics

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
