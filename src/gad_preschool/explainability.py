"""Feature-importance, decision-tree, and SHAP explanations for fitted primary models."""

from __future__ import annotations

from .settings import *  # noqa: F403
from .figures import _save

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
