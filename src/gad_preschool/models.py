"""Model specifications, weighted metrics, threshold selection, and hyperparameter search."""

from __future__ import annotations

from .settings import *  # noqa: F403

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
