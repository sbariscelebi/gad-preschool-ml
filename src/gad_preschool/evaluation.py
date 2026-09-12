"""Nested cross-validation and one-time same-source internal holdout evaluation."""

from __future__ import annotations

from .settings import *  # noqa: F403
from .models import (
    _fit_search,
    _round_metric,
    _validated_weights,
    calculate_metrics,
    cross_fitted_predict_proba,
    derive_screening_threshold,
    extract_fold_feature_importance,
    make_search,
)

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
