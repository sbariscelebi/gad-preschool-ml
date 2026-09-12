"""Top-level orchestration for the completed preschool GAD analysis workflow."""

from __future__ import annotations

from .settings import *  # noqa: F403
from .data import build_feature_sets, load_data, make_train_holdout_indices
from .models import build_model_specs, metrics_to_dataframe, show_dataframe
from .evaluation import fit_and_evaluate_holdout, run_nested_cv
from .figures import (
    build_calibration_and_ci_frames,
    build_clinical_impact_table,
    paired_feature_set_differences,
    plot_class_distribution,
    plot_confusion_matrices,
    plot_metric_comparison,
    plot_pr_curves,
    plot_radar,
    plot_reliability,
    plot_resampling_comparison,
    plot_roc_curves,
    plot_threshold_tradeoff,
    summarize_feature_stability,
)
from .explainability import (
    plot_decision_tree_rules,
    plot_ensemble_feature_importance_supplement,
    plot_feature_importance,
    run_shap,
)
from .reporting import (
    build_feature_audit,
    default_metrics_to_dataframe,
    run_acceptance_checks,
    save_actual_pred_excel,
    save_frames_excel,
    save_revision_audit_excel,
    save_run_manifest,
    select_primary_model,
    unweighted_metrics_to_dataframe,
)

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
