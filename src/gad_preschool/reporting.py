"""Excel exports, model selection, acceptance checks, provenance, and run manifests."""

from __future__ import annotations

from .settings import *  # noqa: F403
from .data import file_sha256, make_train_holdout_indices
from .models import calculate_metrics, metrics_to_dataframe

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
