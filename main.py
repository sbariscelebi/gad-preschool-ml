"""
main.py
=======
Entry point for the GAD preschool ML pipeline.

Usage
-----
    python main.py

Steps
-----
1. Load data
2. 10-fold stratified cross-validation
3. Independent 20% holdout evaluation
4. Standard figures (Figs 1-6)
5. Explainability: feature importance, DT rules, SHAP (Figs 7-12)
6. Excel exports (metrics + actual-vs-predicted)
"""

import os

from config         import CV_FOLDS, OUTPUT_DIR
from data_loader    import load_data
from pipelines      import build_pipelines
from evaluation     import run_cv, run_holdout
from visualisation  import (
    plot_class_distribution, plot_confusion_matrices,
    plot_metric_comparison, plot_pr_curves,
    plot_radar, plot_roc_curves,
)
from xai            import (
    plot_decision_tree_rules, plot_feature_importance, run_shap,
)
from excel_export   import save_actual_pred_excel, save_metrics_excel


def main() -> None:
    div = "=" * 62
    print(div)
    print("Early Detection of GAD in Preschool Children")
    print("Naive Bayes  |  Logistic Regression  |  "
          "Decision Tree  |  KNN")
    print(div)

    X, y_gad, features = load_data()
    pipes               = build_pipelines()

    print(f"\n[1/5]  {CV_FOLDS}-fold stratified cross-validation ...")
    results, probas = run_cv(pipes, X, y_gad)

    print("\n[2/5]  Holdout evaluation (80/20 split) ...")
    df_ap = run_holdout(pipes, X, y_gad)

    print("\n[3/5]  Standard figures ...")
    plot_class_distribution(y_gad)
    plot_metric_comparison(results)
    plot_roc_curves(probas)
    plot_pr_curves(probas)
    plot_confusion_matrices(probas)
    plot_radar(results)

    print("\n[4/5]  Explainability (XAI) ...")
    plot_feature_importance(X, y_gad, features)
    plot_decision_tree_rules(X, y_gad, features)
    run_shap(X, y_gad, features)

    print("\n[5/5]  Excel outputs ...")
    save_metrics_excel(results)
    save_actual_pred_excel(df_ap)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{div}")
    print("RESULTS SUMMARY  —  GAD")
    print(div)
    print(f"  {'Model':<24} {'AUC':>7} {'F1':>7} "
          f"{'Recall':>8} {'MCC':>7}")
    print("  " + "-" * 54)
    best = max(results, key=lambda k: results[k]["AUC-ROC"])
    for name, r in results.items():
        marker = "  <-- best AUC" if name == best else ""
        print(f"  {name:<24} {r['AUC-ROC']:>7.4f} "
              f"{r['F1']:>7.4f} {r['Recall']:>8.4f} "
              f"{r['MCC']:>7.4f}{marker}")

    n_files = len([
        f for f in os.listdir(OUTPUT_DIR)
        if f.endswith((".xlsx", ".svg", ".png"))
    ])
    print(f"\n  Output directory  :  {OUTPUT_DIR}")
    print(f"  Files generated   :  {n_files}")
    print(div)


if __name__ == "__main__":
    main()
