# Analysis architecture

The codebase is organised as a small Python package whose modules follow the analysis workflow. The separation is structural: it improves navigation, testing, and maintenance without changing the archived analyses or numerical results.

```text
settings
   ├── data
   ├── models ── evaluation
   ├── figures ── explainability
   └── reporting
          ↑
       main (orchestration)
```

## Module responsibilities

- `settings.py` contains environment-driven paths, analysis switches, constants, shared imports, and plotting style.
- `data.py` loads the source workbook, checks expected fields, constructs feature sets, and creates the development/holdout split.
- `models.py` defines preprocessing pipelines, candidate estimators, hyperparameter searches, prediction helpers, and metric calculations.
- `evaluation.py` runs nested cross-validation and evaluates the selected model on the same-source internal holdout.
- `figures.py` prepares performance summaries and creates diagnostic and comparison figures.
- `explainability.py` produces model-specific importance, decision-tree, and SHAP outputs.
- `reporting.py` writes workbooks and manifests, runs acceptance checks, and applies the prespecified model-selection rule.
- `main.py` coordinates these modules in analysis order; it is the only end-to-end workflow entry point.

## Design rules

Lower-level modules do not call the workflow orchestrator. `main.py` passes data and fitted objects between stages explicitly. Generated artifacts remain outside the package and are written to the configured output directory.

Run the complete workflow from the repository root with `python -m gad_preschool` after installing the project and setting the required input and output paths.
