# Reproducibility notes

## Configuration

The analysis is controlled near the beginning of `src/gad_preschool_analysis.py`.

- `QUICK_TEST = False` selects the full analysis.
- `RUN_NATIVE_SENSITIVITY = False` leaves the optional native sensitivity analysis disabled.
- `RUN_MISSINGNESS_AUDIT = False` records that the dedicated missingness sensitivity analysis was not run.
- `SKIP_SHAP = False` enables SHAP outputs.
- `REQUIRE_XGBOOST = True` prevents silent omission of XGBoost.
- `RANDOM_STATE = 42` fixes the top-level seed.

## Archived outputs

`results/full_analysis` contains the retained workbooks and figures from the completed full analysis. The repository was refreshed without rerunning the computational pipeline. `results/manifest.json` records hashes of committed artifacts so downloaded files can be checked for accidental modification.

## Environment limitation

The archived notebook recorded Python 3.12.12, but exact historical package versions were not embedded in the retained result files. `requirements.txt` therefore specifies minimum compatible versions rather than claiming an unavailable lockfile.

