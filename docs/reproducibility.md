# Reproducibility notes

## Configuration

The analysis is configured in `src/gad_preschool/settings.py` and orchestrated by `src/gad_preschool/main.py`.

- `QUICK_TEST = False` selects the full analysis.
- `RUN_NATIVE_SENSITIVITY = False` leaves the optional native sensitivity analysis disabled.
- `RUN_MISSINGNESS_AUDIT = False` records that the dedicated missingness sensitivity analysis was not run.
- `SKIP_SHAP = False` enables SHAP outputs.
- `REQUIRE_XGBOOST = True` prevents silent omission of XGBoost.
- `RANDOM_STATE = 42` fixes the top-level seed.

Install the package and run the complete workflow from the repository root:

```bash
python -m pip install -e .
python -m gad_preschool
```

Set `GAD_TRAIN_PATH` and `GAD_OUTPUT_DIR` before running, as described in the main README.

## Archived outputs

`results/full_analysis` contains the retained workbooks and figures from the completed full analysis. The repository was refreshed without rerunning the computational pipeline. `results/manifest.json` records hashes of committed artifacts so downloaded files can be checked for accidental modification.

## Environment limitation

The archived notebook recorded Python 3.12.12, but exact historical package versions were not embedded in the retained result files. `requirements.txt` therefore specifies minimum compatible versions rather than claiming an unavailable lockfile.
