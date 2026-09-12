# Data access

The raw participant-level workbook is intentionally excluded from this Git repository.

1. Obtain the Duke Preschool Anxiety Study training data from the [Harvard Dataverse record](https://doi.org/10.7910/DVN/N42LWG).
2. Place the training workbook outside the repository or in this directory under the name `Training Data.xlsx`.
3. Set the `GAD_TRAIN_PATH` environment variable to its full path.
4. Set `GAD_OUTPUT_DIR` to a writable output directory.

The analysis expects 917 observations before the deterministic 80/20 development–holdout split. It does not use a separate repository-provided testing workbook.

Do not commit raw data, participant identifiers, or locally generated prediction files without checking the data provider's terms.

