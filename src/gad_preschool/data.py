"""Dataset loading, validation, feature-set construction, and deterministic splitting."""

from __future__ import annotations

from .settings import *  # noqa: F403

# ── 5. DATA LOADING AND PRESPECIFIED FEATURE SETS ──────────────────────────
def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data(path: str):
    """Load PAPA data and preserve outcome, IDs, and sampling-design weights."""
    df = pd.read_excel(path)
    required = {"Subject", "GAD", "Sampling Weight",
                *DIRECT_DIAGNOSTIC_FEATURES}
    missing_required = sorted(required.difference(df.columns))
    if missing_required:
        raise ValueError(f"Required columns are missing: {missing_required}")
    if df["Subject"].duplicated().any():
        raise ValueError("Subject IDs are not unique; grouped splitting is required.")
    if not set(df["GAD"].dropna().unique()).issubset({0, 1}):
        raise ValueError("GAD must be binary and encoded as 0/1.")

    drop_fixed = ["Subject", "SAD", "GAD", "Sampling Weight"]
    onset_cols = [c for c in df.columns if "Onset" in c]
    feature_cols = [c for c in df.columns
                    if c not in drop_fixed + onset_cols]

    X = df[feature_cols].copy()
    for col in feature_cols:
        X[col] = pd.to_numeric(
            X[col].astype(str).str.strip().replace({".": np.nan, "": np.nan}),
            errors="coerce",
        )

    y = df["GAD"].astype(int).to_numpy()
    sampling_weight = pd.to_numeric(
        df["Sampling Weight"], errors="coerce").to_numpy(dtype=float)
    if (not np.isfinite(sampling_weight).all() or
            np.any(sampling_weight <= 0)):
        raise ValueError("Sampling Weight must contain finite positive values.")
    print(f"Dataset loaded: {len(df)} samples, {len(feature_cols)} predictors")
    print(f"GAD prevalence: {y.mean():.3f} ({int(y.sum())}/{len(y)})")
    weighted_prevalence = float(np.average(y, weights=sampling_weight))
    kish_n = float(sampling_weight.sum() ** 2 /
                   np.square(sampling_weight).sum())
    print(f"Design-weighted GAD prevalence: {weighted_prevalence:.3f}")
    print(f"Approximate Kish effective sample size: {kish_n:.1f}")
    print(f"Onset columns excluded: {len(onset_cols)}")
    return df, X, y, sampling_weight


def build_feature_sets(X: pd.DataFrame) -> dict:
    """Create the reviewer-requested primary and sensitivity feature sets."""
    missing = [c for c in DIRECT_DIAGNOSTIC_FEATURES if c not in X.columns]
    if missing:
        raise ValueError(f"Diagnostic-definition features missing: {missing}")
    reduced = X.drop(columns=DIRECT_DIAGNOSTIC_FEATURES)
    if set(reduced.columns).intersection(DIRECT_DIAGNOSTIC_FEATURES):
        raise AssertionError("Diagnostic-definition features remain in primary set.")
    return {
        "Primary_reduced": reduced,
        "Full_features_sensitivity": X.copy(),
        "Diagnostic_only_sensitivity": X[DIRECT_DIAGNOSTIC_FEATURES].copy(),
    }


def make_train_holdout_indices(y: np.ndarray) -> tuple:
    """Create the untouched internal holdout before all model development."""
    all_idx = np.arange(len(y))
    train_idx, holdout_idx = train_test_split(
        all_idx, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    if set(train_idx).intersection(holdout_idx):
        raise AssertionError("Training/holdout index overlap detected.")
    if len(train_idx) + len(holdout_idx) != len(y):
        raise AssertionError("Train/holdout split does not cover every sample.")
    return np.sort(train_idx), np.sort(holdout_idx)
