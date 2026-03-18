"""Shared input validation for all catboost_utility modules."""

from __future__ import annotations

import warnings

import pandas as pd


def validate_dataframe(
    X: pd.DataFrame,
    *,
    min_rows: int = 20,
    max_null_fraction: float = 0.5,
    caller: str = "",
) -> None:
    """Validate a feature DataFrame before fitting.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix to validate.
    min_rows : int
        Minimum number of rows required.
    max_null_fraction : float
        Warn if any column exceeds this null fraction (0–1).
    caller : str
        Name of the calling module for clearer error messages.
    """
    prefix = f"[{caller}] " if caller else ""

    # --- Type check ---
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"{prefix}X must be a pandas DataFrame, got {type(X).__name__}."
        )

    # --- Empty check ---
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"{prefix}X must have at least 1 row and 1 column.")

    # --- Minimum rows ---
    if X.shape[0] < min_rows:
        warnings.warn(
            f"{prefix}X has only {X.shape[0]} rows (min recommended: {min_rows}). "
            "Estimates may be unstable.",
            UserWarning,
            stacklevel=3,
        )

    # --- Duplicate column names ---
    dupes = X.columns[X.columns.duplicated()].unique().tolist()
    if dupes:
        raise ValueError(
            f"{prefix}X contains duplicate column names: {dupes}."
        )

    # --- Unsupported dtypes ---
    unsupported = []
    for col in X.columns:
        dtype = X[col].dtype
        if pd.api.types.is_datetime64_any_dtype(dtype):
            unsupported.append((col, str(dtype)))
        elif pd.api.types.is_complex_dtype(dtype):
            unsupported.append((col, str(dtype)))
        elif pd.api.types.is_timedelta64_dtype(dtype):
            unsupported.append((col, str(dtype)))
    if unsupported:
        raise TypeError(
            f"{prefix}Unsupported column dtypes: "
            + ", ".join(f"'{c}' ({d})" for c, d in unsupported)
            + ". Remove or convert these columns before fitting."
        )

    # --- Constant columns ---
    constant_cols = [col for col in X.columns if X[col].nunique(dropna=True) <= 1]
    if constant_cols:
        warnings.warn(
            f"{prefix}Constant (zero-variance) columns detected: {constant_cols}. "
            "These are uninformative and may cause issues.",
            UserWarning,
            stacklevel=3,
        )

    # --- High null fraction ---
    null_fractions = X.isnull().mean()
    high_null = null_fractions[null_fractions > max_null_fraction]
    if not high_null.empty:
        cols = dict(high_null.round(3))
        warnings.warn(
            f"{prefix}Columns with >{max_null_fraction:.0%} null values: {cols}.",
            UserWarning,
            stacklevel=3,
        )


def validate_target(
    y: pd.Series,
    X: pd.DataFrame,
    *,
    caller: str = "",
) -> None:
    """Validate the target Series against feature DataFrame.

    Parameters
    ----------
    y : pd.Series
        Target variable.
    X : pd.DataFrame
        Feature matrix (for length matching).
    caller : str
        Name of the calling module for clearer error messages.
    """
    prefix = f"[{caller}] " if caller else ""

    if not isinstance(y, pd.Series):
        raise TypeError(
            f"{prefix}y must be a pandas Series, got {type(y).__name__}."
        )

    if len(y) != len(X):
        raise ValueError(
            f"{prefix}Length mismatch: X has {len(X)} rows but y has {len(y)} elements."
        )

    if y.nunique(dropna=True) < 2:
        raise ValueError(
            f"{prefix}y has fewer than 2 unique values. Cannot fit a model."
        )

    null_count = y.isnull().sum()
    if null_count > 0:
        warnings.warn(
            f"{prefix}y contains {null_count} null values. "
            "Rows with null targets will be dropped during fitting.",
            UserWarning,
            stacklevel=3,
        )
