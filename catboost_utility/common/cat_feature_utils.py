"""Utilities for resolving and managing categorical feature specifications."""

from __future__ import annotations

import pandas as pd
from pandas import CategoricalDtype
from pandas.api.types import is_bool_dtype, is_object_dtype


def resolve_cat_features(
    X: pd.DataFrame,
    cat_features: list[str] | list[int] | None = None,
) -> list[str]:
    """Resolve cat_features to a canonical list of column names.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    cat_features : list[str] | list[int] | None
        - None: auto-detect from dtype (object, category, bool).
        - list[str]: column names (validated against X.columns).
        - list[int]: positional indices (converted to names).

    Returns
    -------
    list[str]
        Canonical list of categorical feature column names.
    """
    if cat_features is None:
        return _auto_detect_cat_features(X)

    if len(cat_features) == 0:
        return []

    # Determine if indices or names
    if all(isinstance(f, (int,)) for f in cat_features):
        return _resolve_from_indices(X, cat_features)
    elif all(isinstance(f, str) for f in cat_features):
        return _resolve_from_names(X, cat_features)
    else:
        raise TypeError(
            "cat_features must be all strings (column names) or all integers "
            f"(column indices), got mixed types: {[type(f).__name__ for f in cat_features]}."
        )


def filter_cat_features(
    cat_features: list[str],
    remaining_columns: list[str] | pd.Index,
) -> list[str]:
    """Filter categorical feature names to only those still present in the column set.

    Use this after dropping columns (e.g., during VIF elimination) to keep
    the cat_features list in sync without positional index drift.

    Parameters
    ----------
    cat_features : list[str]
        Canonical list of categorical feature names.
    remaining_columns : list[str] | pd.Index
        Columns that are still present after dropping.

    Returns
    -------
    list[str]
        Filtered list containing only names that exist in remaining_columns.
    """
    remaining_set = set(remaining_columns)
    return [f for f in cat_features if f in remaining_set]


def get_cat_feature_indices(
    X: pd.DataFrame,
    cat_features: list[str],
) -> list[int]:
    """Convert canonical cat_feature names to positional indices for CatBoost.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    cat_features : list[str]
        Canonical categorical feature names.

    Returns
    -------
    list[int]
        Positional indices of categorical features in X.
    """
    col_to_idx = {col: i for i, col in enumerate(X.columns)}
    return [col_to_idx[f] for f in cat_features if f in col_to_idx]


def _auto_detect_cat_features(X: pd.DataFrame) -> list[str]:
    """Auto-detect categorical features from dtype."""
    cat_cols = []
    for col in X.columns:
        dtype = X[col].dtype
        if (
            is_object_dtype(dtype)
            or isinstance(dtype, CategoricalDtype)
            or is_bool_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
        ):
            cat_cols.append(col)
    return cat_cols


def _resolve_from_indices(X: pd.DataFrame, indices: list[int]) -> list[str]:
    """Convert integer indices to column names."""
    n_cols = len(X.columns)
    invalid = [i for i in indices if i < 0 or i >= n_cols]
    if invalid:
        raise IndexError(
            f"cat_features indices out of range (0–{n_cols - 1}): {invalid}."
        )
    return [X.columns[i] for i in indices]


def _resolve_from_names(X: pd.DataFrame, names: list[str]) -> list[str]:
    """Validate column names exist in the DataFrame."""
    missing = [n for n in names if n not in X.columns]
    if missing:
        raise ValueError(
            f"cat_features names not found in X.columns: {missing}."
        )
    return list(names)
