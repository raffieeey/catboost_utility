"""CatBoost-based Variance Inflation Factor with native categorical support."""

from __future__ import annotations

import logging
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostError, CatBoostRegressor, Pool
from joblib import Parallel, delayed
from sklearn.metrics import log_loss, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

try:  # package import
    from ..common.cat_feature_utils import (
        filter_cat_features,
        get_cat_feature_indices,
        resolve_cat_features,
    )
    from ..common.result import SelectionResult
    from ..common.validation import validate_dataframe
except ImportError:  # pragma: no cover - fallback for direct local imports
    from common.cat_feature_utils import (
        filter_cat_features,
        get_cat_feature_indices,
        resolve_cat_features,
    )
    from common.result import SelectionResult
    from common.validation import validate_dataframe

logger = logging.getLogger(__name__)

_DEFAULT_CATBOOST_PARAMS = {
    "iterations": 200,
    "depth": 4,
    "learning_rate": 0.1,
    "verbose": 0,
    "allow_writing_files": False,
}

_EPS = 1e-10


class CatBoostVIF:
    """Variance Inflation Factor using CatBoost for native categorical support."""

    def __init__(
        self,
        cat_features: list[str] | list[int] | None = None,
        threshold: float = 5.0,
        scoring_method: str = "oof",
        cv_folds: int = 5,
        holdout_fraction: float = 0.2,
        n_jobs: int = 1,
        max_target_cardinality: int = 50,
        catboost_params: dict | None = None,
        random_state: int | None = 42,
    ):
        self._validate_init_params(
            scoring_method=scoring_method,
            threshold=threshold,
            cv_folds=cv_folds,
            holdout_fraction=holdout_fraction,
            n_jobs=n_jobs,
            max_target_cardinality=max_target_cardinality,
        )

        self.cat_features = cat_features
        self.threshold = threshold
        self.scoring_method = scoring_method
        self.cv_folds = cv_folds
        self.holdout_fraction = holdout_fraction
        self.n_jobs = n_jobs
        self.max_target_cardinality = max_target_cardinality
        self.catboost_params = catboost_params or {}
        self.random_state = random_state

        self._retained_features: list[str] | None = None
        self._vif_table: pd.DataFrame | None = None

    def fit(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute VIF for all features."""
        validate_dataframe(X, caller="CatBoostVIF")
        cat_names = resolve_cat_features(X, self.cat_features)
        self._vif_table = self._compute_all_vif(X, cat_names)
        return self._vif_table.copy()

    def fit_eliminate(self, X: pd.DataFrame) -> SelectionResult:
        """Iteratively drop the highest-VIF feature until all VIF < threshold."""
        validate_dataframe(X, caller="CatBoostVIF")
        cat_names = resolve_cat_features(X, self.cat_features)

        X_current = X.copy()
        current_cats = list(cat_names)
        rejected: list[str] = []
        elimination_history: list[dict] = []

        iteration = 0
        while True:
            vif_table = self._compute_all_vif(X_current, current_cats)
            max_vif_row = self._find_max_vif_row(vif_table)
            if max_vif_row is None:
                warnings.warn(
                    "All VIF values are NaN. Stopping elimination.",
                    UserWarning,
                    stacklevel=2,
                )
                break

            logger.info(
                "Iteration %d: %d features, max VIF=%.4f (%s)",
                iteration,
                len(X_current.columns),
                max_vif_row["vif"],
                max_vif_row["feature"],
            )

            if float(max_vif_row["vif"]) <= self.threshold:
                break

            drop_col = str(max_vif_row["feature"])
            elimination_history.append(
                {
                    "iteration": iteration,
                    "dropped_feature": drop_col,
                    "dropped_vif": float(max_vif_row["vif"]),
                    "remaining_features": len(X_current.columns) - 1,
                }
            )
            rejected.append(drop_col)

            X_current = X_current.drop(columns=[drop_col])
            current_cats = filter_cat_features(current_cats, X_current.columns)

            if X_current.shape[1] < 2:
                warnings.warn(
                    "Only 1 feature remaining - stopping elimination.",
                    UserWarning,
                    stacklevel=2,
                )
                break

            iteration += 1

        self._vif_table = vif_table
        self._retained_features = vif_table["feature"].tolist()

        return SelectionResult(
            selected_features=self._retained_features,
            rejected_features=rejected,
            tentative_features=[],
            metrics=vif_table,
            config={
                "threshold": self.threshold,
                "scoring_method": self.scoring_method,
                "cv_folds": self.cv_folds,
                "holdout_fraction": self.holdout_fraction,
                "n_jobs": self.n_jobs,
                "max_target_cardinality": self.max_target_cardinality,
                "catboost_params": self._get_merged_params(),
                "elimination_history": elimination_history,
            },
            random_state=self.random_state,
        )

    def get_retained_features(self) -> list[str]:
        """Return feature names surviving elimination."""
        if self._retained_features is None:
            raise RuntimeError("Call fit_eliminate() before get_retained_features().")
        return list(self._retained_features)

    def _validate_init_params(
        self,
        *,
        scoring_method: str,
        threshold: float,
        cv_folds: int,
        holdout_fraction: float,
        n_jobs: int,
        max_target_cardinality: int,
    ) -> None:
        if scoring_method not in ("oof", "holdout"):
            raise ValueError(
                f"scoring_method must be 'oof' or 'holdout', got '{scoring_method}'."
            )
        if threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {threshold}.")
        if cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2, got {cv_folds}.")
        if not (0 < holdout_fraction < 1):
            raise ValueError(
                f"holdout_fraction must be in (0, 1), got {holdout_fraction}."
            )
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0.")
        if max_target_cardinality < 2:
            raise ValueError(
                "max_target_cardinality must be >= 2, "
                f"got {max_target_cardinality}."
            )

    def _compute_all_vif(self, X: pd.DataFrame, cat_names: list[str]) -> pd.DataFrame:
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._compute_single_vif)(X, col, cat_names) for col in X.columns
        )
        return (
            pd.DataFrame(results)
            .sort_values("vif", ascending=False, na_position="last")
            .reset_index(drop=True)
        )

    def _find_max_vif_row(self, vif_table: pd.DataFrame) -> pd.Series | None:
        valid_vif = vif_table["vif"].dropna()
        if valid_vif.empty:
            return None
        return vif_table.loc[valid_vif.idxmax()]

    def _compute_single_vif(
        self, X: pd.DataFrame, target_col: str, cat_names: list[str]
    ) -> dict:
        is_categorical = bool(target_col in cat_names)
        y = X[target_col]
        X_pred = X.drop(columns=[target_col])
        pred_cats = filter_cat_features(cat_names, X_pred.columns)

        if is_categorical and y.nunique(dropna=True) > self.max_target_cardinality:
            warnings.warn(
                f"Skipping VIF for '{target_col}': {y.nunique(dropna=True)} unique "
                "values exceeds max_target_cardinality="
                f"{self.max_target_cardinality}.",
                UserWarning,
                stacklevel=2,
            )
            return self._nan_result(target_col, is_categorical)

        y, X_pred = self._drop_null_target_rows(y, X_pred)

        if y.nunique(dropna=True) <= 1 or len(y) < 2:
            return self._no_signal_result(target_col, is_categorical)

        try:
            if is_categorical:
                r2 = self._compute_categorical_r2(X_pred, y, pred_cats)
            else:
                r2 = self._compute_continuous_r2(X_pred, y, pred_cats)
        except (ValueError, CatBoostError) as exc:
            warnings.warn(
                f"VIF failed for '{target_col}' ({exc}). Returning NaN.",
                UserWarning,
                stacklevel=2,
            )
            return self._nan_result(target_col, is_categorical)

        if not np.isfinite(r2):
            return self._nan_result(target_col, is_categorical)

        r2, clamped = self._clamp_r2(float(r2))
        vif = 1.0 / (1.0 - r2)
        return {
            "feature": target_col,
            "vif": vif,
            "r_squared": r2,
            "is_categorical": is_categorical,
            "clamped": clamped,
        }

    def _compute_continuous_r2(
        self, X: pd.DataFrame, y: pd.Series, cat_names: list[str]
    ) -> float:
        params = self._get_merged_params()
        params["random_seed"] = self.random_state or 0
        params["loss_function"] = "RMSE"

        if self.scoring_method == "oof":
            return self._oof_r2_regression(X, y, cat_names, params)
        return self._holdout_r2_regression(X, y, cat_names, params)

    def _compute_categorical_r2(
        self, X: pd.DataFrame, y: pd.Series, cat_names: list[str]
    ) -> float:
        params = self._get_merged_params()
        params["random_seed"] = self.random_state or 0
        params["loss_function"] = "MultiClass"
        params["auto_class_weights"] = "Balanced"

        if self.scoring_method == "oof":
            return self._oof_r2_classification(X, y, cat_names, params)
        return self._holdout_r2_classification(X, y, cat_names, params)

    def _oof_r2_regression(
        self, X: pd.DataFrame, y: pd.Series, cat_names: list[str], params: dict
    ) -> float:
        n_splits = min(self.cv_folds, len(X))
        if n_splits < 2:
            warnings.warn(
                "Insufficient rows for OOF regression. Falling back to holdout.",
                UserWarning,
                stacklevel=2,
            )
            return self._holdout_r2_regression(X, y, cat_names, params)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        oof_preds = np.full(len(y), np.nan)

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            cat_indices = get_cat_feature_indices(X_train, cat_names)

            model = CatBoostRegressor(**params)
            model.fit(
                Pool(X_train, y_train, cat_features=cat_indices if cat_indices else None),
                eval_set=Pool(X_val, y_val, cat_features=cat_indices if cat_indices else None),
                early_stopping_rounds=20,
            )
            oof_preds[val_idx] = model.predict(X_val)

        return float(r2_score(y, oof_preds))

    def _holdout_r2_regression(
        self, X: pd.DataFrame, y: pd.Series, cat_names: list[str], params: dict
    ) -> float:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.holdout_fraction,
            random_state=self.random_state,
        )
        cat_indices = get_cat_feature_indices(X_train, cat_names)

        model = CatBoostRegressor(**params)
        model.fit(
            Pool(X_train, y_train, cat_features=cat_indices if cat_indices else None),
            eval_set=Pool(X_val, y_val, cat_features=cat_indices if cat_indices else None),
            early_stopping_rounds=20,
        )
        preds = model.predict(X_val)
        return float(r2_score(y_val, preds))

    def _oof_r2_classification(
        self, X: pd.DataFrame, y: pd.Series, cat_names: list[str], params: dict
    ) -> float:
        y_encoded, n_classes = _encode_labels(y)
        min_class_count = int(pd.Series(y_encoded).value_counts().min())
        n_splits = min(self.cv_folds, min_class_count)

        if n_splits < 2:
            warnings.warn(
                "Insufficient class frequency for StratifiedKFold. "
                "Falling back to holdout.",
                UserWarning,
                stacklevel=2,
            )
            return self._holdout_r2_classification(X, y, cat_names, params)

        kf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        oof_proba = np.full((len(y), n_classes), np.nan)

        for train_idx, val_idx in kf.split(X, y_encoded):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_encoded.iloc[train_idx], y_encoded.iloc[val_idx]
            cat_indices = get_cat_feature_indices(X_train, cat_names)

            model = CatBoostClassifier(**params)
            model.fit(
                Pool(X_train, y_train, cat_features=cat_indices if cat_indices else None),
                eval_set=Pool(X_val, y_val, cat_features=cat_indices if cat_indices else None),
                early_stopping_rounds=20,
            )
            oof_proba[val_idx] = model.predict_proba(X_val)

        return _mcfadden_r2(y_encoded.values, oof_proba, n_classes)

    def _holdout_r2_classification(
        self, X: pd.DataFrame, y: pd.Series, cat_names: list[str], params: dict
    ) -> float:
        y_encoded, n_classes = _encode_labels(y)
        stratify = y_encoded if _can_stratify(y_encoded) else None
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y_encoded,
                test_size=self.holdout_fraction,
                random_state=self.random_state,
                stratify=stratify,
            )
        except ValueError:
            warnings.warn(
                "Stratified holdout split failed; retrying without stratification.",
                UserWarning,
                stacklevel=2,
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y_encoded,
                test_size=self.holdout_fraction,
                random_state=self.random_state,
                stratify=None,
            )

        cat_indices = get_cat_feature_indices(X_train, cat_names)
        model = CatBoostClassifier(**params)
        model.fit(
            Pool(X_train, y_train, cat_features=cat_indices if cat_indices else None),
            eval_set=Pool(X_val, y_val, cat_features=cat_indices if cat_indices else None),
            early_stopping_rounds=20,
        )
        proba = model.predict_proba(X_val)
        return _mcfadden_r2(y_val.values, proba, n_classes)

    def _drop_null_target_rows(
        self, y: pd.Series, X_pred: pd.DataFrame
    ) -> tuple[pd.Series, pd.DataFrame]:
        mask = y.notna()
        if not mask.all():
            y = y[mask]
            X_pred = X_pred[mask]
        return y, X_pred

    def _clamp_r2(self, r2: float) -> tuple[float, bool]:
        if r2 < 0:
            return 0.0, True
        if r2 >= 1.0:
            return 1.0 - _EPS, True
        return r2, False

    def _no_signal_result(self, target_col: str, is_categorical: bool) -> dict:
        return {
            "feature": target_col,
            "vif": 1.0,
            "r_squared": 0.0,
            "is_categorical": is_categorical,
            "clamped": False,
        }

    def _nan_result(self, target_col: str, is_categorical: bool) -> dict:
        return {
            "feature": target_col,
            "vif": np.nan,
            "r_squared": np.nan,
            "is_categorical": is_categorical,
            "clamped": False,
        }

    def _get_merged_params(self) -> dict:
        params = deepcopy(_DEFAULT_CATBOOST_PARAMS)
        params.update(self.catboost_params)
        return params


def _encode_labels(y: pd.Series) -> tuple[pd.Series, int]:
    classes = y.dropna().unique()
    label_map = {label: idx for idx, label in enumerate(sorted(classes, key=str))}
    y_encoded = y.map(label_map)
    return y_encoded, len(classes)


def _can_stratify(y: pd.Series) -> bool:
    counts = y.value_counts()
    return len(counts) > 1 and bool((counts >= 2).all())


def _mcfadden_r2(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int) -> float:
    """Compute McFadden pseudo-R2 from log-loss."""
    ll_model = log_loss(y_true, y_proba, labels=list(range(n_classes)))

    class_counts = np.bincount(y_true.astype(int), minlength=n_classes)
    class_freq = class_counts / class_counts.sum()
    null_proba = np.tile(class_freq, (len(y_true), 1))
    ll_null = log_loss(y_true, null_proba, labels=list(range(n_classes)))

    if ll_null == 0:
        return 0.0
    return float(1.0 - (ll_model / ll_null))
