"""Recursive Feature Elimination (RFE) using CatBoost with native categorical support."""

from __future__ import annotations

import logging
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

try:  # package import
    from ..common.cat_feature_utils import (
        filter_cat_features,
        get_cat_feature_indices,
        resolve_cat_features,
    )
    from ..common.result import SelectionResult
    from ..common.validation import validate_dataframe, validate_target
except ImportError:  # pragma: no cover - fallback for direct local imports
    from common.cat_feature_utils import (
        filter_cat_features,
        get_cat_feature_indices,
        resolve_cat_features,
    )
    from common.result import SelectionResult
    from common.validation import validate_dataframe, validate_target

logger = logging.getLogger(__name__)

_DEFAULT_CATBOOST_PARAMS = {
    "iterations": 100,
    "depth": 5,
    "learning_rate": 0.1,
    "verbose": 0,
    "allow_writing_files": False,
}


class CatBoostRFE:
    """Recursive Feature Elimination using CatBoost.

    Repeatedly trains a CatBoost model, ranks features by importance, removes
    the least important, and continues until ``n_features_to_select`` features
    remain. Categorical features are handled natively through CatBoost's Pool
    API, so no encoding is required.
    """

    def __init__(
        self,
        n_features_to_select: int | float | None = None,
        step: int | float = 1,
        cat_features: list[str] | list[int] | None = None,
        importance_type: str = "PredictionValuesChange",
        task_type: str = "classification",
        catboost_params: dict | None = None,
        random_state: int | None = 42,
    ):
        self._validate_init_params(
            n_features_to_select=n_features_to_select,
            step=step,
            task_type=task_type,
        )

        self.n_features_to_select = n_features_to_select
        self.step = step
        self.cat_features = cat_features
        self.importance_type = importance_type
        self.task_type = task_type
        self.catboost_params = catboost_params or {}
        self.random_state = random_state

        # Fitted attributes (set in fit)
        self.n_features_: int | None = None
        self.feature_names_: list[str] | None = None
        self.support_: np.ndarray | None = None
        self.ranking_: np.ndarray | None = None
        self.importance_history_: pd.DataFrame | None = None
        self.n_features_selected_: int | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> CatBoostRFE:
        """Run recursive feature elimination."""
        X_fit, y_fit = self._prepare_fit_inputs(X, y)

        feature_names = list(X_fit.columns)
        n_features = X_fit.shape[1]
        self.n_features_ = n_features
        self.feature_names_ = feature_names

        n_select = self._resolve_n_select(n_features)
        cat_names_all = resolve_cat_features(X_fit, self.cat_features)

        current_features = list(feature_names)
        elimination_rounds: list[list[str]] = []
        importance_records: list[dict] = []

        iteration = 0
        while True:
            iteration += 1
            importances = self._fit_and_get_importance(
                X_fit, y_fit, current_features, cat_names_all
            )

            self._record_importance_history(
                importance_records=importance_records,
                iteration=iteration,
                current_features=current_features,
                importances=importances,
                n_select=n_select,
            )

            n_remaining = len(current_features)
            if n_remaining <= n_select:
                break

            n_to_remove = self._resolve_step_count(n_remaining)
            n_to_remove = min(n_to_remove, n_remaining - n_select)

            # Least important features first.
            order = np.argsort(importances, kind="stable")
            to_remove = [current_features[i] for i in order[:n_to_remove]]
            elimination_rounds.append(to_remove)
            remove_set = set(to_remove)
            current_features = [f for f in current_features if f not in remove_set]

            logger.info(
                "Iteration %d: %d features remaining after removing %d.",
                iteration,
                len(current_features),
                n_to_remove,
            )

        self._finalize_fit_state(
            selected_features=current_features,
            elimination_rounds=elimination_rounds,
            importance_records=importance_records,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return only the selected features."""
        self._check_is_fitted()
        missing = [col for col in self.feature_names_ if col not in X.columns]
        if missing:
            raise ValueError(
                "Input DataFrame is missing columns seen during fit: "
                f"{missing}."
            )
        selected = self.get_feature_names_out()
        return X[selected].copy()

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and return only the selected features."""
        return self.fit(X, y).transform(X)

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Get a boolean mask or index array of selected features."""
        self._check_is_fitted()
        if indices:
            return np.where(self.support_)[0]
        return self.support_.copy()

    def get_feature_names_out(self) -> list[str]:
        """Return names of selected features."""
        self._check_is_fitted()
        return [f for f, s in zip(self.feature_names_, self.support_) if s]

    def get_selection_result(self) -> SelectionResult:
        """Return a standardized SelectionResult."""
        self._check_is_fitted()
        return SelectionResult(
            selected_features=self.get_feature_names_out(),
            rejected_features=[
                f for f, s in zip(self.feature_names_, self.support_) if not s
            ],
            tentative_features=[],
            metrics=self.importance_history_,
            config={
                "n_features_to_select": self.n_features_to_select,
                "n_features_selected": self.n_features_selected_,
                "step": self.step,
                "importance_type": self.importance_type,
                "task_type": self.task_type,
                "catboost_params": self._get_merged_params(),
            },
            random_state=self.random_state,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_init_params(
        self,
        *,
        n_features_to_select: int | float | None,
        step: int | float,
        task_type: str,
    ) -> None:
        if task_type not in ("classification", "regression"):
            raise ValueError(
                f"task_type must be 'classification' or 'regression', got '{task_type}'."
            )

        if n_features_to_select is not None:
            if isinstance(n_features_to_select, bool):
                raise TypeError("n_features_to_select must be an int, float, or None.")
            if isinstance(n_features_to_select, float):
                if not (0 < n_features_to_select <= 1):
                    raise ValueError(
                        "n_features_to_select as a float must be in (0, 1], "
                        f"got {n_features_to_select}."
                    )
            elif isinstance(n_features_to_select, (int, np.integer)):
                n_features_to_select = int(n_features_to_select)
                if n_features_to_select <= 0:
                    raise ValueError(
                        "n_features_to_select as an int must be > 0, "
                        f"got {n_features_to_select}."
                    )
            else:
                raise TypeError("n_features_to_select must be an int, float, or None.")

        if isinstance(step, bool):
            raise TypeError("step must be an int or float.")
        if isinstance(step, float):
            if not (0 < step <= 1):
                raise ValueError(
                    f"step as a float must be in (0, 1], got {step}."
                )
        elif isinstance(step, (int, np.integer)):
            step = int(step)
            if step <= 0:
                raise ValueError(f"step as an int must be > 0, got {step}.")
        else:
            raise TypeError("step must be an int or float.")

    def _prepare_fit_inputs(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        validate_dataframe(X, caller="CatBoostRFE")
        validate_target(y, X, caller="CatBoostRFE")

        if y.isnull().any():
            mask = y.notna()
            dropped = int((~mask).sum())
            warnings.warn(
                f"[CatBoostRFE] Dropping {dropped} rows with null targets.",
                UserWarning,
                stacklevel=2,
            )
            X = X.loc[mask.values].copy()
            y = y.loc[mask.values].copy()

        if len(X) < 2:
            raise ValueError(
                "[CatBoostRFE] Need at least 2 rows after dropping null targets."
            )

        return X, y

    def _resolve_n_select(self, n_features: int) -> int:
        val = self.n_features_to_select
        if val is None:
            return max(1, n_features // 2)
        if isinstance(val, float):
            return min(n_features, max(1, int(round(val * n_features))))
        return min(int(val), n_features)

    def _resolve_step_count(self, n_remaining: int) -> int:
        if isinstance(self.step, float):
            count = int(self.step * n_remaining)
        else:
            count = int(self.step)
        return max(1, count)

    def _fit_and_get_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        current_features: list[str],
        cat_names_all: list[str],
    ) -> np.ndarray:
        X_sub = X[current_features]
        cat_current = filter_cat_features(cat_names_all, current_features)
        cat_indices = get_cat_feature_indices(X_sub, cat_current)

        model = self._build_model()
        pool = Pool(
            X_sub,
            y,
            cat_features=cat_indices if cat_indices else None,
        )
        model.fit(pool)

        importances = model.get_feature_importance(pool, type=self.importance_type)
        return np.asarray(importances, dtype=float)

    def _build_model(self):
        params = self._get_merged_params()
        params["random_seed"] = self.random_state if self.random_state is not None else 0
        if self.task_type == "classification":
            return CatBoostClassifier(**params)
        return CatBoostRegressor(**params)

    def _record_importance_history(
        self,
        *,
        importance_records: list[dict],
        iteration: int,
        current_features: list[str],
        importances: np.ndarray,
        n_select: int,
    ) -> None:
        # Within-iteration rank: 1 = most important.
        order = np.argsort(-importances, kind="stable")
        within_rank = np.empty(len(current_features), dtype=int)
        for pos, idx in enumerate(order):
            within_rank[idx] = pos + 1

        for idx, feature_name in enumerate(current_features):
            importance_records.append(
                {
                    "iteration": iteration,
                    "feature": feature_name,
                    "importance": float(importances[idx]),
                    "rank": int(within_rank[idx]),
                    # A feature is 'selected' this iteration if it would survive
                    # the target cut based on within-iteration ranking.
                    "selected": bool(within_rank[idx] <= n_select),
                }
            )

    def _finalize_fit_state(
        self,
        *,
        selected_features: list[str],
        elimination_rounds: list[list[str]],
        importance_records: list[dict],
    ) -> None:
        name_to_idx = {name: i for i, name in enumerate(self.feature_names_)}

        ranking = np.zeros(self.n_features_, dtype=int)
        selected_set = set(selected_features)
        for name in selected_features:
            ranking[name_to_idx[name]] = 1

        # Features removed earliest get the highest rank number.
        rank = 2
        for round_features in reversed(elimination_rounds):
            for name in round_features:
                ranking[name_to_idx[name]] = rank
            rank += 1

        self.ranking_ = ranking
        self.support_ = np.array(
            [name in selected_set for name in self.feature_names_], dtype=bool
        )
        self.n_features_selected_ = int(self.support_.sum())
        self.importance_history_ = pd.DataFrame(
            importance_records,
            columns=["iteration", "feature", "importance", "rank", "selected"],
        )

    def _check_is_fitted(self) -> None:
        if self.support_ is None:
            raise RuntimeError("Call fit() before using this method.")

    def _get_merged_params(self) -> dict:
        params = deepcopy(_DEFAULT_CATBOOST_PARAMS)
        params.update(self.catboost_params)
        return params
