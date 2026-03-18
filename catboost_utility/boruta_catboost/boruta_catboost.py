"""Boruta feature selection using CatBoost with native categorical support."""

from __future__ import annotations

import logging
import warnings
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from scipy import stats

try:  # package import
    from ..common.cat_feature_utils import get_cat_feature_indices, resolve_cat_features
    from ..common.result import SelectionResult
    from ..common.validation import validate_dataframe, validate_target
except ImportError:  # pragma: no cover - fallback for direct local imports
    from common.cat_feature_utils import get_cat_feature_indices, resolve_cat_features
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

# Feature status constants
_TENTATIVE = 0
_CONFIRMED = 1
_REJECTED = 2

_STATUS_TO_LABEL = {
    _TENTATIVE: "tentative",
    _CONFIRMED: "confirmed",
    _REJECTED: "rejected",
}


@dataclass
class _IterationArtifacts:
    """Container for one Boruta iteration output."""

    iteration_seed: int
    real_importances: np.ndarray
    shadow_max: float


class BorutaCatBoost:
    """Boruta feature selection using CatBoost for native categorical support.

    Creates shadow (shuffled) features, trains CatBoost on originals + shadows,
    and uses statistical testing to determine which features are genuinely
    important vs. noise. Categorical shadow features preserve their dtype.
    """

    def __init__(
        self,
        cat_features: list[str] | list[int] | None = None,
        max_iter: int = 100,
        alpha: float = 0.05,
        correction_method: str = "bonferroni",
        importance_type: str = "PredictionValuesChange",
        patience: int = 5,
        task_type: str = "classification",
        catboost_params: dict | None = None,
        random_state: int | None = 42,
    ):
        self._validate_init_params(
            max_iter=max_iter,
            alpha=alpha,
            correction_method=correction_method,
            patience=patience,
            task_type=task_type,
        )

        self.cat_features = cat_features
        self.max_iter = max_iter
        self.alpha = alpha
        self.correction_method = correction_method
        self.importance_type = importance_type
        self.patience = patience
        self.task_type = task_type
        self.catboost_params = catboost_params or {}
        self.random_state = random_state

        # Fitted attributes (set in fit)
        self.n_features_: int | None = None
        self.feature_names_: list[str] | None = None
        self.support_: np.ndarray | None = None
        self.support_weak_: np.ndarray | None = None
        self.ranking_: np.ndarray | None = None
        self.importance_history_: pd.DataFrame | None = None
        self.decision_log_: pd.DataFrame | None = None
        self._n_iter_done: int = 0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BorutaCatBoost:
        """Run Boruta feature selection."""
        X_fit, y_fit = self._prepare_fit_inputs(X, y)
        cat_names = resolve_cat_features(X_fit, self.cat_features)

        n_features = X_fit.shape[1]
        feature_names = list(X_fit.columns)

        self.n_features_ = n_features
        self.feature_names_ = feature_names

        hits = np.zeros(n_features, dtype=int)
        status = np.full(n_features, _TENTATIVE, dtype=int)

        importance_records: list[dict] = []
        decision_records: list[dict] = []
        no_tentative_streak = 0

        for iteration in range(1, self.max_iter + 1):
            artifacts = self._run_iteration(
                X=X_fit,
                y=y_fit,
                cat_names=cat_names,
                iteration=iteration,
                n_features=n_features,
            )

            self._record_importance_history(
                importance_records=importance_records,
                feature_names=feature_names,
                iteration=iteration,
                real_importances=artifacts.real_importances,
                shadow_max=artifacts.shadow_max,
                iteration_seed=artifacts.iteration_seed,
            )
            self._update_hits(
                hits=hits,
                status=status,
                real_importances=artifacts.real_importances,
                shadow_max=artifacts.shadow_max,
            )

            n_undecided = int((status == _TENTATIVE).sum())
            if n_undecided > 0:
                no_tentative_streak = 0
                self._update_decisions(
                    status=status,
                    hits=hits,
                    n_iter=iteration,
                    n_undecided=n_undecided,
                    decision_records=decision_records,
                    feature_names=feature_names,
                    iteration_seed=artifacts.iteration_seed,
                    shadow_max=artifacts.shadow_max,
                )
            else:
                no_tentative_streak += 1

            logger.info(
                "Iteration %d/%d: confirmed=%d, rejected=%d, tentative=%d, shadow_max=%.4f",
                iteration,
                self.max_iter,
                int((status == _CONFIRMED).sum()),
                int((status == _REJECTED).sum()),
                int((status == _TENTATIVE).sum()),
                artifacts.shadow_max,
            )

            if no_tentative_streak >= self.patience:
                logger.info(
                    "Early stopping: no tentative features for %d consecutive iterations.",
                    self.patience,
                )
                break

        self._n_iter_done = iteration
        self._finalize_fit_state(
            status=status,
            importance_records=importance_records,
            decision_records=decision_records,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return only confirmed features."""
        self._check_is_fitted()
        missing = [col for col in self.feature_names_ if col not in X.columns]
        if missing:
            raise ValueError(
                "Input DataFrame is missing columns seen during fit: "
                f"{missing}."
            )

        confirmed = [f for f, s in zip(self.feature_names_, self.support_) if s]
        return X[confirmed].copy()

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and return only confirmed features."""
        return self.fit(X, y).transform(X)

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Get a boolean mask or index array of confirmed features."""
        self._check_is_fitted()
        if indices:
            return np.where(self.support_)[0]
        return self.support_.copy()

    def get_feature_names_out(self) -> list[str]:
        """Return names of confirmed features."""
        self._check_is_fitted()
        return [f for f, s in zip(self.feature_names_, self.support_) if s]

    def get_selection_result(self) -> SelectionResult:
        """Return a standardized SelectionResult."""
        self._check_is_fitted()
        return SelectionResult(
            selected_features=self.get_feature_names_out(),
            rejected_features=[
                f for f, r in zip(self.feature_names_, self.ranking_) if r == 3
            ],
            tentative_features=[
                f for f, r in zip(self.feature_names_, self.ranking_) if r == 2
            ],
            metrics=self.importance_history_,
            config={
                "max_iter": self.max_iter,
                "alpha": self.alpha,
                "correction_method": self.correction_method,
                "importance_type": self.importance_type,
                "patience": self.patience,
                "task_type": self.task_type,
                "catboost_params": self._get_merged_params(),
                "n_iterations_done": self._n_iter_done,
            },
            random_state=self.random_state,
        )

    def _validate_init_params(
        self,
        *,
        max_iter: int,
        alpha: float,
        correction_method: str,
        patience: int,
        task_type: str,
    ) -> None:
        if correction_method not in ("bonferroni", "bh"):
            raise ValueError(
                "correction_method must be 'bonferroni' or 'bh', "
                f"got '{correction_method}'."
            )
        if task_type not in ("classification", "regression"):
            raise ValueError(
                f"task_type must be 'classification' or 'regression', got '{task_type}'."
            )
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}.")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}.")
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    def _prepare_fit_inputs(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        validate_dataframe(X, caller="BorutaCatBoost")
        validate_target(y, X, caller="BorutaCatBoost")

        if y.isnull().any():
            mask = y.notna()
            dropped = int((~mask).sum())
            warnings.warn(
                f"[BorutaCatBoost] Dropping {dropped} rows with null targets.",
                UserWarning,
                stacklevel=2,
            )
            X = X.loc[mask].copy()
            y = y.loc[mask].copy()

        if len(X) < 2:
            raise ValueError(
                "[BorutaCatBoost] Need at least 2 rows after dropping null targets."
            )

        return X, y

    def _run_iteration(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        cat_names: list[str],
        iteration: int,
        n_features: int,
    ) -> _IterationArtifacts:
        iteration_seed = (self.random_state or 0) + iteration
        rng = np.random.RandomState(iteration_seed)

        X_shadow = self._create_shadow_features(X, cat_names, rng)
        X_augmented = pd.concat([X, X_shadow], axis=1)

        aug_cat_names = list(cat_names) + [f"shadow_{c}" for c in cat_names]
        aug_cat_indices = get_cat_feature_indices(X_augmented, aug_cat_names)

        model = self._build_model(iteration_seed)
        pool = Pool(
            X_augmented,
            y,
            cat_features=aug_cat_indices if aug_cat_indices else None,
        )
        model.fit(pool)

        importances = model.get_feature_importance(pool, type=self.importance_type)
        real_importances = np.asarray(importances[:n_features], dtype=float)
        shadow_importances = np.asarray(importances[n_features:], dtype=float)

        if shadow_importances.size == 0:
            raise RuntimeError("Internal error: no shadow importances were produced.")

        shadow_max = float(np.nanmax(shadow_importances))
        return _IterationArtifacts(
            iteration_seed=iteration_seed,
            real_importances=real_importances,
            shadow_max=shadow_max,
        )

    def _create_shadow_features(
        self, X: pd.DataFrame, cat_names: list[str], rng: np.random.RandomState
    ) -> pd.DataFrame:
        """Create shuffled shadow features while preserving categorical dtypes."""
        X_shadow = X.apply(lambda col: rng.permutation(col.values), axis=0)
        X_shadow.columns = [f"shadow_{col}" for col in X.columns]
        X_shadow.index = X.index

        for col in cat_names:
            shadow_col = f"shadow_{col}"
            X_shadow[shadow_col] = X_shadow[shadow_col].astype(X[col].dtype)
        return X_shadow

    def _build_model(self, iteration_seed: int):
        params = self._get_merged_params()
        params["random_seed"] = iteration_seed
        if self.task_type == "classification":
            return CatBoostClassifier(**params)
        return CatBoostRegressor(**params)

    def _record_importance_history(
        self,
        *,
        importance_records: list[dict],
        feature_names: list[str],
        iteration: int,
        real_importances: np.ndarray,
        shadow_max: float,
        iteration_seed: int,
    ) -> None:
        for idx, feature_name in enumerate(feature_names):
            importance = float(real_importances[idx])
            importance_records.append(
                {
                    "iteration": iteration,
                    "iteration_seed": iteration_seed,
                    "feature": feature_name,
                    "importance": importance,
                    "shadow_max": shadow_max,
                    "hit": bool(importance > shadow_max),
                }
            )

    def _update_hits(
        self,
        *,
        hits: np.ndarray,
        status: np.ndarray,
        real_importances: np.ndarray,
        shadow_max: float,
    ) -> None:
        for idx, importance in enumerate(real_importances):
            if status[idx] == _TENTATIVE and importance > shadow_max:
                hits[idx] += 1

    def _update_decisions(
        self,
        *,
        status: np.ndarray,
        hits: np.ndarray,
        n_iter: int,
        n_undecided: int,
        decision_records: list[dict],
        feature_names: list[str],
        iteration_seed: int,
        shadow_max: float,
    ) -> None:
        """Update feature decisions using binomial test with correction."""
        tentative_indices = np.where(status == _TENTATIVE)[0]
        p_values_upper = np.ones(len(tentative_indices))
        p_values_lower = np.ones(len(tentative_indices))

        for j, idx in enumerate(tentative_indices):
            p_values_upper[j] = stats.binomtest(
                hits[idx], n_iter, 0.5, alternative="greater"
            ).pvalue
            p_values_lower[j] = stats.binomtest(
                hits[idx], n_iter, 0.5, alternative="less"
            ).pvalue

        if self.correction_method == "bonferroni":
            adj_upper = np.minimum(p_values_upper * n_undecided, 1.0)
            adj_lower = np.minimum(p_values_lower * n_undecided, 1.0)
        else:
            adj_upper = _benjamini_hochberg(p_values_upper)
            adj_lower = _benjamini_hochberg(p_values_lower)

        for j, idx in enumerate(tentative_indices):
            if adj_upper[j] < self.alpha:
                status[idx] = _CONFIRMED
            elif adj_lower[j] < self.alpha:
                status[idx] = _REJECTED

            decision_records.append(
                {
                    "iteration": n_iter,
                    "iteration_seed": iteration_seed,
                    "feature": feature_names[idx],
                    "shadow_max": shadow_max,
                    "hits": int(hits[idx]),
                    "p_upper": float(p_values_upper[j]),
                    "p_lower": float(p_values_lower[j]),
                    "adj_p_upper": float(adj_upper[j]),
                    "adj_p_lower": float(adj_lower[j]),
                    "status": _STATUS_TO_LABEL[int(status[idx])],
                }
            )

    def _finalize_fit_state(
        self,
        *,
        status: np.ndarray,
        importance_records: list[dict],
        decision_records: list[dict],
    ) -> None:
        self.support_ = status == _CONFIRMED
        self.support_weak_ = status == _TENTATIVE
        self.ranking_ = np.where(
            status == _CONFIRMED,
            1,
            np.where(status == _TENTATIVE, 2, 3),
        )
        self.importance_history_ = pd.DataFrame(importance_records)
        self.decision_log_ = (
            pd.DataFrame(decision_records) if decision_records else pd.DataFrame()
        )

    def _check_is_fitted(self) -> None:
        if self.support_ is None:
            raise RuntimeError("Call fit() before using this method.")

    def _get_merged_params(self) -> dict:
        params = deepcopy(_DEFAULT_CATBOOST_PARAMS)
        params.update(self.catboost_params)
        return params


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    if n == 0:
        return p_values

    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    adjusted = np.empty(n)

    adjusted[sorted_idx[-1]] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        rank = i + 1
        adjusted[sorted_idx[i]] = min(
            adjusted[sorted_idx[i + 1]],
            sorted_p[i] * n / rank,
        )

    return np.minimum(adjusted, 1.0)
