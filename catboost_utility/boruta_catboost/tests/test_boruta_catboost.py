"""Tests for BorutaCatBoost."""

import warnings

import numpy as np
import pandas as pd
import pytest

from catboost_utility.boruta_catboost import BorutaCatBoost
from catboost_utility.common.result import SelectionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def classification_data():
    """Binary classification dataset with informative and noise features."""
    rng = np.random.RandomState(42)
    n = 300
    x_informative = rng.randn(n)
    x_cat_informative = rng.choice(["a", "b"], size=n)
    x_noise1 = rng.randn(n)
    x_noise2 = rng.choice(["p", "q", "r"], size=n)
    y = pd.Series((x_informative + (x_cat_informative == "a").astype(float) > 0.5).astype(int))
    X = pd.DataFrame({
        "x_informative": x_informative,
        "x_cat_informative": x_cat_informative,
        "x_noise1": x_noise1,
        "x_noise2": x_noise2,
    })
    return X, y


@pytest.fixture
def regression_data():
    """Regression dataset with informative and noise features."""
    rng = np.random.RandomState(42)
    n = 300
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    noise = rng.randn(n)
    y = pd.Series(x1 * 2 + x2 * 0.5 + rng.randn(n) * 0.1)
    X = pd.DataFrame({"x1": x1, "x2": x2, "noise": noise})
    return X, y


@pytest.fixture
def fast_params():
    """Fast CatBoost params for testing."""
    return {"iterations": 20, "depth": 2, "learning_rate": 0.3}


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_rejects_non_dataframe(self):
        boruta = BorutaCatBoost()
        with pytest.raises(TypeError, match="pandas DataFrame"):
            boruta.fit(np.array([[1, 2]]), pd.Series([0]))

    def test_rejects_non_series_target(self, classification_data):
        X, y = classification_data
        boruta = BorutaCatBoost()
        with pytest.raises(TypeError, match="pandas Series"):
            boruta.fit(X, [0, 1, 0])

    def test_rejects_length_mismatch(self, classification_data):
        X, y = classification_data
        boruta = BorutaCatBoost()
        with pytest.raises(ValueError, match="Length mismatch"):
            boruta.fit(X, y.iloc[:10])

    def test_rejects_invalid_correction_method(self):
        with pytest.raises(ValueError, match="correction_method"):
            BorutaCatBoost(correction_method="invalid")

    def test_rejects_invalid_task_type(self):
        with pytest.raises(ValueError, match="task_type"):
            BorutaCatBoost(task_type="invalid")

    def test_rejects_invalid_max_iter(self):
        with pytest.raises(ValueError, match="max_iter"):
            BorutaCatBoost(max_iter=0)

    def test_rejects_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            BorutaCatBoost(alpha=1.2)

    def test_rejects_invalid_patience(self):
        with pytest.raises(ValueError, match="patience"):
            BorutaCatBoost(patience=0)


# ---------------------------------------------------------------------------
# fit() tests — classification
# ---------------------------------------------------------------------------

class TestFitClassification:
    def test_runs_and_returns_self(self, classification_data, fast_params):
        X, y = classification_data
        boruta = BorutaCatBoost(
            max_iter=5, catboost_params=fast_params, random_state=42
        )
        result = boruta.fit(X, y)
        assert result is boruta

    def test_fitted_attributes_set(self, classification_data, fast_params):
        X, y = classification_data
        boruta = BorutaCatBoost(
            max_iter=5, catboost_params=fast_params, random_state=42
        )
        boruta.fit(X, y)

        assert boruta.support_ is not None
        assert boruta.support_weak_ is not None
        assert boruta.ranking_ is not None
        assert boruta.importance_history_ is not None
        assert len(boruta.support_) == X.shape[1]
        assert len(boruta.ranking_) == X.shape[1]

    def test_ranking_values(self, classification_data, fast_params):
        X, y = classification_data
        boruta = BorutaCatBoost(
            max_iter=5, catboost_params=fast_params, random_state=42
        )
        boruta.fit(X, y)
        assert set(boruta.ranking_).issubset({1, 2, 3})

    def test_importance_history_structure(self, classification_data, fast_params):
        X, y = classification_data
        boruta = BorutaCatBoost(
            max_iter=5, catboost_params=fast_params, random_state=42
        )
        boruta.fit(X, y)
        hist = boruta.importance_history_
        assert "iteration" in hist.columns
        assert "feature" in hist.columns
        assert "importance" in hist.columns
        assert "shadow_max" in hist.columns
        assert "hit" in hist.columns
        assert len(hist) == 5 * X.shape[1]  # max_iter * n_features


# ---------------------------------------------------------------------------
# fit() tests — regression
# ---------------------------------------------------------------------------

class TestFitRegression:
    def test_regression_runs(self, regression_data, fast_params):
        X, y = regression_data
        boruta = BorutaCatBoost(
            task_type="regression", max_iter=5,
            catboost_params=fast_params, random_state=42,
        )
        boruta.fit(X, y)
        assert boruta.support_ is not None

    def test_regression_informative_features(self, regression_data, fast_params):
        """With enough iterations, informative features should have more hits."""
        X, y = regression_data
        boruta = BorutaCatBoost(
            task_type="regression", max_iter=20,
            catboost_params=fast_params, random_state=42,
        )
        boruta.fit(X, y)
        hist = boruta.importance_history_
        x1_hits = hist.loc[hist["feature"] == "x1", "hit"].sum()
        noise_hits = hist.loc[hist["feature"] == "noise", "hit"].sum()
        assert x1_hits > noise_hits


# ---------------------------------------------------------------------------
# transform() tests
# ---------------------------------------------------------------------------

class TestTransform:
    def test_transform_returns_subset(self, classification_data, fast_params):
        X, y = classification_data
        boruta = BorutaCatBoost(
            max_iter=5, catboost_params=fast_params, random_state=42
        )
        boruta.fit(X, y)
        X_transformed = boruta.transform(X)
        assert X_transformed.shape[1] <= X.shape[1]
        # All columns in transformed should be confirmed
        confirmed = boruta.get_feature_names_out()
        assert list(X_transformed.columns) == confirmed

    def test_transform_before_fit_raises(self):
        boruta = BorutaCatBoost()
        with pytest.raises(RuntimeError, match="fit"):
            boruta.transform(pd.DataFrame({"a": [1]}))

    def test_fit_transform(self, classification_data, fast_params):
        X, y = classification_data
        boruta = BorutaCatBoost(
            max_iter=5, catboost_params=fast_params, random_state=42
        )
        result = boruta.fit_transform(X, y)
        assert isinstance(result, pd.DataFrame)

    def test_transform_missing_original_columns_raises(self, classification_data, fast_params):
        X, y = classification_data
        boruta = BorutaCatBoost(
            max_iter=5, catboost_params=fast_params, random_state=42
        )
        boruta.fit(X, y)
        with pytest.raises(ValueError, match="missing columns"):
            boruta.transform(X.drop(columns=[X.columns[0]]))


# ---------------------------------------------------------------------------
# get_support / get_feature_names_out tests
# ---------------------------------------------------------------------------

class TestSklearnCompat:
    def test_get_support_mask(self, classification_data, fast_params):
        X, y = classification_data
        boruta = BorutaCatBoost(
            max_iter=5, catboost_params=fast_params, random_state=42
        )
        boruta.fit(X, y)
        mask = boruta.get_support()
        assert mask.dtype == bool
        assert len(mask) == X.shape[1]

    def test_get_support_indices(self, classification_data, fast_params):
        X, y = classification_data
        boruta = BorutaCatBoost(
            max_iter=5, catboost_params=fast_params, random_state=42
        )
        boruta.fit(X, y)
        indices = boruta.get_support(indices=True)
        assert indices.dtype in (np.int64, np.int32, np.intp)

    def test_get_support_before_fit_raises(self):
        boruta = BorutaCatBoost()
        with pytest.raises(RuntimeError, match="fit"):
            boruta.get_support()

    def test_get_feature_names_out(self, classification_data, fast_params):
        X, y = classification_data
        boruta = BorutaCatBoost(
            max_iter=5, catboost_params=fast_params, random_state=42
        )
        boruta.fit(X, y)
        names = boruta.get_feature_names_out()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)


# ---------------------------------------------------------------------------
# get_selection_result() test
# ---------------------------------------------------------------------------

class TestSelectionResult:
    def test_returns_selection_result(self, classification_data, fast_params):
        X, y = classification_data
        boruta = BorutaCatBoost(
            max_iter=5, catboost_params=fast_params, random_state=42
        )
        boruta.fit(X, y)
        sr = boruta.get_selection_result()
        assert isinstance(sr, SelectionResult)
        total = (
            len(sr.selected_features)
            + len(sr.rejected_features)
            + len(sr.tentative_features)
        )
        assert total == X.shape[1]

    def test_decision_log_has_iteration_seed_and_shadow_max(self, classification_data, fast_params):
        X, y = classification_data
        boruta = BorutaCatBoost(
            max_iter=5, catboost_params=fast_params, random_state=42
        )
        boruta.fit(X, y)
        if not boruta.decision_log_.empty:
            assert "iteration_seed" in boruta.decision_log_.columns
            assert "shadow_max" in boruta.decision_log_.columns


# ---------------------------------------------------------------------------
# Multiple-testing correction tests
# ---------------------------------------------------------------------------

class TestCorrectionMethods:
    def test_bonferroni(self, classification_data, fast_params):
        X, y = classification_data
        boruta = BorutaCatBoost(
            max_iter=5, correction_method="bonferroni",
            catboost_params=fast_params, random_state=42,
        )
        boruta.fit(X, y)
        assert boruta.support_ is not None

    def test_bh(self, classification_data, fast_params):
        X, y = classification_data
        boruta = BorutaCatBoost(
            max_iter=5, correction_method="bh",
            catboost_params=fast_params, random_state=42,
        )
        boruta.fit(X, y)
        assert boruta.support_ is not None


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_results(self, classification_data, fast_params):
        X, y = classification_data
        b1 = BorutaCatBoost(
            max_iter=5, catboost_params=fast_params, random_state=42
        )
        b2 = BorutaCatBoost(
            max_iter=5, catboost_params=fast_params, random_state=42
        )
        b1.fit(X, y)
        b2.fit(X, y)
        np.testing.assert_array_equal(b1.support_, b2.support_)
        np.testing.assert_array_equal(b1.ranking_, b2.ranking_)
        pd.testing.assert_frame_equal(b1.importance_history_, b2.importance_history_)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_categorical_features(self, fast_params):
        rng = np.random.RandomState(42)
        n = 200
        X = pd.DataFrame({
            "c1": rng.choice(["a", "b", "c"], size=n),
            "c2": rng.choice(["x", "y"], size=n),
            "c3": rng.choice(["p", "q", "r"], size=n),
        })
        y = pd.Series(rng.choice([0, 1], size=n))
        boruta = BorutaCatBoost(
            max_iter=5, catboost_params=fast_params, random_state=42
        )
        boruta.fit(X, y)
        assert boruta.support_ is not None
        assert len(boruta.support_) == 3

    def test_missing_values(self, fast_params):
        rng = np.random.RandomState(42)
        n = 200
        x1 = rng.randn(n)
        x1[rng.choice(n, 20, replace=False)] = np.nan
        X = pd.DataFrame({"x1": x1, "x2": rng.randn(n)})
        y = pd.Series(rng.choice([0, 1], size=n))
        boruta = BorutaCatBoost(
            max_iter=3, catboost_params=fast_params, random_state=42
        )
        boruta.fit(X, y)
        assert boruta.support_ is not None

    def test_bool_features(self, fast_params):
        rng = np.random.RandomState(42)
        n = 200
        X = pd.DataFrame({
            "flag": rng.choice([True, False], size=n),
            "x1": rng.randn(n),
        })
        y = pd.Series(rng.choice([0, 1], size=n))
        boruta = BorutaCatBoost(
            max_iter=3, catboost_params=fast_params, random_state=42
        )
        boruta.fit(X, y)
        assert boruta.support_ is not None

    def test_multiclass_target(self, fast_params):
        rng = np.random.RandomState(42)
        n = 300
        X = pd.DataFrame({"x1": rng.randn(n), "x2": rng.randn(n), "x3": rng.randn(n)})
        y = pd.Series(rng.choice([0, 1, 2], size=n))
        boruta = BorutaCatBoost(
            max_iter=3, catboost_params=fast_params, random_state=42
        )
        boruta.fit(X, y)
        assert boruta.support_ is not None

    def test_null_targets_are_dropped(self, fast_params):
        rng = np.random.RandomState(42)
        n = 200
        X = pd.DataFrame({"x1": rng.randn(n), "x2": rng.randn(n)})
        y = pd.Series(rng.choice([0, 1], size=n).astype(float))
        y.iloc[0:5] = np.nan
        boruta = BorutaCatBoost(
            max_iter=3, catboost_params=fast_params, random_state=42
        )
        boruta.fit(X, y)
        assert boruta.support_ is not None


# ---------------------------------------------------------------------------
# Early stopping test
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def test_early_stopping_with_patience(self, fast_params):
        """If all features are decided, should stop before max_iter."""
        rng = np.random.RandomState(42)
        n = 300
        x1 = rng.randn(n)
        y = pd.Series((x1 > 0).astype(int))
        X = pd.DataFrame({"x1": x1, "noise": rng.randn(n) * 0.001})
        boruta = BorutaCatBoost(
            max_iter=100, patience=3,
            catboost_params=fast_params, random_state=42,
        )
        boruta.fit(X, y)
        # Should have stopped before 100 iterations (once decisions are made)
        n_iters = boruta.importance_history_["iteration"].nunique()
        assert n_iters < 100
