"""Tests for CatBoostRFE."""

import numpy as np
import pandas as pd
import pytest

from catboost_utility.rfe_catboost import CatBoostRFE
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
    x_noise3 = rng.randn(n)
    y = pd.Series(
        (x_informative + (x_cat_informative == "a").astype(float) > 0.5).astype(int)
    )
    X = pd.DataFrame({
        "x_informative": x_informative,
        "x_cat_informative": x_cat_informative,
        "x_noise1": x_noise1,
        "x_noise2": x_noise2,
        "x_noise3": x_noise3,
    })
    return X, y


@pytest.fixture
def regression_data():
    """Regression dataset with informative and noise features."""
    rng = np.random.RandomState(42)
    n = 300
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    noise1 = rng.randn(n)
    noise2 = rng.randn(n)
    y = pd.Series(x1 * 2 + x2 * 0.5 + rng.randn(n) * 0.1)
    X = pd.DataFrame({"x1": x1, "x2": x2, "noise1": noise1, "noise2": noise2})
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
        rfe = CatBoostRFE()
        with pytest.raises(TypeError, match="pandas DataFrame"):
            rfe.fit(np.array([[1, 2]]), pd.Series([0, 1]))

    def test_rejects_non_series_target(self, classification_data):
        X, y = classification_data
        rfe = CatBoostRFE()
        with pytest.raises(TypeError, match="pandas Series"):
            rfe.fit(X, [0, 1, 0])

    def test_rejects_length_mismatch(self, classification_data):
        X, y = classification_data
        rfe = CatBoostRFE()
        with pytest.raises(ValueError, match="Length mismatch"):
            rfe.fit(X, y.iloc[:10])

    def test_rejects_invalid_task_type(self):
        with pytest.raises(ValueError, match="task_type"):
            CatBoostRFE(task_type="invalid")

    def test_rejects_invalid_n_features_int(self):
        with pytest.raises(ValueError, match="n_features_to_select"):
            CatBoostRFE(n_features_to_select=0)

    def test_rejects_invalid_n_features_negative(self):
        with pytest.raises(ValueError, match="n_features_to_select"):
            CatBoostRFE(n_features_to_select=-3)

    def test_rejects_invalid_n_features_float(self):
        with pytest.raises(ValueError, match="n_features_to_select"):
            CatBoostRFE(n_features_to_select=1.5)

    def test_rejects_invalid_n_features_float_zero(self):
        with pytest.raises(ValueError, match="n_features_to_select"):
            CatBoostRFE(n_features_to_select=0.0)

    def test_rejects_invalid_step_int(self):
        with pytest.raises(ValueError, match="step"):
            CatBoostRFE(step=0)

    def test_rejects_invalid_step_float(self):
        with pytest.raises(ValueError, match="step"):
            CatBoostRFE(step=1.5)

    def test_rejects_invalid_step_float_zero(self):
        with pytest.raises(ValueError, match="step"):
            CatBoostRFE(step=0.0)

    def test_warns_on_small_dataset(self, fast_params):
        rng = np.random.RandomState(0)
        X = pd.DataFrame({"a": rng.randn(5), "b": rng.randn(5), "c": rng.randn(5)})
        y = pd.Series(rng.choice([0, 1], size=5))
        rfe = CatBoostRFE(
            n_features_to_select=1, catboost_params=fast_params, random_state=42
        )
        with pytest.warns(UserWarning, match="rows"):
            rfe.fit(X, y)

    def test_warns_on_constant_columns(self, fast_params):
        rng = np.random.RandomState(0)
        n = 100
        X = pd.DataFrame({
            "a": rng.randn(n),
            "const": np.ones(n),
            "b": rng.randn(n),
        })
        y = pd.Series(rng.choice([0, 1], size=n))
        rfe = CatBoostRFE(
            n_features_to_select=1, catboost_params=fast_params, random_state=42
        )
        with pytest.warns(UserWarning, match="Constant"):
            rfe.fit(X, y)


# ---------------------------------------------------------------------------
# fit() tests — classification
# ---------------------------------------------------------------------------

class TestFitClassification:
    def test_runs_and_returns_self(self, classification_data, fast_params):
        X, y = classification_data
        rfe = CatBoostRFE(
            n_features_to_select=2, catboost_params=fast_params, random_state=42
        )
        result = rfe.fit(X, y)
        assert result is rfe

    def test_fitted_attributes_set(self, classification_data, fast_params):
        X, y = classification_data
        rfe = CatBoostRFE(
            n_features_to_select=2, catboost_params=fast_params, random_state=42
        )
        rfe.fit(X, y)

        assert rfe.support_ is not None
        assert rfe.ranking_ is not None
        assert rfe.feature_names_ == list(X.columns)
        assert rfe.n_features_ == X.shape[1]
        assert len(rfe.support_) == X.shape[1]
        assert len(rfe.ranking_) == X.shape[1]

    def test_selected_count_matches_target(self, classification_data, fast_params):
        X, y = classification_data
        rfe = CatBoostRFE(
            n_features_to_select=2, catboost_params=fast_params, random_state=42
        )
        rfe.fit(X, y)
        assert rfe.support_.sum() == 2
        assert rfe.n_features_selected_ == 2

    def test_ranking_values(self, classification_data, fast_params):
        X, y = classification_data
        rfe = CatBoostRFE(
            n_features_to_select=2, catboost_params=fast_params, random_state=42
        )
        rfe.fit(X, y)
        # Selected features have rank 1.
        assert (rfe.ranking_[rfe.support_] == 1).all()
        # Eliminated features have rank > 1.
        assert (rfe.ranking_[~rfe.support_] > 1).all()
        # Ranks are positive integers.
        assert rfe.ranking_.min() == 1
        assert np.issubdtype(rfe.ranking_.dtype, np.integer)

    def test_importance_history_columns(self, classification_data, fast_params):
        X, y = classification_data
        rfe = CatBoostRFE(
            n_features_to_select=2, catboost_params=fast_params, random_state=42
        )
        rfe.fit(X, y)
        hist = rfe.importance_history_
        assert list(hist.columns) == [
            "iteration", "feature", "importance", "rank", "selected"
        ]
        assert len(hist) > 0
        # First iteration records all features.
        first = hist[hist["iteration"] == 1]
        assert len(first) == X.shape[1]

    def test_selected_count_positive_with_informative(
        self, classification_data, fast_params
    ):
        X, y = classification_data
        rfe = CatBoostRFE(
            n_features_to_select=2, catboost_params=fast_params, random_state=42
        )
        rfe.fit(X, y)
        assert rfe.support_.sum() > 0


# ---------------------------------------------------------------------------
# fit() tests — regression
# ---------------------------------------------------------------------------

class TestFitRegression:
    def test_regression_runs(self, regression_data, fast_params):
        X, y = regression_data
        rfe = CatBoostRFE(
            n_features_to_select=2, task_type="regression",
            catboost_params=fast_params, random_state=42,
        )
        rfe.fit(X, y)
        assert rfe.support_ is not None
        assert rfe.support_.sum() == 2

    def test_regression_valid_rankings(self, regression_data, fast_params):
        X, y = regression_data
        rfe = CatBoostRFE(
            n_features_to_select=2, task_type="regression",
            catboost_params=fast_params, random_state=42,
        )
        rfe.fit(X, y)
        assert (rfe.ranking_[rfe.support_] == 1).all()
        assert set(rfe.ranking_).issuperset({1})
        # Informative features should be retained over pure noise.
        selected = set(rfe.get_feature_names_out())
        assert "x1" in selected


# ---------------------------------------------------------------------------
# transform() tests
# ---------------------------------------------------------------------------

class TestTransform:
    def test_transform_returns_subset(self, classification_data, fast_params):
        X, y = classification_data
        rfe = CatBoostRFE(
            n_features_to_select=2, catboost_params=fast_params, random_state=42
        )
        rfe.fit(X, y)
        X_t = rfe.transform(X)
        assert X_t.shape[1] == 2
        assert list(X_t.columns) == rfe.get_feature_names_out()
        assert set(X_t.columns).issubset(set(X.columns))

    def test_transform_before_fit_raises(self):
        rfe = CatBoostRFE()
        with pytest.raises(RuntimeError, match="fit"):
            rfe.transform(pd.DataFrame({"a": [1]}))

    def test_fit_transform(self, classification_data, fast_params):
        X, y = classification_data
        rfe = CatBoostRFE(
            n_features_to_select=3, catboost_params=fast_params, random_state=42
        )
        result = rfe.fit_transform(X, y)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 3

    def test_transform_missing_columns_raises(self, classification_data, fast_params):
        X, y = classification_data
        rfe = CatBoostRFE(
            n_features_to_select=2, catboost_params=fast_params, random_state=42
        )
        rfe.fit(X, y)
        with pytest.raises(ValueError, match="missing columns"):
            rfe.transform(X.drop(columns=[X.columns[0]]))


# ---------------------------------------------------------------------------
# sklearn compat tests
# ---------------------------------------------------------------------------

class TestSklearnCompat:
    def test_get_support_mask(self, classification_data, fast_params):
        X, y = classification_data
        rfe = CatBoostRFE(
            n_features_to_select=2, catboost_params=fast_params, random_state=42
        )
        rfe.fit(X, y)
        mask = rfe.get_support()
        assert mask.dtype == bool
        assert len(mask) == X.shape[1]

    def test_get_support_indices(self, classification_data, fast_params):
        X, y = classification_data
        rfe = CatBoostRFE(
            n_features_to_select=2, catboost_params=fast_params, random_state=42
        )
        rfe.fit(X, y)
        indices = rfe.get_support(indices=True)
        assert indices.dtype in (np.int64, np.int32, np.intp)
        assert len(indices) == 2

    def test_get_support_before_fit_raises(self):
        rfe = CatBoostRFE()
        with pytest.raises(RuntimeError, match="fit"):
            rfe.get_support()

    def test_get_feature_names_out(self, classification_data, fast_params):
        X, y = classification_data
        rfe = CatBoostRFE(
            n_features_to_select=2, catboost_params=fast_params, random_state=42
        )
        rfe.fit(X, y)
        names = rfe.get_feature_names_out()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)
        assert len(names) == 2


# ---------------------------------------------------------------------------
# get_selection_result() tests
# ---------------------------------------------------------------------------

class TestSelectionResult:
    def test_returns_selection_result(self, classification_data, fast_params):
        X, y = classification_data
        rfe = CatBoostRFE(
            n_features_to_select=2, catboost_params=fast_params, random_state=42
        )
        rfe.fit(X, y)
        sr = rfe.get_selection_result()
        assert isinstance(sr, SelectionResult)
        assert len(sr.selected_features) == 2
        total = (
            len(sr.selected_features)
            + len(sr.rejected_features)
            + len(sr.tentative_features)
        )
        assert total == X.shape[1]
        assert sr.tentative_features == []

    def test_config_contains_params(self, classification_data, fast_params):
        X, y = classification_data
        rfe = CatBoostRFE(
            n_features_to_select=2, step=1, catboost_params=fast_params,
            random_state=42,
        )
        rfe.fit(X, y)
        cfg = rfe.get_selection_result().config
        for key in (
            "n_features_to_select", "n_features_selected", "step",
            "importance_type", "task_type", "catboost_params",
        ):
            assert key in cfg
        assert cfg["catboost_params"]["iterations"] == 20


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_results(self, classification_data, fast_params):
        X, y = classification_data
        r1 = CatBoostRFE(
            n_features_to_select=2, catboost_params=fast_params, random_state=42
        )
        r2 = CatBoostRFE(
            n_features_to_select=2, catboost_params=fast_params, random_state=42
        )
        r1.fit(X, y)
        r2.fit(X, y)
        np.testing.assert_array_equal(r1.support_, r2.support_)
        np.testing.assert_array_equal(r1.ranking_, r2.ranking_)
        pd.testing.assert_frame_equal(r1.importance_history_, r2.importance_history_)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_n_features_as_float(self, classification_data, fast_params):
        X, y = classification_data  # 5 features
        rfe = CatBoostRFE(
            n_features_to_select=0.4, catboost_params=fast_params, random_state=42
        )
        rfe.fit(X, y)
        # round(0.4 * 5) = 2
        assert rfe.support_.sum() == 2

    def test_step_as_float(self, classification_data, fast_params):
        X, y = classification_data
        rfe = CatBoostRFE(
            n_features_to_select=1, step=0.5,
            catboost_params=fast_params, random_state=42,
        )
        rfe.fit(X, y)
        assert rfe.support_.sum() == 1
        # Multiple features can be removed per iteration.
        n_iters = rfe.importance_history_["iteration"].nunique()
        assert n_iters >= 1

    def test_all_categorical_features(self, fast_params):
        rng = np.random.RandomState(42)
        n = 200
        X = pd.DataFrame({
            "c1": rng.choice(["a", "b", "c"], size=n),
            "c2": rng.choice(["x", "y"], size=n),
            "c3": rng.choice(["p", "q", "r"], size=n),
            "c4": rng.choice(["m", "n"], size=n),
        })
        y = pd.Series(rng.choice([0, 1], size=n))
        rfe = CatBoostRFE(
            n_features_to_select=2, catboost_params=fast_params, random_state=42
        )
        rfe.fit(X, y)
        assert rfe.support_.sum() == 2
        assert len(rfe.support_) == 4

    def test_missing_values(self, fast_params):
        rng = np.random.RandomState(42)
        n = 200
        x1 = rng.randn(n)
        x1[rng.choice(n, 20, replace=False)] = np.nan
        X = pd.DataFrame({"x1": x1, "x2": rng.randn(n), "x3": rng.randn(n)})
        y = pd.Series(rng.choice([0, 1], size=n))
        rfe = CatBoostRFE(
            n_features_to_select=1, catboost_params=fast_params, random_state=42
        )
        rfe.fit(X, y)
        assert rfe.support_.sum() == 1

    def test_bool_features(self, fast_params):
        rng = np.random.RandomState(42)
        n = 200
        X = pd.DataFrame({
            "flag": rng.choice([True, False], size=n),
            "x1": rng.randn(n),
            "x2": rng.randn(n),
        })
        y = pd.Series(rng.choice([0, 1], size=n))
        rfe = CatBoostRFE(
            n_features_to_select=1, catboost_params=fast_params, random_state=42
        )
        rfe.fit(X, y)
        assert rfe.support_.sum() == 1

    def test_multiclass_target(self, fast_params):
        rng = np.random.RandomState(42)
        n = 300
        X = pd.DataFrame({
            "x1": rng.randn(n), "x2": rng.randn(n),
            "x3": rng.randn(n), "x4": rng.randn(n),
        })
        y = pd.Series(rng.choice([0, 1, 2], size=n))
        rfe = CatBoostRFE(
            n_features_to_select=2, catboost_params=fast_params, random_state=42
        )
        rfe.fit(X, y)
        assert rfe.support_.sum() == 2

    def test_null_targets_are_dropped(self, fast_params):
        rng = np.random.RandomState(42)
        n = 200
        X = pd.DataFrame({"x1": rng.randn(n), "x2": rng.randn(n), "x3": rng.randn(n)})
        y = pd.Series(rng.choice([0, 1], size=n).astype(float))
        y.iloc[0:5] = np.nan
        rfe = CatBoostRFE(
            n_features_to_select=1, catboost_params=fast_params, random_state=42
        )
        with pytest.warns(UserWarning, match="null targets"):
            rfe.fit(X, y)
        assert rfe.support_.sum() == 1

    def test_n_features_greater_than_total_returns_all(
        self, classification_data, fast_params
    ):
        X, y = classification_data  # 5 features
        rfe = CatBoostRFE(
            n_features_to_select=99, catboost_params=fast_params, random_state=42
        )
        rfe.fit(X, y)
        assert rfe.support_.all()
        assert (rfe.ranking_ == 1).all()
        # Still records an importance history.
        assert len(rfe.importance_history_) == X.shape[1]

    def test_n_features_none_defaults_to_half(self, classification_data, fast_params):
        X, y = classification_data  # 5 features
        rfe = CatBoostRFE(catboost_params=fast_params, random_state=42)
        rfe.fit(X, y)
        assert rfe.support_.sum() == max(1, X.shape[1] // 2)

    def test_single_feature_dataset(self, fast_params):
        rng = np.random.RandomState(42)
        n = 100
        X = pd.DataFrame({"only": rng.randn(n)})
        y = pd.Series((X["only"] > 0).astype(int))
        rfe = CatBoostRFE(catboost_params=fast_params, random_state=42)
        rfe.fit(X, y)
        assert rfe.support_.sum() == 1
        assert rfe.ranking_[0] == 1
        assert list(rfe.transform(X).columns) == ["only"]
