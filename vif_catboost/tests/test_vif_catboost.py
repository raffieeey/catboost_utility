"""Tests for CatBoostVIF."""

import warnings

import numpy as np
import pandas as pd
import pytest

from catboost_utility.vif_catboost import CatBoostVIF
from catboost_utility.common.result import SelectionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def numeric_df():
    """DataFrame with correlated numeric features."""
    rng = np.random.RandomState(42)
    n = 200
    x1 = rng.randn(n)
    x2 = x1 * 0.9 + rng.randn(n) * 0.1  # highly correlated with x1
    x3 = rng.randn(n)                     # independent
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})


@pytest.fixture
def mixed_df():
    """DataFrame with mixed numeric + categorical features."""
    rng = np.random.RandomState(42)
    n = 200
    x_num = rng.randn(n)
    x_cat = rng.choice(["a", "b", "c"], size=n)
    x_ind = rng.randn(n)
    return pd.DataFrame({"x_num": x_num, "x_cat": x_cat, "x_ind": x_ind})


@pytest.fixture
def all_categorical_df():
    """DataFrame with only categorical features."""
    rng = np.random.RandomState(42)
    n = 200
    c1 = rng.choice(["a", "b", "c"], size=n)
    c2 = rng.choice(["x", "y"], size=n)
    c3 = rng.choice(["p", "q", "r", "s"], size=n)
    return pd.DataFrame({"c1": c1, "c2": c2, "c3": c3})


@pytest.fixture
def fast_params():
    """Fast CatBoost params for testing."""
    return {"iterations": 20, "depth": 2, "learning_rate": 0.3}


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_rejects_invalid_threshold(self):
        with pytest.raises(ValueError, match="threshold"):
            CatBoostVIF(threshold=0)

    def test_rejects_invalid_cv_folds(self):
        with pytest.raises(ValueError, match="cv_folds"):
            CatBoostVIF(cv_folds=1)

    def test_rejects_invalid_holdout_fraction(self):
        with pytest.raises(ValueError, match="holdout_fraction"):
            CatBoostVIF(holdout_fraction=1.0)

    def test_rejects_invalid_n_jobs(self):
        with pytest.raises(ValueError, match="n_jobs"):
            CatBoostVIF(n_jobs=0)

    def test_rejects_non_dataframe(self):
        vif = CatBoostVIF()
        with pytest.raises(TypeError, match="pandas DataFrame"):
            vif.fit(np.array([[1, 2], [3, 4]]))

    def test_rejects_duplicate_columns(self):
        df = pd.DataFrame([[1, 2]], columns=["a", "a"])
        vif = CatBoostVIF()
        with pytest.raises(ValueError, match="duplicate column"):
            vif.fit(df)

    def test_rejects_datetime_columns(self):
        df = pd.DataFrame({"a": [1, 2], "b": pd.to_datetime(["2021-01-01", "2021-01-02"])})
        vif = CatBoostVIF()
        with pytest.raises(TypeError, match="Unsupported"):
            vif.fit(df)

    def test_warns_on_small_dataset(self, fast_params):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        vif = CatBoostVIF(catboost_params=fast_params)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                vif.fit(df)
            except Exception:
                pass
            assert any("unstable" in str(x.message).lower() for x in w)

    def test_warns_on_constant_column(self, fast_params):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "a": rng.randn(50),
            "b": rng.randn(50),
            "constant": 1.0,
        })
        vif = CatBoostVIF(catboost_params=fast_params)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            vif.fit(df)
            assert any("constant" in str(x.message).lower() for x in w)


# ---------------------------------------------------------------------------
# fit() tests
# ---------------------------------------------------------------------------

class TestFit:
    def test_numeric_returns_correct_columns(self, numeric_df, fast_params):
        vif = CatBoostVIF(catboost_params=fast_params)
        result = vif.fit(numeric_df)
        assert set(result.columns) == {"feature", "vif", "r_squared", "is_categorical", "clamped"}
        assert len(result) == 3

    def test_numeric_correlated_features_have_higher_vif(self, numeric_df, fast_params):
        vif = CatBoostVIF(catboost_params=fast_params)
        result = vif.fit(numeric_df)
        vif_x1 = result.loc[result["feature"] == "x1", "vif"].values[0]
        vif_x3 = result.loc[result["feature"] == "x3", "vif"].values[0]
        assert vif_x1 > vif_x3

    def test_mixed_dtype_runs(self, mixed_df, fast_params):
        vif = CatBoostVIF(catboost_params=fast_params)
        result = vif.fit(mixed_df)
        assert len(result) == 3
        cat_row = result.loc[result["feature"] == "x_cat"]
        assert cat_row["is_categorical"].values[0] == True

    def test_all_categorical(self, all_categorical_df, fast_params):
        vif = CatBoostVIF(catboost_params=fast_params)
        result = vif.fit(all_categorical_df)
        assert len(result) == 3
        assert all(result["is_categorical"])

    def test_vif_values_are_positive(self, numeric_df, fast_params):
        vif = CatBoostVIF(catboost_params=fast_params)
        result = vif.fit(numeric_df)
        assert (result["vif"] >= 1.0).all()

    def test_r_squared_in_valid_range(self, numeric_df, fast_params):
        vif = CatBoostVIF(catboost_params=fast_params)
        result = vif.fit(numeric_df)
        assert (result["r_squared"] >= 0.0).all()
        assert (result["r_squared"] < 1.0).all()

    def test_holdout_scoring_method(self, numeric_df, fast_params):
        vif = CatBoostVIF(scoring_method="holdout", catboost_params=fast_params)
        result = vif.fit(numeric_df)
        assert len(result) == 3

    def test_cat_features_by_name(self, mixed_df, fast_params):
        vif = CatBoostVIF(cat_features=["x_cat"], catboost_params=fast_params)
        result = vif.fit(mixed_df)
        cat_row = result.loc[result["feature"] == "x_cat"]
        assert cat_row["is_categorical"].values[0] == True
        num_row = result.loc[result["feature"] == "x_num"]
        assert num_row["is_categorical"].values[0] == False

    def test_cat_features_by_index(self, mixed_df, fast_params):
        vif = CatBoostVIF(cat_features=[1], catboost_params=fast_params)
        result = vif.fit(mixed_df)
        cat_row = result.loc[result["feature"] == "x_cat"]
        assert cat_row["is_categorical"].values[0] == True


# ---------------------------------------------------------------------------
# fit_eliminate() tests
# ---------------------------------------------------------------------------

class TestFitEliminate:
    def test_eliminates_correlated_feature(self, numeric_df, fast_params):
        vif = CatBoostVIF(threshold=3.0, catboost_params=fast_params)
        result = vif.fit_eliminate(numeric_df)
        assert isinstance(result, SelectionResult)
        assert len(result.rejected_features) > 0 or all(
            result.metrics["vif"] <= 3.0
        )

    def test_all_below_threshold_no_elimination(self, fast_params):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "a": rng.randn(200),
            "b": rng.randn(200),
            "c": rng.randn(200),
        })
        vif = CatBoostVIF(threshold=10.0, catboost_params=fast_params)
        result = vif.fit_eliminate(df)
        assert len(result.rejected_features) == 0
        assert len(result.selected_features) == 3

    def test_get_retained_features_after_eliminate(self, numeric_df, fast_params):
        vif = CatBoostVIF(threshold=3.0, catboost_params=fast_params)
        vif.fit_eliminate(numeric_df)
        retained = vif.get_retained_features()
        assert isinstance(retained, list)
        assert len(retained) > 0

    def test_get_retained_features_before_eliminate_raises(self):
        vif = CatBoostVIF()
        with pytest.raises(RuntimeError, match="fit_eliminate"):
            vif.get_retained_features()

    def test_elimination_config_has_history(self, numeric_df, fast_params):
        vif = CatBoostVIF(threshold=3.0, catboost_params=fast_params)
        result = vif.fit_eliminate(numeric_df)
        assert "elimination_history" in result.config

    def test_mixed_dtype_elimination(self, mixed_df, fast_params):
        vif = CatBoostVIF(threshold=5.0, catboost_params=fast_params)
        result = vif.fit_eliminate(mixed_df)
        assert isinstance(result, SelectionResult)


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_results(self, numeric_df, fast_params):
        vif1 = CatBoostVIF(random_state=123, catboost_params=fast_params)
        vif2 = CatBoostVIF(random_state=123, catboost_params=fast_params)
        result1 = vif1.fit(numeric_df)
        result2 = vif2.fit(numeric_df)
        pd.testing.assert_frame_equal(result1, result2)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_high_cardinality_categorical_skipped(self, fast_params):
        """Categorical target with too many classes should be skipped."""
        rng = np.random.RandomState(42)
        n = 200
        df = pd.DataFrame({
            "high_card": [f"cat_{i}" for i in range(n)],
            "x1": rng.randn(n),
            "x2": rng.randn(n),
        })
        vif = CatBoostVIF(max_target_cardinality=50, catboost_params=fast_params)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = vif.fit(df)
            high_card_row = result.loc[result["feature"] == "high_card"]
            assert np.isnan(high_card_row["vif"].values[0])

    def test_bool_column_treated_as_categorical(self, fast_params):
        rng = np.random.RandomState(42)
        n = 200
        df = pd.DataFrame({
            "flag": rng.choice([True, False], size=n),
            "x1": rng.randn(n),
            "x2": rng.randn(n),
        })
        vif = CatBoostVIF(catboost_params=fast_params)
        result = vif.fit(df)
        flag_row = result.loc[result["feature"] == "flag"]
        assert flag_row["is_categorical"].values[0] == True

    def test_missing_values_handled(self, fast_params):
        rng = np.random.RandomState(42)
        n = 200
        x1 = rng.randn(n)
        x2 = rng.randn(n)
        x1[rng.choice(n, 20, replace=False)] = np.nan
        df = pd.DataFrame({"x1": x1, "x2": x2, "x3": rng.randn(n)})
        vif = CatBoostVIF(catboost_params=fast_params)
        result = vif.fit(df)
        assert len(result) == 3

    def test_fit_eliminate_handles_all_nan_vif(self, fast_params):
        n = 120
        df = pd.DataFrame(
            {
                "high_card_1": [f"a_{i}" for i in range(n)],
                "high_card_2": [f"b_{i}" for i in range(n)],
            }
        )
        vif = CatBoostVIF(
            max_target_cardinality=10,
            catboost_params=fast_params,
        )
        result = vif.fit_eliminate(df)
        assert isinstance(result, SelectionResult)
        assert len(result.selected_features) == 2
