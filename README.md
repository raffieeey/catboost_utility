# catboost-utility

CatBoost-based feature selection for mixed-type tabular data.

> **Note:** This package is not part of the official CatBoost project. It uses CatBoost as a modeling backend for feature-selection methods on mixed numeric + categorical data.

## Why

Most feature-selection tools assume all columns are numeric. Real data has categorical fields. Forcing one-hot or ordinal encoding before selection distorts relationships, inflates dimensionality, and makes pipelines harder to audit.

CatBoost handles categoricals natively. This library uses CatBoost as the engine for three selection methods — VIF, Boruta, and RFE — so you never encode before selecting.

## Installation

```bash
pip install catboost-utility
```

For development:

```bash
pip install -e .
```

Requires Python 3.9+. Core deps: catboost, pandas, numpy, scipy, joblib, scikit-learn.

## Quick start

```python
import pandas as pd
from catboost_utility.boruta_catboost import BorutaCatBoost
from catboost_utility.vif_catboost import CatBoostVIF
from catboost_utility.rfe_catboost import CatBoostRFE

X = pd.DataFrame({
    "age": [34, 27, 52, 41, 29, 38],
    "income": [70_000, 42_000, 110_000, 87_000, 50_000, 76_000],
    "city": ["A", "B", "A", "C", "B", "A"],
    "segment": ["retail", "retail", "enterprise", "enterprise", "retail", "enterprise"],
})
y = pd.Series([0, 0, 1, 1, 0, 1], name="target")

# 1) Boruta — remove irrelevant features
boruta = BorutaCatBoost(cat_features=["city", "segment"], max_iter=30, random_state=42)
boruta.fit(X, y)
X_sel = boruta.transform(X)

# 2) VIF — remove redundant features
vif = CatBoostVIF(cat_features=["city", "segment"], threshold=5.0, random_state=42)
vif_result = vif.fit_eliminate(X_sel)
X_sel = X_sel[vif_result.selected_features]

# 3) RFE — select top-N features
rfe = CatBoostRFE(n_features_to_select=2, cat_features=["city", "segment"], random_state=42)
rfe.fit(X_sel, y)
X_final = rfe.transform(X_sel)

print("Final features:", list(X_final.columns))
```

## Modules

### `BorutaCatBoost` — all-relevant selection

Creates shuffled shadow copies of all features, trains CatBoost on originals + shadows, and uses binomial tests to decide which features are genuinely better than random.

```python
boruta = BorutaCatBoost(
    cat_features=["city"],
    max_iter=100,
    alpha=0.05,
    correction_method="bonferroni",  # or "bh"
    task_type="classification",
    random_state=42,
)
boruta.fit(X, y)
print(boruta.get_feature_names_out())   # confirmed features
print(boruta.decision_log_.head())      # per-iteration decisions
```

**Decision rules:** Each feature gets a status — `confirmed` (beats shadows often enough), `rejected` (worse than noise), or `tentative` (undecided). P-values are corrected for multiple testing via Bonferroni (FWER) or Benjamini-Hochberg (FDR). Early stopping when no tentative features remain for `patience` iterations.

**Fitted attributes:** `support_`, `support_weak_`, `ranking_`, `importance_history_`, `decision_log_`.

### `CatBoostVIF` — collinearity analysis

Predicts each feature from all others using CatBoost and converts prediction quality into a VIF score. Numeric targets use regression R²; categorical targets use McFadden pseudo-R² from log-loss.

```python
vif = CatBoostVIF(cat_features=["city"], threshold=5.0, scoring_method="oof", random_state=42)

# Get VIF table
vif_table = vif.fit(X)

# Iteratively drop highest-VIF features until all below threshold
result = vif.fit_eliminate(X)
print(result.selected_features, result.rejected_features)
```

**Key params:** `threshold` (default 5.0), `scoring_method` (`"oof"` or `"holdout"`), `cv_folds`, `n_jobs`.

**Output:** `fit()` returns a DataFrame with columns `feature`, `vif`, `r_squared`, `is_categorical`, `clamped`. `fit_eliminate()` returns a `SelectionResult` with elimination history in `config`.

### `CatBoostRFE` — recursive feature elimination

Repeatedly trains CatBoost, ranks features by importance, removes the least important, and continues until `n_features_to_select` remain.

```python
rfe = CatBoostRFE(
    n_features_to_select=2,     # int, float (0-1), or None (defaults to n_features // 2)
    step=1,                      # int or float (0-1)
    cat_features=["city"],
    task_type="classification",
    random_state=42,
)
rfe.fit(X, y)
print(rfe.get_feature_names_out())
print(rfe.ranking_)              # 1 = selected, higher = eliminated earlier
print(rfe.importance_history_)    # per-iteration rankings
```

**Edge cases:** If `n_features_to_select >= total features`, all features return with rank 1. Float `step` removes that fraction of remaining features per iteration. Null targets are dropped with a warning.

## Practical workflow

1. **Boruta** — remove clearly irrelevant features (target-aware)
2. **VIF** — reduce redundancy among retained features (unsupervised)
3. **RFE** — select top-N features for your final model (target-aware)
4. Train your final model on the reduced set

## Testing

```bash
python -m pytest -v --tb=short
```

## Roadmap

- Permutation importance utilities
- Stability selection
- Calibration and threshold optimization
- Interaction discovery
- OOF meta-feature generation
