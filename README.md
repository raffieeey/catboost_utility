# catboost-utility

CatBoost-based feature selection utilities for mixed-type tabular data.

> **Note:** This package is not part of the official CatBoost project. It uses CatBoost as a modeling backend / scientific utility for feature-selection methods on mixed-type tabular data.

This project focuses on one core problem: most classical feature-selection tools assume all columns are numeric, while many real datasets have important categorical fields. CatBoost can model categorical predictors directly, so this library uses CatBoost as the engine for:

- VIF-style multicollinearity analysis that works on mixed numeric + categorical data.
- Boruta-style all-relevant feature selection with statistical decision rules.

Current release: `v0.1.0`

Implemented modules:

- `CatBoostVIF`
- `BorutaCatBoost`

---

## Why this exists

Traditional VIF and Boruta implementations commonly rely on linear models or sklearn random forests. In mixed-type data, that usually forces manual encoding before selection, which can:

- distort relationships (for example, one-hot expansion changes geometry),
- inject arbitrary ordinality (label encoding),
- make interpretation harder.

`catboost-utility` keeps categorical handling native through CatBoost and adds reproducibility, input validation, and audit-friendly outputs.

---

## Why use this instead of a standard sklearn-only approach?

Short answer: use this when your feature selection must work well on mixed numeric + categorical data without forcing manual encoding first.

### What existing sklearn workflows usually do

- For VIF-like analysis, people often use linear-model tooling (or statsmodels VIF), which assumes numeric inputs.
- For Boruta-style selection, common implementations are tree-based wrappers around sklearn estimators, which also usually expect encoded categoricals.
- So teams typically add preprocessing (one-hot/ordinal/target encoding) before feature selection.

### Why that can be a problem

- Encoding can change the geometry of the problem:
  - one-hot can inflate dimensionality and split one concept into many sparse columns,
  - ordinal encoding can inject fake ordering.
- Feature-importance and collinearity signals can become encoding-dependent.
- Pipelines become harder to audit because preprocessing and selection interact.

### What this library changes technically

1. Native categorical handling via CatBoost:
- No forced one-hot/ordinal conversion before selection.
- Same modeling backend for numeric and categorical predictors.

2. VIF adapted for mixed data:
- Numeric targets use regression `R^2`.
- Categorical targets use McFadden-like pseudo-`R^2` from log-loss.
- Out-of-fold/holdout scoring avoids overly optimistic in-sample scores.

3. Boruta with explicit statistical controls:
- Shadow-feature comparisons are done each iteration.
- Multiple-testing correction is built in (`bonferroni` or `bh`).
- Decision logs, seeds, and histories are retained for auditability.

4. Reproducibility and robustness defaults:
- Centralized validation, deterministic seeding, edge-case warnings, and standardized result artifacts.

### When sklearn-only is still the right choice

Use standard sklearn tooling if:

- your dataset is already fully numeric and cleanly preprocessed,
- you need a lightweight baseline quickly,
- you do not need categorical-native selection behavior,
- or you prioritize ecosystem familiarity over mixed-type rigor.

In short:

- If your selection quality depends on handling raw categorical columns correctly, this library is usually the better fit.
- If your problem is purely numeric and simple, sklearn-only workflows may be sufficient and faster to operationalize.

---

## Installation

### Install from PyPI

```bash
pip install catboost-utility
```

### Local editable install (development)

Use this if you are developing the package locally:

```bash
pip install -e .
```

### Dependencies

Core dependencies:

- `catboost`
- `pandas`
- `numpy`
- `scipy`
- `joblib`
- `scikit-learn`

Optional:

- `tqdm` (progress utilities planned for future expansion)

---

## Quick Start

```python
import pandas as pd
from catboost_utility.vif_catboost import CatBoostVIF
from catboost_utility.boruta_catboost import BorutaCatBoost

# Example mixed-type data
X = pd.DataFrame(
    {
        "age": [34, 27, 52, 41, 29, 38],
        "income": [70_000, 42_000, 110_000, 87_000, 50_000, 76_000],
        "city": ["A", "B", "A", "C", "B", "A"],
        "segment": ["retail", "retail", "enterprise", "enterprise", "retail", "enterprise"],
    }
)

y = pd.Series([0, 0, 1, 1, 0, 1], name="target")

# 1) VIF analysis
vif = CatBoostVIF(cat_features=["city", "segment"], random_state=42)
vif_table = vif.fit(X)
print(vif_table)

# 2) Iterative elimination by VIF threshold
elim_result = vif.fit_eliminate(X)
print(elim_result.selected_features)
print(elim_result.config["elimination_history"])

# 3) Boruta feature selection
boruta = BorutaCatBoost(
    cat_features=["city", "segment"],
    max_iter=30,
    correction_method="bonferroni",
    random_state=42,
)
boruta.fit(X, y)
print(boruta.get_feature_names_out())
print(boruta.decision_log_.head())
```
### Result

```python

   feature       vif  r_squared  is_categorical  clamped
0  segment  3.279003   0.695029            True    False
1     city  2.228692   0.551306            True    False
2   income  2.117880   0.527830           False    False
3      age  1.916274   0.478154           False    False

# other features explain about 69.5% of 'segment' feature

['segment', 'city', 'income', 'age']
[]

# since our vif threshold is 5, so none is removed []

##### Boruta
['age', 'income', 'segment']

# city is not in this list, which means it was not confirmed. From here, it could have ended up rejected or still tentative see below.

   iteration  iteration_seed  feature  shadow_max  hits  p_upper  p_lower  \
0          1              43      age    1.251390     1     0.50      1.0   
1          1              43   income    1.251390     1     0.50      1.0   
2          1              43     city    1.251390     0     1.00      0.5   
3          1              43  segment    1.251390     1     0.50      1.0   
4          2              44      age    3.350493     2     0.25      1.0   

   adj_p_upper  adj_p_lower     status  
0          1.0          1.0  tentative  
1          1.0          1.0  tentative  
2          1.0          1.0  tentative  
3          1.0          1.0  tentative  
4          1.0          1.0  tentative

# shadow_max is the strongest importance among the shuffled “shadow” features in that iteration.
# hits is how many times a real feature beat shadow_max so far.
# p_upper tests “is this feature better than random often enough to confirm it?”
# p_lower tests “is this feature weak enough often enough to reject it?”
# adj_p_upper and adj_p_lower are those p-values after multiple-testing correction
# status is the current state after that iteration.

### STATUS LIST ###
# tentative: the feature is still undecided. It has not yet shown strong enough evidence to be confirmed, and not weak enough evidence to be rejected. This is common in early iterations.
# confirmed: the feature beat the shadow features often enough that its adjusted upper-tail p-value fell below alpha, so the code treats it as genuinely useful. Confirmed features are returned by get_feature_names_out() and included in support_.
# rejected: the feature failed badly enough against the shadow features that its adjusted lower-tail p-value fell below alpha, so the code treats it as no better than random noise. Rejected features are excluded from the selected set.

```
---

## Module 1: `CatBoostVIF`

### What it does

`CatBoostVIF` estimates feature-level collinearity by predicting each feature from all other features and converting prediction quality into a VIF-like score.

Classical VIF for feature `X_j`:

`VIF_j = 1 / (1 - R_j^2)`

where `R_j^2` comes from regressing `X_j` on all remaining predictors.

### Adaptation used here

For each feature `X_j`:

- If `X_j` is numeric, fit `CatBoostRegressor` and compute standard `R^2`.
- If `X_j` is categorical, fit `CatBoostClassifier` and compute a McFadden-like pseudo-`R^2` from log-loss:

`R2_mcfadden = 1 - (LL_model / LL_null)`

Where:

- `LL_model`: log-loss of CatBoost predictions.
- `LL_null`: log-loss of a baseline that predicts class frequencies.

### Detailed VIF algorithm (how this implementation computes it)

For each column `X_j`:

1. Set `y = X_j`, predictors = `X \\ {X_j}`.
2. Resolve categorical predictor indices for CatBoost.
3. If `X_j` is categorical and unique classes exceed `max_target_cardinality`, skip and return `NaN`.
4. Drop rows where target `y` is null (CatBoost target cannot be null).
5. Train a model and compute validation-based goodness:
- numeric target -> CatBoostRegressor -> `R^2`
- categorical target -> CatBoostClassifier -> McFadden-like pseudo-`R^2`
6. Clamp `R^2` into valid VIF range:
- `R^2 < 0` -> `0`
- `R^2 >= 1` -> `1 - eps`
7. Compute `VIF = 1 / (1 - R^2)`.
8. Repeat for all columns; optionally run elimination loop by dropping max-VIF feature until threshold is satisfied.

### Scoring method options in VIF (`scoring_method`)

#### 1) `oof` (out-of-fold cross-validation)

- More statistically stable for small/medium datasets.
- Lower optimism bias vs single split.
- More computationally expensive.
- Better when feature decisions are high impact.

#### 2) `holdout` (single train/validation split)

- Faster.
- More variance due dependence on one split.
- Better for large datasets where speed matters and split instability is less severe.

Decision rule:

- Use `oof` when selection accuracy/stability matters more than speed.
- Use `holdout` for rapid iteration or very large data.

### Threshold strategy (`threshold`)

Common practical ranges:

- around `2.5`: strict, aggressively removes collinearity.
- around `5.0`: balanced default for many applied settings.
- around `10.0`: conservative; keeps more features.

Interpretation caveat:

- This VIF is model-based (CatBoost), not linear OLS VIF. Use thresholds as heuristics, not absolute statistical law.

### What VIF captures and what it does not

VIF is a redundancy diagnostic, not a direct relevance-to-target test.

- High VIF means a feature is predictable from other features.
- It does not mean the feature is useless for your final target task.
- Best practice: combine VIF (redundancy reduction) with target-aware selection (for example Boruta).

### Why this is robust

- Uses out-of-fold (`scoring_method="oof"`) or holdout validation, not in-sample fit.
- Explicit `R^2` clamping:
  - `< 0` becomes `0`
  - `>= 1` becomes `1 - eps`
- Handles categorical targets with excessive cardinality by returning `NaN` and warning.
- Handles elimination edge case when all VIF values are `NaN`.

### Main API

#### Constructor

```python
CatBoostVIF(
    cat_features=None,
    threshold=5.0,
    scoring_method="oof",
    cv_folds=5,
    holdout_fraction=0.2,
    n_jobs=1,
    max_target_cardinality=50,
    catboost_params=None,
    random_state=42,
)
```

#### Methods

- `fit(X) -> pd.DataFrame`
- `fit_eliminate(X) -> SelectionResult`
- `get_retained_features() -> list[str]`

### Output schema

`fit(X)` returns a DataFrame with:

- `feature`
- `vif`
- `r_squared`
- `is_categorical`
- `clamped`

`fit_eliminate(X)` returns `SelectionResult`:

- `selected_features`
- `rejected_features`
- `tentative_features` (empty for VIF)
- `metrics` (final VIF table)
- `config` (includes threshold, scoring settings, elimination history)
- `random_state`

---

## Module 2: `BorutaCatBoost`

### What it does

`BorutaCatBoost` is an all-relevant feature selection method:

1. Builds shuffled shadow copies of all features.
2. Trains CatBoost on original + shadow features.
3. Compares each original feature against the strongest shadow.
4. Tracks iterative "hits" and applies statistical testing.
5. Labels each feature as:
   - Confirmed
   - Rejected
   - Tentative

### Statistical decision logic

For each feature after `i` iterations:

- Let `hits` = number of times feature importance > `shadow_max`.
- Under null hypothesis (feature no better than random), hits follow approximately:
  - `Binomial(n=i, p=0.5)`
- Two one-sided tests are computed:
  - Upper tail (`greater`) for confirmation.
  - Lower tail (`less`) for rejection.

Then p-values are corrected for multiple testing.

### Multiple-testing correction options

#### 1) Bonferroni (`correction_method="bonferroni"`)

- Adjusted p-value:
  - `p_adj = min(p * m, 1.0)`
- `m` = number of currently undecided features.
- Very conservative. Strong control of family-wise error.

Use when false positives are very costly and you prefer stricter feature confirmation.

#### 2) Benjamini-Hochberg (`correction_method="bh"`)

BH controls False Discovery Rate (FDR), not Family-Wise Error Rate (FWER).

Formal target:

- If `R` = number of rejected hypotheses and `V` = number of false rejections,
- then FDR is `E[V / max(R, 1)]`.

So BH controls the expected false-discovery proportion among selected features, while Bonferroni controls the probability of even one false positive.

### BH algorithm (adjusted p-value form used in this project)

Given `m` hypotheses and raw p-values `p1..pm`:

1. Sort p-values:
- `p_(1) <= p_(2) <= ... <= p_(m)` where `(i)` means rank position.

2. Initial rank-scaled values:
- `q_raw(i) = p_(i) * m / i`

3. Enforce monotone non-decreasing adjusted values (from right to left):
- `q(i) = min(q_raw(i), q(i+1))`

4. Map adjusted values back to original feature order.

5. Decision:
- reject hypothesis `H_j` if `q_j < alpha`.

This is equivalent to the classic BH step-up threshold rule:

- find largest `k` such that `p_(k) <= (k/m) * alpha`,
- reject all hypotheses with rank `<= k`.

### Worked numeric example

Suppose one Boruta decision pass has `m = 5` undecided features and upper-tail p-values:

- `[0.003, 0.011, 0.019, 0.041, 0.20]`

Compute rank-scaled values:

- rank 1: `0.003 * 5/1 = 0.015`
- rank 2: `0.011 * 5/2 = 0.0275`
- rank 3: `0.019 * 5/3 = 0.0317`
- rank 4: `0.041 * 5/4 = 0.05125`
- rank 5: `0.20 * 5/5 = 0.20`

Monotone correction from right to left gives adjusted values:

- `[0.015, 0.0275, 0.0317, 0.05125, 0.20]` (already monotone here)

If `alpha = 0.05`, first three are rejected (`< 0.05`) and rank 4+ are not.

Interpretation:

- among the confirmed set, expected false-discovery proportion is controlled at 5% (under BH assumptions), instead of trying to eliminate every single false positive as Bonferroni does.

### How BH is used inside `BorutaCatBoost`

At each iteration:

1. For each undecided feature, compute two p-values from Binomial tests:
- upper-tail (`greater`) for potential confirmation,
- lower-tail (`less`) for potential rejection.

2. Apply BH separately to each p-value family (upper and lower arrays).

3. Feature status update:
- Confirmed if adjusted upper-tail p-value `< alpha`.
- Rejected if adjusted lower-tail p-value `< alpha`.
- Tentative otherwise.

Why separate families:

- confirmation and rejection are distinct directional hypotheses.
- this avoids mixing opposite-tail evidence into one correction pass.

### Practical guidance: BH vs Bonferroni

#### What options are available right now?

Current `correction_method` options in this project:

- `bonferroni`
- `bh` (Benjamini-Hochberg)

There are no other correction methods implemented in `v0.1.0`.

#### How to choose between them

`bonferroni`:

- controls FWER (probability of at least one false positive),
- most conservative,
- lower false-positive risk, higher false-negative risk.

Use when:

- feature confirmation must be very strict,
- false-positive features are expensive (regulatory, scientific, or high-stakes settings),
- feature count is moderate and you can tolerate missing weak-but-real signals.

`bh`:

- controls FDR (expected false discovery proportion among selected),
- less conservative, more powerful,
- higher recall, but may allow some controlled false positives.

Use when:

- you have wider feature sets,
- discovery/recall matters,
- you can tolerate some false positives in exchange for better sensitivity.

#### Why only these two in v0.1.0?

They provide a strong practical pair:

- Bonferroni for strictness.
- BH for power.

This keeps behavior understandable while covering the two common operating modes in feature selection.

#### Other methods you may know (not implemented yet)

- Holm-Bonferroni: step-down FWER control, less conservative than Bonferroni.
- Benjamini-Yekutieli: FDR control under broader dependence, more conservative than BH.
- Storey q-value variants: estimate proportion of true nulls for more power.

These are good candidates for future versions if you need finer control.

### Statistical assumptions and caveats

- BH is exact under independent tests and remains valid under many positive-dependence settings (common in feature-selection contexts, though not guaranteed in every dataset).
- Boruta feature tests are not perfectly independent because features can be correlated, so BH should be interpreted as pragmatic control rather than a strict guarantee under all dependence structures.
- In very small `m` (few undecided features), BH and Bonferroni often behave similarly.

### Other important decision rules

- Strict hit rule: `importance > shadow_max` (ties are not hits).
- Early stopping: if no tentative features remain for `patience` consecutive iterations, stop before `max_iter`.
- Iteration seed is logged for reproducibility (`random_state + iteration_index`).

### Main API

#### Constructor

```python
BorutaCatBoost(
    cat_features=None,
    max_iter=100,
    alpha=0.05,
    correction_method="bonferroni",  # or "bh"
    importance_type="PredictionValuesChange",
    patience=5,
    task_type="classification",      # or "regression"
    catboost_params=None,
    random_state=42,
)
```

#### Methods

- `fit(X, y) -> self`
- `transform(X) -> pd.DataFrame`
- `fit_transform(X, y) -> pd.DataFrame`
- `get_support(indices=False) -> np.ndarray`
- `get_feature_names_out() -> list[str]`
- `get_selection_result() -> SelectionResult`

### Fitted attributes

- `support_`: bool mask for confirmed features
- `support_weak_`: bool mask for tentative features
- `ranking_`: 1 confirmed, 2 tentative, 3 rejected
- `importance_history_`: per-iteration feature importances and hit indicators
- `decision_log_`: per-iteration p-values, adjusted p-values, shadow threshold, status

---

## Shared infrastructure (`common/`)

### Validation (`validation.py`)

- enforces DataFrame / Series types,
- checks duplicate column names,
- rejects unsupported dtypes (datetime, timedelta, complex),
- warns for constant columns and high-null columns,
- validates target shape and basic target viability.

### Categorical utilities (`cat_feature_utils.py`)

- accepts `cat_features` as `None`, list of names, or list of indices,
- canonicalizes internally to names,
- resolves CatBoost indices safely after column changes.

### Result artifact (`result.py`)

`SelectionResult` is a common output object with:

- selected / rejected / tentative features,
- metrics DataFrame,
- config snapshot,
- random state.

---

## Reproducibility

- `random_state` is propagated through splits/shuffling/model seeds.
- Boruta stores `iteration_seed` in histories/logs.
- Determinism is validated with tests, but full bitwise reproducibility is not guaranteed for all hardware/threading modes.

---

## Robustness behavior and edge cases

### `CatBoostVIF`

- High-cardinality categorical targets can be skipped (returns `NaN` VIF).
- If all VIFs are `NaN` during elimination, process stops with warning.
- Handles classification fold constraints by falling back to holdout when stratified OOF is not feasible.
- Catches common model failures (`ValueError`, `CatBoostError`) and returns `NaN` for the affected feature.

### `BorutaCatBoost`

- Rows with null targets are dropped with warning.
- `transform()` checks that required training columns are present.
- Supports both classification and regression tasks.

---

## Performance notes

- VIF is computationally expensive: one model per feature per iteration.
- OOF scoring is more robust than holdout but slower.
- Boruta runtime scales with `max_iter * n_features`.
- Start with smaller CatBoost settings for exploration:

```python
catboost_params = {"iterations": 50, "depth": 4, "learning_rate": 0.1}
```

Then increase iterations/depth for final selection stability.

---

## Practical guidance

### When to use `CatBoostVIF`

Use when your goal is reducing redundancy / multicollinearity among predictors.

### When to use `BorutaCatBoost`

Use when your goal is identifying all relevant predictors for a specific target.

### Typical workflow

1. Use `BorutaCatBoost` to remove clearly irrelevant features.
2. Use `CatBoostVIF.fit_eliminate()` to reduce redundancy among retained features.
3. Train your final predictive model on the reduced set.

---

## Testing

Run tests from the repository root:

```bash
python -m pytest -v --tb=short
```

To target the module test directories explicitly:

```bash
python -m pytest catboost_utility/vif_catboost/tests catboost_utility/boruta_catboost/tests -v --tb=short
```

---

## Roadmap

Planned post-v0.1.0 modules include:

- CatBoost RFE
- permutation importance utilities
- stability selection
- calibration and threshold optimization
- interaction discovery
- OOF meta-feature generation

