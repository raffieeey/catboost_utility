"""Common result dataclass for feature selection outputs."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class SelectionResult:
    """Standardized output from any feature selection method.

    Attributes
    ----------
    selected_features : list[str]
        Features that passed selection.
    rejected_features : list[str]
        Features that were eliminated.
    tentative_features : list[str]
        Features that were inconclusive (empty for methods like VIF/RFE).
    metrics : pd.DataFrame
        Method-specific metrics table (VIF scores, importance history, etc.).
    config : dict
        Full run configuration for reproducibility.
    random_state : int | None
        Random seed used for the run.
    """

    selected_features: list[str] = field(default_factory=list)
    rejected_features: list[str] = field(default_factory=list)
    tentative_features: list[str] = field(default_factory=list)
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    config: dict = field(default_factory=dict)
    random_state: int | None = None

    def __repr__(self) -> str:
        return (
            f"SelectionResult("
            f"selected={len(self.selected_features)}, "
            f"rejected={len(self.rejected_features)}, "
            f"tentative={len(self.tentative_features)})"
        )
