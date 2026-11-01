from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from .data import DEFAULT_INPUT, clean_person_data, load_source, validate_input_file


@lru_cache(maxsize=2)
def load_dashboard_data(input_path: str | Path = DEFAULT_INPUT) -> pd.DataFrame:
    """Load and clean the source once for reuse by Dash callbacks."""
    metadata = validate_input_file(Path(input_path))
    cleaned, _ = clean_person_data(load_source(metadata))
    return cleaned


def filter_workers(
    data: pd.DataFrame,
    age_range: tuple[int, int] | list[int] | None = None,
    education_levels: list[str] | None = None,
    worker_classes: list[str] | None = None,
) -> pd.DataFrame:
    """Apply common dashboard filters without changing the shared data frame."""
    filtered = data
    if age_range is not None:
        lower, upper = age_range
        filtered = filtered.loc[filtered["AGEP"].between(lower, upper)]
    if education_levels:
        filtered = filtered.loc[
            filtered["EDUCATION_LEVEL"].astype("string").isin(education_levels)
        ]
    if worker_classes:
        filtered = filtered.loc[filtered["COW_LABEL"].isin(worker_classes)]
    return filtered.copy()


def weighted_wage_summary(data: pd.DataFrame) -> dict[str, float | int]:
    """Return callback-ready wage metrics for the current dashboard slice."""
    if data.empty:
        return {"sample_count": 0, "weighted_population": 0, "mean_wage": 0.0}
    weights = data["PWGTP"]
    return {
        "sample_count": int(len(data)),
        "weighted_population": int(weights.sum()),
        "mean_wage": float((data["WAGP"] * weights).sum() / weights.sum()),
    }
