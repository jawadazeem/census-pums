from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from .data import DEFAULT_INPUT, clean_person_data, load_source, validate_input_file
from .statistics import (
    calculate_vif,
    flag_outliers,
    run_pca_svd,
    run_regularized_regression,
)


@lru_cache(maxsize=2)
def load_dashboard_data(input_path: str | Path = DEFAULT_INPUT) -> pd.DataFrame:
    """load and clean the data once for dash callbacks"""
    metadata = validate_input_file(Path(input_path))
    cleaned, _ = clean_person_data(load_source(metadata))
    return cleaned


def filter_workers(
    data: pd.DataFrame,
    age_range: tuple[int, int] | list[int] | None = None,
    education_levels: list[str] | None = None,
    worker_classes: list[str] | None = None,
) -> pd.DataFrame:
    """filter a copy of the shared data"""
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
    """get a few wage numbers for the dashboard"""
    if data.empty:
        return {"sample_count": 0, "weighted_population": 0, "mean_wage": 0.0}
    weights = data["PWGTP"]
    return {
        "sample_count": int(len(data)),
        "weighted_population": int(weights.sum()),
        "mean_wage": float((data["WAGP"] * weights).sum() / weights.sum()),
    }


def outlier_analysis(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return flag_outliers(data)


def regression_analysis(
    data: pd.DataFrame,
    sample_size: int = 20_000,
    random_state: int = 3654,
) -> dict[str, pd.DataFrame]:
    return run_regularized_regression(data, sample_size, random_state)


def vif_table(data: pd.DataFrame) -> pd.DataFrame:
    return calculate_vif(data)


def pca_svd_analysis(
    data: pd.DataFrame,
    sample_size: int = 50_000,
    random_state: int = 3654,
) -> dict[str, pd.DataFrame]:
    return run_pca_svd(data, sample_size, random_state)
