from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from .data import DEFAULT_INPUT, load_workers
from .modeling import calculate_vif, flag_outliers, run_pca_svd, run_regression


@lru_cache(maxsize=2)
def load_dashboard_data(input_path: str | Path = DEFAULT_INPUT) -> pd.DataFrame:
    return load_workers(input_path)


def filter_workers(data: pd.DataFrame, age_range=None, education_levels=None) -> pd.DataFrame:
    filtered = data
    if age_range:
        filtered = filtered.loc[filtered["AGEP"].between(*age_range)]
    if education_levels:
        filtered = filtered.loc[filtered["EDUCATION_LEVEL"].astype("string").isin(education_levels)]
    return filtered.copy()


def outlier_analysis(data: pd.DataFrame):
    return flag_outliers(data)


def regression_analysis(data: pd.DataFrame):
    return run_regression(data)


def vif_table(data: pd.DataFrame):
    return calculate_vif(data)


def pca_svd_analysis(data: pd.DataFrame):
    return run_pca_svd(data)
