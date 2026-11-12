from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cityblock, cosine, euclidean

from .data import NUMERIC_FEATURES


def descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    return data[NUMERIC_FEATURES].describe().T.reset_index(names="variable").round(2)


def correlation_tables(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    numeric = data[NUMERIC_FEATURES]
    return {
        "pearson": numeric.corr(method="pearson").round(4),
        "spearman": numeric.corr(method="spearman").round(4),
    }


def distance_metrics(data: pd.DataFrame) -> pd.DataFrame:
    features = ["AGEP", "WKHP", "JWMNP", "WKWN"]
    complete = data.dropna(subset=features)
    means = complete.groupby("HIGH_EARNER")[features].mean()
    scale = complete[features].std(ddof=0).replace(0, 1).to_numpy()
    lower = means.loc[0].to_numpy() / scale
    higher = means.loc[1].to_numpy() / scale
    return pd.DataFrame([
        {"distance": "Euclidean", "value": euclidean(lower, higher)},
        {"distance": "Manhattan", "value": cityblock(lower, higher)},
        {"distance": "Cosine", "value": cosine(lower, higher)},
    ]).round(4)


def normality_results(data: pd.DataFrame, random_state: int = 3654) -> pd.DataFrame:
    rows = []
    series = {
        "Wage": data["WAGP"].dropna(),
        "Log wage": np.log1p(data["WAGP"].dropna()),
        "Commute time": data["JWMNP"].dropna(),
        "Log commute time": np.log1p(data["JWMNP"].dropna()),
    }
    for label, values in series.items():
        sample = values.sample(min(5_000, len(values)), random_state=random_state)
        standardized = (sample - sample.mean()) / sample.std(ddof=1)
        shapiro = stats.shapiro(sample)
        ks = stats.kstest(standardized, "norm")
        anderson = stats.anderson(sample, dist="norm")
        rows.extend([
            {"variable": label, "test": "Shapiro-Wilk", "statistic": shapiro.statistic, "p_value": shapiro.pvalue},
            {"variable": label, "test": "Kolmogorov-Smirnov", "statistic": ks.statistic, "p_value": ks.pvalue},
            {"variable": label, "test": "Anderson-Darling", "statistic": anderson.statistic, "p_value": np.nan},
        ])
    return pd.DataFrame(rows).round(6)


def wage_sex_t_test(data: pd.DataFrame) -> pd.DataFrame:
    male = data.loc[data["SEX"].eq(1), "WAGP"].dropna()
    female = data.loc[data["SEX"].eq(2), "WAGP"].dropna()
    result = stats.ttest_ind(male, female, equal_var=False)
    return pd.DataFrame([{
        "test": "Welch two-sample t-test",
        "null_hypothesis": "Mean wages are equal for male and female respondents.",
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "conclusion": "Reject H0" if result.pvalue < 0.05 else "Fail to reject H0",
    }]).round(6)
