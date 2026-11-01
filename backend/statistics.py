from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cityblock, cosine, euclidean

from .data import NUMERIC_FEATURES


def covariance_and_correlation(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    complete = data[NUMERIC_FEATURES].dropna()
    return complete.cov().round(2), complete.corr(method="pearson").round(4)


def distance_metrics(data: pd.DataFrame) -> dict[str, object]:
    features = ["AGEP", "WKHP", "JWMNP", "WKWN"]
    complete = data.dropna(subset=features)
    means = complete.groupby("HIGH_EARNER")[features].mean()
    scale = complete[features].std(ddof=0).replace(0, 1).to_numpy()
    lower = means.loc[0].to_numpy() / scale
    higher = means.loc[1].to_numpy() / scale
    return {
        "comparison": "Standardized mean profile: high earners versus other workers",
        "features": features,
        "euclidean": euclidean(lower, higher),
        "manhattan": cityblock(lower, higher),
        "cosine": cosine(lower, higher),
    }


def normality_results(
    data: pd.DataFrame, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tests = []
    qq_rows = []
    series_to_test = {
        "WAGP_raw": data["WAGP"].dropna(),
        "WAGP_log1p": np.log1p(data["WAGP"].dropna()),
        "JWMNP_raw": data["JWMNP"].dropna(),
        "JWMNP_log1p": np.log1p(data["JWMNP"].dropna()),
    }
    for label, series in series_to_test.items():
        sample = series.sample(min(5_000, len(series)), random_state=random_state).to_numpy()
        standardized = (sample - sample.mean()) / sample.std(ddof=1)
        shapiro = stats.shapiro(sample)
        ks = stats.kstest(standardized, "norm")
        anderson = stats.anderson(sample, dist="norm")
        tests.extend(
            [
                normality_row(label, "Shapiro-Wilk", shapiro.statistic, shapiro.pvalue, len(sample)),
                normality_row(label, "Kolmogorov-Smirnov", ks.statistic, ks.pvalue, len(sample)),
                normality_row(label, "Anderson-Darling", anderson.statistic, np.nan, len(sample)),
            ]
        )
        theoretical, ordered = stats.probplot(sample, dist="norm", fit=False)
        qq_rows.extend(
            {
                "variable": label,
                "theoretical_quantile": x,
                "ordered_value": y,
            }
            for x, y in zip(theoretical, ordered)
        )
    return pd.DataFrame(tests).round(6), pd.DataFrame(qq_rows).round(6)


def normality_row(
    variable: str, method: str, statistic: float, p_value: float, sample_n: int
) -> dict[str, object]:
    return {
        "variable": variable,
        "method": method,
        "statistic": statistic,
        "p_value": p_value,
        "sample_n": sample_n,
    }


def hypothesis_tests(data: pd.DataFrame) -> pd.DataFrame:
    male = data.loc[data["SEX"].eq(1), "WAGP"].dropna().to_numpy()
    female = data.loc[data["SEX"].eq(2), "WAGP"].dropna().to_numpy()

    t_result = stats.ttest_ind(male, female, equal_var=False)
    male_variance = np.var(male, ddof=1)
    female_variance = np.var(female, ddof=1)
    f_statistic = male_variance / female_variance
    f_cdf = stats.f.cdf(f_statistic, len(male) - 1, len(female) - 1)
    f_p_value = 2 * min(f_cdf, 1 - f_cdf)

    return pd.DataFrame(
        [
            {
                "test": "Welch two-sample t-test",
                "null_hypothesis": "Mean annual wages are equal for male and female respondents.",
                "statistic": t_result.statistic,
                "p_value": t_result.pvalue,
                "conclusion_alpha_0_05": conclusion(t_result.pvalue),
                "note": "Unweighted test; ACS weights are used for population descriptions.",
            },
            {
                "test": "Two-sided F-test for variance",
                "null_hypothesis": "Annual wage variances are equal for male and female respondents.",
                "statistic": f_statistic,
                "p_value": f_p_value,
                "conclusion_alpha_0_05": conclusion(f_p_value),
                "note": "Wage skew and the large sample make this test sensitive.",
            },
        ]
    ).round(6)


def conclusion(p_value: float) -> str:
    return "Reject H0" if p_value < 0.05 else "Fail to reject H0"
