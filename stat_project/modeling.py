from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .data import NUMERIC_FEATURES


def flag_outliers(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Flag unusual wages and hours using the 1.5 x IQR rule."""
    flagged = data.copy()
    rules = []

    for column in ["WAGP", "WKHP"]:
        q1, q3 = flagged[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flag_column = f"{column}_OUTLIER"
        flagged[flag_column] = flagged[column].lt(lower) | flagged[column].gt(upper)
        rules.append(
            {
                "variable": column,
                "lower_bound": lower,
                "upper_bound": upper,
                "flagged_rows": int(flagged[flag_column].sum()),
                "action": (
                    "Use a second dataset without these wages"
                    if column == "WAGP"
                    else "Keep valid weekly-hour values"
                ),
            }
        )

    sensitivity = flagged.loc[~flagged["WAGP_OUTLIER"]].copy()
    return {
        "data": flagged,
        "rules": pd.DataFrame(rules).round(2),
        "sensitivity": sensitivity,
    }


def simple_wage_regression(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Fit the course's simple linear model: annual wage versus weekly hours."""
    frame = data[["WKHP", "WAGP"]].dropna()
    x = sm.add_constant(frame["WKHP"])
    model = sm.OLS(frame["WAGP"], x).fit()

    confidence = model.conf_int()
    coefficients = pd.DataFrame(
        {
            "term": ["intercept", "weekly hours"],
            "coefficient": model.params.values,
            "standard_error": model.bse.values,
            "t_statistic": model.tvalues.values,
            "p_value": model.pvalues.values,
            "ci_lower": confidence[0].values,
            "ci_upper": confidence[1].values,
        }
    ).round(4)

    residual_standard_error = np.sqrt(model.ssr / model.df_resid)
    metrics = pd.DataFrame(
        [
            {
                "rows": len(frame),
                "r_squared": model.rsquared,
                "residual_standard_error": residual_standard_error,
                "mse": np.mean(model.resid**2),
            }
        ]
    ).round(4)

    hours = np.arange(int(frame["WKHP"].min()), int(frame["WKHP"].max()) + 1)
    fit_line = pd.DataFrame(
        {
            "WKHP": hours,
            "predicted_WAGP": model.predict(sm.add_constant(hours)),
        }
    ).round(2)

    return {
        "coefficients": coefficients,
        "metrics": metrics,
        "fit_line": fit_line,
    }


def compare_outlier_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    """Fit the same simple regression with and without extreme wages."""
    datasets = {
        "all valid wages": data,
        "without wage outliers": flag_outliers(data)["sensitivity"],
    }
    rows = []
    for label, frame in datasets.items():
        result = simple_wage_regression(frame)
        metrics = result["metrics"].iloc[0]
        slope = result["coefficients"].loc[
            result["coefficients"]["term"] == "weekly hours", "coefficient"
        ].iloc[0]
        rows.append(
            {
                "dataset": label,
                "rows": int(metrics["rows"]),
                "weekly_hours_slope": slope,
                "r_squared": metrics["r_squared"],
                "residual_standard_error": metrics["residual_standard_error"],
            }
        )
    return pd.DataFrame(rows).round(4)


def run_pca_svd(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Standardize the five numerical features and apply PCA/SVD."""
    features = ["AGEP", "WAGP", "WKHP", "COMMUTE_MINUTES", "WKWN"]
    matrix = SimpleImputer(strategy="median").fit_transform(data[features])
    scaled = StandardScaler().fit_transform(matrix)
    pca = PCA().fit(scaled)
    scores = pca.transform(scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    components = pd.DataFrame(
        {
            "component": np.arange(1, len(features) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": cumulative,
            "singular_value": pca.singular_values_,
        }
    ).round(6)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=[f"PC{i}" for i in range(1, len(features) + 1)],
    ).reset_index(names="feature").round(6)
    score_table = pd.DataFrame(scores[:, :2], columns=["PC1", "PC2"]).round(6)
    summary = pd.DataFrame(
        [
            {
                "condition_number": pca.singular_values_[0] / pca.singular_values_[-1],
                "first_three_components_variance": cumulative[2],
            }
        ]
    ).round(6)
    return {
        "components": components,
        "loadings": loadings,
        "scores": score_table,
        "summary": summary,
    }
