from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import NUMERIC_FEATURES


REGRESSION_NUMERIC = ["AGEP", "WKHP", "COMMUTE_MINUTES", "WKWN"]
REGRESSION_CATEGORICAL = [
    "EDUCATION_LEVEL", "SEX_LABEL", "MAR_LABEL", "COW_LABEL",
    "RAC1P_LABEL", "JWTRNS_LABEL",
]


def flag_outliers(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Flag IQR outliers and create a wage-only sensitivity dataset."""
    flagged = data.copy()
    rules = []
    for column in ["WAGP", "WKHP"]:
        q1, q3 = flagged[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flag = f"{column}_OUTLIER"
        flagged[flag] = flagged[column].lt(lower) | flagged[column].gt(upper)
        rules.append({
            "variable": column,
            "lower_bound": lower,
            "upper_bound": upper,
            "flagged_rows": int(flagged[flag].sum()),
            "action": "Remove only for sensitivity" if column == "WAGP" else "Keep valid 1-99 hour values",
        })
    sensitivity = flagged.loc[~flagged["WAGP_OUTLIER"]].copy()
    return {"data": flagged, "rules": pd.DataFrame(rules).round(2), "sensitivity": sensitivity}


def prepare_regression_data(data: pd.DataFrame):
    frame = data[[*REGRESSION_NUMERIC, *REGRESSION_CATEGORICAL, "WAGP"]].copy()
    for column in REGRESSION_NUMERIC:
        frame[column] = frame[column].fillna(frame[column].median())
    for column in REGRESSION_CATEGORICAL:
        frame[column] = frame[column].astype("string").fillna("Missing or unknown")
    x = pd.get_dummies(
        frame[REGRESSION_NUMERIC + REGRESSION_CATEGORICAL],
        columns=REGRESSION_CATEGORICAL,
        drop_first=True,
        dtype=float,
    )
    return x, np.log1p(frame["WAGP"])


def run_regression(data: pd.DataFrame, random_state: int = 3654) -> dict[str, pd.DataFrame]:
    """Compare ordinary least squares with the regularized models from Lecture 11."""
    x, y = prepare_regression_data(data)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state
    )
    models = {
        "ordinary least squares": LinearRegression(),
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 30), cv=5)),
        "lasso": make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=random_state, max_iter=10_000)),
        "elastic net": make_pipeline(
            StandardScaler(),
            ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, random_state=random_state, max_iter=10_000),
        ),
    }

    metrics = []
    coefficients = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        mse = mean_squared_error(y_test, predicted)
        fitted = model[-1] if hasattr(model, "steps") else model
        metrics.append({
            "model": name,
            "test_mse": mse,
            "test_rmse": np.sqrt(mse),
            "test_r2": r2_score(y_test, predicted),
            "alpha": getattr(fitted, "alpha_", np.nan),
            "l1_ratio": getattr(fitted, "l1_ratio_", np.nan),
        })
        coefficients.extend(
            {
                "model": name,
                "feature": feature,
                "coefficient": value,
                "coefficient_scale": "original" if name == "ordinary least squares" else "standardized",
            }
            for feature, value in zip(x.columns, fitted.coef_)
        )

    inference = sm.OLS(y_train, sm.add_constant(x_train)).fit()
    overall_f_test = pd.DataFrame([{
        "null_hypothesis": "All regression slopes equal zero.",
        "f_statistic": inference.fvalue,
        "p_value": inference.f_pvalue,
        "r_squared": inference.rsquared,
        "adjusted_r_squared": inference.rsquared_adj,
        "conclusion": "At least one predictor helps" if inference.f_pvalue < 0.05 else "No evidence that predictors help",
    }]).round(6)
    return {
        "metrics": pd.DataFrame(metrics).round(6),
        "coefficients": pd.DataFrame(coefficients).round(6),
        "overall_f_test": overall_f_test,
    }


def compare_outlier_sensitivity(data: pd.DataFrame, random_state: int = 3654) -> pd.DataFrame:
    sets = {"all valid wages": data, "without wage outliers": flag_outliers(data)["sensitivity"]}
    rows = []
    for label, frame in sets.items():
        x, y = prepare_regression_data(frame)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)
        predicted = LinearRegression().fit(x_train, y_train).predict(x_test)
        rows.append({
            "dataset": label,
            "rows": len(frame),
            "test_mse": mean_squared_error(y_test, predicted),
            "test_r2": r2_score(y_test, predicted),
        })
    return pd.DataFrame(rows).round(6)


def calculate_vif(data: pd.DataFrame) -> pd.DataFrame:
    matrix = data[REGRESSION_NUMERIC].copy().fillna(data[REGRESSION_NUMERIC].median())
    rows = []
    for feature in REGRESSION_NUMERIC:
        others = [column for column in REGRESSION_NUMERIC if column != feature]
        r_squared = LinearRegression().fit(matrix[others], matrix[feature]).score(matrix[others], matrix[feature])
        rows.append({"feature": feature, "vif": 1 / (1 - r_squared)})
    return pd.DataFrame(rows).round(4)


def run_pca_svd(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pca_features = ["AGEP", "WAGP", "WKHP", "COMMUTE_MINUTES", "WKWN"]
    matrix = SimpleImputer(strategy="median").fit_transform(data[pca_features])
    scaled = StandardScaler().fit_transform(matrix)
    pca = PCA().fit(scaled)
    scores = pca.transform(scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    components = pd.DataFrame({
        "component": np.arange(1, len(pca_features) + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": cumulative,
        "singular_value": pca.singular_values_,
    }).round(6)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=pca_features,
        columns=[f"PC{i}" for i in range(1, len(pca_features) + 1)],
    ).reset_index(names="feature").round(6)
    score_table = pd.DataFrame(scores[:, :2], columns=["PC1", "PC2"]).round(6)
    summary = pd.DataFrame([{
        "condition_number": pca.singular_values_[0] / pca.singular_values_[-1],
        "first_three_components_variance": cumulative[2],
    }]).round(6)
    return {"components": components, "loadings": loadings, "scores": score_table, "summary": summary}
