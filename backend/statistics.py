from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cityblock, cosine, euclidean
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import NUMERIC_FEATURES


REGRESSION_NUMERIC = ["AGEP", "WKHP", "JWMNP", "WKWN"]
REGRESSION_CATEGORICAL = [
    "EDUCATION_LEVEL",
    "SEX_LABEL",
    "MAR_LABEL",
    "COW_LABEL",
    "RAC1P_LABEL",
    "JWTRNS_LABEL",
]


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


def flag_outliers(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """flag iqr outliers and make a wage sensitivity dataset"""
    flagged = data.copy()
    rules = []

    for column in ["WAGP", "WKHP"]:
        q1, q3 = flagged[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flag_column = f"{column}_IQR_OUTLIER"
        flagged[flag_column] = flagged[column].lt(lower) | flagged[column].gt(upper)
        action = (
            "Removed from wage sensitivity dataset"
            if column == "WAGP"
            else "Flagged only; ACS weekly hours from 1 to 99 remain valid"
        )
        rules.append(
            {
                "variable": column,
                "lower_bound": lower,
                "upper_bound": upper,
                "flagged_rows": int(flagged[flag_column].sum()),
                "action": action,
            }
        )

    sensitivity = flagged.loc[~flagged["WAGP_IQR_OUTLIER"]].copy()
    return {
        "data": flagged,
        "rules": pd.DataFrame(rules).round(2),
        "sensitivity": sensitivity,
    }


def calculate_vif(data: pd.DataFrame) -> pd.DataFrame:
    """calculate vif for the numeric predictors"""
    matrix = data[REGRESSION_NUMERIC].copy()
    matrix = matrix.fillna(matrix.median())
    rows = []

    for feature in REGRESSION_NUMERIC:
        other_features = [name for name in REGRESSION_NUMERIC if name != feature]
        model = LinearRegression().fit(matrix[other_features], matrix[feature])
        r_squared = model.score(matrix[other_features], matrix[feature])
        vif = float("inf") if r_squared >= 1 else 1 / (1 - r_squared)
        rows.append({"feature": feature, "vif": vif})

    return pd.DataFrame(rows).round(4)


def run_regularized_regression(
    data: pd.DataFrame,
    sample_size: int = 20_000,
    random_state: int = 3654,
    remove_wage_outliers: bool = True,
) -> dict[str, pd.DataFrame]:
    """compare ridge, lasso, and elastic net for log annual wage"""
    source = flag_outliers(data)["sensitivity"] if remove_wage_outliers else data
    predictors = REGRESSION_NUMERIC + REGRESSION_CATEGORICAL
    frame = source.loc[source["WAGP"].notna(), [*predictors, "WAGP"]].copy()
    for column in REGRESSION_CATEGORICAL:
        frame[column] = frame[column].astype("string").fillna("Missing or unknown")
    if len(frame) > sample_size:
        frame = frame.sample(sample_size, random_state=random_state)

    x_train, x_test, y_train, y_test = train_test_split(
        frame[predictors],
        np.log1p(frame["WAGP"]),
        test_size=0.20,
        random_state=random_state,
    )
    estimators = {
        "ridge": RidgeCV(alphas=np.logspace(-3, 3, 13)),
        "lasso": LassoCV(cv=5, random_state=random_state, max_iter=20_000),
        "elastic_net": ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            cv=5,
            random_state=random_state,
            max_iter=20_000,
        ),
    }

    metrics = []
    coefficients = []
    predictions = []
    for name, estimator in estimators.items():
        pipeline = Pipeline(
            [
                ("preprocess", regression_preprocessor()),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)
        fitted_model = pipeline.named_steps["model"]
        metrics.append(
            {
                "model": name,
                "sample_rows": len(frame),
                "train_rows": len(x_train),
                "test_rows": len(x_test),
                "r2_log_wage": r2_score(y_test, predicted),
                "mae_log_wage": mean_absolute_error(y_test, predicted),
                "rmse_log_wage": np.sqrt(mean_squared_error(y_test, predicted)),
                "alpha": getattr(fitted_model, "alpha_", np.nan),
                "l1_ratio": getattr(fitted_model, "l1_ratio_", np.nan),
            }
        )

        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        coefficients.extend(
            {
                "model": name,
                "feature": feature,
                "coefficient": coefficient,
            }
            for feature, coefficient in zip(feature_names, fitted_model.coef_)
        )
        predictions.extend(
            {
                "model": name,
                "actual_log_wage": actual,
                "predicted_log_wage": estimate,
            }
            for actual, estimate in zip(y_test, predicted)
        )

    return {
        "metrics": pd.DataFrame(metrics).round(6),
        "coefficients": pd.DataFrame(coefficients).round(6),
        "predictions": pd.DataFrame(predictions).round(6),
    }


def regression_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, REGRESSION_NUMERIC),
            ("categorical", categorical, REGRESSION_CATEGORICAL),
        ]
    )


def run_pca_svd(
    data: pd.DataFrame,
    sample_size: int = 50_000,
    random_state: int = 3654,
    remove_wage_outliers: bool = True,
) -> dict[str, pd.DataFrame]:
    """run pca and svd on the standardized numeric columns"""
    source = flag_outliers(data)["sensitivity"] if remove_wage_outliers else data
    if len(source) > sample_size:
        source = source.sample(sample_size, random_state=random_state)

    matrix = SimpleImputer(strategy="median").fit_transform(source[NUMERIC_FEATURES])
    scaled = StandardScaler().fit_transform(matrix)
    pca = PCA().fit(scaled)
    scores = pca.transform(scaled)
    singular_values = np.linalg.svd(scaled, compute_uv=False, full_matrices=False)
    condition_number = singular_values.max() / singular_values.min()

    components = pd.DataFrame(
        {
            "component": np.arange(1, len(NUMERIC_FEATURES) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
            "singular_value": singular_values,
        }
    ).round(6)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=NUMERIC_FEATURES,
        columns=[f"PC{number}" for number in range(1, len(NUMERIC_FEATURES) + 1)],
    ).reset_index(names="feature").round(6)
    score_table = pd.DataFrame(
        scores[:, :2],
        columns=["PC1", "PC2"],
        index=source.index,
    )
    if "PERSON_ID" in source:
        score_table.insert(0, "PERSON_ID", source.loc[score_table.index, "PERSON_ID"])
    score_table = score_table.reset_index(drop=True).round(6)
    summary = pd.DataFrame(
        [
            {
                "sample_rows": len(source),
                "features": ", ".join(NUMERIC_FEATURES),
                "condition_number": condition_number,
                "first_three_components_variance": components.loc[
                    2, "cumulative_explained_variance"
                ],
            }
        ]
    ).round(6)

    return {
        "components": components,
        "loadings": loadings,
        "scores": score_table,
        "summary": summary,
    }
