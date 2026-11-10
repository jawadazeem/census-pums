from __future__ import annotations

from backend.data import (
    DEFAULT_INPUT,
    clean_person_data,
    descriptive_tables,
    load_source,
    validate_input_file,
)
from backend.statistics import (
    calculate_vif,
    covariance_and_correlation,
    flag_outliers,
    hypothesis_tests,
    run_pca_svd,
    run_regularized_regression,
)


def main() -> None:
    print("Washington Workforce Backend - Advanced Analysis WIP", flush=True)
    print("----------------------------------------------------", flush=True)
    print(
        "Regression, VIF, outlier, and PCA/SVD calculations are implemented, "
        "but their interpretation is still under review.\n",
        flush=True,
    )

    print("[1/8] Validating the original ACS person file...", flush=True)
    metadata = validate_input_file(DEFAULT_INPUT)
    print(
        f"      Found {metadata['raw_rows']:,} Washington person records "
        f"from {metadata['years'][0]}-{metadata['years'][-1]}.",
        flush=True,
    )

    print("[2/8] Loading and cleaning the DataFrame...", flush=True)
    workers, audit = clean_person_data(load_source(metadata))
    print(
        f"      Retained {len(workers):,} employed respondents age 16+.",
        flush=True,
    )
    print(
        f"      Preserved {audit['work_from_home_records']:,} work-from-home records.",
        flush=True,
    )
    print("\nCleaned DataFrame sample:", flush=True)
    sample_columns = [
        "YEAR",
        "AGEP",
        "WAGP",
        "WKHP",
        "EDUCATION_LEVEL",
        "COW_LABEL",
        "COMMUTE_STATUS",
    ]
    print(
        workers[sample_columns]
        .head(5)
        .rename(
            columns={
                "YEAR": "Year",
                "AGEP": "Age",
                "WAGP": "Wage",
                "WKHP": "Hours",
                "EDUCATION_LEVEL": "Education",
                "COW_LABEL": "Worker class",
                "COMMUTE_STATUS": "Commute",
            }
        )
        .to_string(index=False),
        flush=True,
    )

    print("[3/8] Calculating descriptive statistics...", flush=True)
    numeric, _ = descriptive_tables(workers)
    wage = numeric.set_index("variable").loc["WAGP"]
    print(f"      Weighted median wage: ${wage['weighted_median']:,.2f}", flush=True)
    print(f"      Weighted mean wage:   ${wage['weighted_mean']:,.2f}", flush=True)
    print("\nNumerical summary:", flush=True)
    summary_columns = ["variable", "count", "mean", "50%", "min", "max"]
    print(numeric[summary_columns].to_string(index=False), flush=True)

    print("\nEducation counts:", flush=True)
    education_counts = (
        workers["EDUCATION_LEVEL"]
        .astype("string")
        .value_counts(dropna=False)
        .rename_axis("Education")
        .reset_index(name="Records")
    )
    print(education_counts.to_string(index=False), flush=True)

    print("[4/8] Running statistical checks...", flush=True)
    _, correlation = covariance_and_correlation(workers)
    tests = hypothesis_tests(workers)
    t_test = tests.loc[tests["test"] == "Welch two-sample t-test"].iloc[0]
    print(
        f"      Wage-hours correlation: {correlation.loc['WAGP', 'WKHP']:.2f}",
        flush=True,
    )
    print(
        f"      Wage-commute correlation: {correlation.loc['WAGP', 'JWMNP']:.2f}",
        flush=True,
    )
    print(
        f"      Wage comparison by sex: t={t_test['statistic']:.2f}, p<0.001",
        flush=True,
    )

    print("[5/8] Flagging outliers...", flush=True)
    outliers = flag_outliers(workers)
    print(outliers["rules"].to_string(index=False), flush=True)
    model_data = outliers["sensitivity"]
    print(
        f"      Wage sensitivity dataset: {len(model_data):,} rows.",
        flush=True,
    )
    print(
        "      [WIP] Final outlier policy: TODO after comparing full and "
        "wage-sensitivity model results.",
        flush=True,
    )

    print("[6/8] Comparing regularized regression models...", flush=True)
    regression = run_regularized_regression(
        model_data,
        sample_size=20_000,
        remove_wage_outliers=False,
    )
    print(regression["metrics"].to_string(index=False), flush=True)
    print(
        "      [WIP] Preferred regression model: TODO after metric and "
        "coefficient review.",
        flush=True,
    )
    print("\nNumerical VIF:", flush=True)
    vif = calculate_vif(model_data)
    print(vif.to_string(index=False), flush=True)
    print(
        "      [WIP] Multicollinearity conclusion: TODO after confirming "
        "the agreed VIF threshold.",
        flush=True,
    )

    print("[7/8] Running PCA and SVD...", flush=True)
    pca_svd = run_pca_svd(
        model_data,
        sample_size=50_000,
        remove_wage_outliers=False,
    )
    print(pca_svd["components"].to_string(index=False), flush=True)
    print("\nPCA/SVD summary:", flush=True)
    print(pca_svd["summary"].to_string(index=False), flush=True)
    print(
        "      [WIP] Components to retain: TODO after reviewing cumulative "
        "explained variance and the scree plot.",
        flush=True,
    )

    print("[8/8] Backend functions are available for integration.", flush=True)
    print(
        "      Frontend code can import load_dashboard_data(), "
        "filter_workers(), regression_analysis(), pca_svd_analysis(), "
        "and the functions in backend.statistics.",
        flush=True,
    )
    print(
        "      Advanced-analysis findings remain provisional until the "
        "placeholders above are resolved.",
        flush=True,
    )


if __name__ == "__main__":
    main()
