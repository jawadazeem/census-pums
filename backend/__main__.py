from __future__ import annotations

from backend.data import (
    DEFAULT_INPUT,
    clean_person_data,
    descriptive_tables,
    load_source,
    validate_input_file,
)
from backend.statistics import covariance_and_correlation, hypothesis_tests


def main() -> None:
    print("Washington Workforce Backend", flush=True)
    print("----------------------------", flush=True)

    print("[1/5] Validating the original ACS person file...", flush=True)
    metadata = validate_input_file(DEFAULT_INPUT)
    print(
        f"      Found {metadata['raw_rows']:,} Washington person records "
        f"from {metadata['years'][0]}-{metadata['years'][-1]}.",
        flush=True,
    )

    print("[2/5] Loading and cleaning the DataFrame...", flush=True)
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

    print("[3/5] Calculating descriptive statistics...", flush=True)
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

    print("[4/5] Running statistical checks...", flush=True)
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

    print("[5/5] Backend is ready for Dash callbacks.", flush=True)
    print(
        "      Frontend code can import load_dashboard_data(), "
        "filter_workers(), and the functions in backend.statistics.",
        flush=True,
    )


if __name__ == "__main__":
    main()
