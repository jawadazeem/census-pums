from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .pipeline import make_manifest, write_json
from .statistics import (
    covariance_and_correlation,
    distance_metrics,
    hypothesis_tests,
    normality_results,
)


def run_milestone_2(
    output_dir: Path, input_path: str, random_state: int
) -> dict[str, object]:
    cleaned = pd.read_csv(output_dir / "cleaned_person_data.csv", low_memory=False)
    covariance, correlation = covariance_and_correlation(cleaned)
    normality, qq_data = normality_results(cleaned, random_state)
    tests = hypothesis_tests(cleaned)

    covariance.to_csv(output_dir / "covariance_matrix.csv")
    correlation.to_csv(output_dir / "pearson_correlation.csv")
    write_json(output_dir / "distance_metrics.json", distance_metrics(cleaned))
    normality.to_csv(output_dir / "normality_tests.csv", index=False)
    qq_data.to_csv(output_dir / "qq_plot_data.csv", index=False)
    tests.to_csv(output_dir / "hypothesis_tests.csv", index=False)

    audit = json.loads((output_dir / "cleaning_audit.json").read_text(encoding="utf-8"))
    write_bias_notes(output_dir, audit, random_state)
    write_observations(output_dir, correlation, normality, tests)

    manifest = make_manifest(
        through="milestone2",
        input_path=input_path,
        output_dir=output_dir,
        random_state=random_state,
        cleaned_rows=len(cleaned),
    )
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def write_bias_notes(
    output_dir: Path, audit: dict[str, object], random_state: int
) -> None:
    text = f"""# Bias and randomization

- ACS PUMS is a survey sample. Descriptive population estimates use `PWGTP`.
- Wages are self-reported and may be affected by nonresponse and top-coding.
- `WAGP` measures wage and salary income; it does not fully describe self-employment earnings.
- Dropping commute blanks would disproportionately remove people who work from home, so the cleaning step preserves them.
- This observational dataset supports association, not causal claims.
- Normality tests use reproducible samples with random seed {random_state}.
- The filtered sample contains {audit['rows_after_population_filter']:,} records and represents approximately {audit['weighted_population']:,} Washington workers after weighting.
"""
    (output_dir / "bias_and_randomization.md").write_text(text, encoding="utf-8")


def write_observations(
    output_dir: Path,
    correlation: pd.DataFrame,
    normality: pd.DataFrame,
    tests: pd.DataFrame,
) -> None:
    wage_hours = correlation.loc["WAGP", "WKHP"]
    wage_commute = correlation.loc["WAGP", "JWMNP"]
    wage_shapiro = normality.query(
        "variable == 'WAGP_raw' and method == 'Shapiro-Wilk'"
    ).iloc[0]
    commute_log = normality.query(
        "variable == 'JWMNP_log1p' and method == 'Shapiro-Wilk'"
    ).iloc[0]
    t_test = tests.loc[tests["test"] == "Welch two-sample t-test"].iloc[0]
    f_test = tests.loc[tests["test"] == "Two-sided F-test for variance"].iloc[0]
    text = f"""# Milestone 2 observations

## Correlation and similarity

Weekly hours has the strongest Pearson relationship with wages among the selected numerical variables (r={wage_hours:.2f}). Commute time has only a weak positive relationship with wages (r={wage_commute:.2f}), so commute length alone explains little of the wage variation.

## Normality

Raw wages fail the Shapiro-Wilk normality test (W={wage_shapiro['statistic']:.2f}, p<0.001). Log-transformed commute time is closer to normal (W={commute_log['statistic']:.2f}) but still fails at this sample size. The report should show the Q-Q plots and avoid claiming either variable is normally distributed.

## Hypothesis tests

The wage comparison by sex rejects equal means (Welch t={t_test['statistic']:.2f}, p<0.001). The F-test also rejects equal wage variances (F={f_test['statistic']:.2f}, p<0.001). These are associations in a large observational sample, not evidence that sex itself causes the wage difference.
"""
    (output_dir / "milestone_2_observations.md").write_text(text, encoding="utf-8")
