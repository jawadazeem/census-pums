from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPOSITORY_ROOT / "data" / "raw"
WASHINGTON_STATE_CODE = 53
EMPLOYED_ESR_CODES = {1, 2, 4, 5}

NUMERIC_FEATURES = ["AGEP", "WAGP", "WKHP", "JWMNP", "WKWN"]
CATEGORICAL_FEATURES = ["SEX", "SCHL", "MAR", "COW", "RAC1P", "JWTRNS"]
SOURCE_COLUMNS = [
    "RT",
    "SERIALNO",
    "SPORDER",
    "STATE",
    "PWGTP",
    "ESR",
    *NUMERIC_FEATURES,
    *CATEGORICAL_FEATURES,
]

CATEGORY_LABELS = {
    "SEX": {1: "Male", 2: "Female"},
    "MAR": {
        1: "Married",
        2: "Widowed",
        3: "Divorced",
        4: "Separated",
        5: "Never married",
    },
    "COW": {
        1: "Private for-profit",
        2: "Private nonprofit",
        3: "Local government",
        4: "State government",
        5: "Federal government",
        6: "Self-employed, not incorporated",
        7: "Self-employed, incorporated",
        8: "Unpaid family worker",
    },
    "RAC1P": {
        1: "White alone",
        2: "Black or African American alone",
        3: "American Indian alone",
        4: "Alaska Native alone",
        5: "American Indian and Alaska Native",
        6: "Asian alone",
        7: "Native Hawaiian or Pacific Islander alone",
        8: "Some other race alone",
        9: "Two or more races",
    },
    "JWTRNS": {
        1: "Car, truck, or van",
        2: "Bus",
        3: "Streetcar or trolley",
        4: "Subway or elevated rail",
        5: "Railroad",
        6: "Ferryboat",
        7: "Taxicab",
        8: "Motorcycle",
        9: "Bicycle",
        10: "Walked",
        11: "Worked from home",
        12: "Other method",
    },
}


class DataValidationError(ValueError):
    """The supplied file does not match the project's data contract."""


def resolve_input_path(input_path: Path) -> Path:
    input_path = Path(input_path)
    if input_path.is_file():
        return input_path
    if input_path.is_dir():
        expected = input_path / "psam_p53.csv"
        if expected.is_file():
            return expected
        candidates = sorted(input_path.glob("*.csv"))
        if len(candidates) == 1:
            return candidates[0]
        raise DataValidationError(
            f"Expected psam_p53.csv or one CSV in {input_path}; found {len(candidates)}."
        )
    raise DataValidationError(f"Input path does not exist: {input_path}")


def validate_input_file(input_path: Path) -> dict[str, object]:
    csv_path = resolve_input_path(input_path)
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    missing = sorted(set(SOURCE_COLUMNS) - set(header))
    if missing:
        raise DataValidationError(
            "The CSV is missing required person fields: " + ", ".join(missing)
        )

    rows = 0
    states: set[int] = set()
    record_types: set[str] = set()
    years: set[int] = set()
    for chunk in pd.read_csv(
        csv_path,
        usecols=["RT", "SERIALNO", "STATE"],
        chunksize=100_000,
        low_memory=False,
    ):
        rows += len(chunk)
        record_types.update(chunk["RT"].dropna().astype(str))
        states.update(pd.to_numeric(chunk["STATE"], errors="coerce").dropna().astype(int))
        years.update(
            pd.to_numeric(chunk["SERIALNO"].astype(str).str[:4], errors="coerce")
            .dropna()
            .astype(int)
        )

    if record_types != {"P"}:
        raise DataValidationError(
            f"Expected ACS person records (RT=P); found {sorted(record_types)}."
        )
    if states != {WASHINGTON_STATE_CODE}:
        raise DataValidationError(
            f"Expected Washington (STATE=53); found {sorted(states)}."
        )

    return {
        "path": str(csv_path.resolve()),
        "record_type": "P",
        "state": WASHINGTON_STATE_CODE,
        "years": sorted(years),
        "raw_rows": rows,
        "source_columns": len(header),
    }


def load_source(metadata: dict[str, object]) -> pd.DataFrame:
    return pd.read_csv(
        Path(str(metadata["path"])),
        usecols=SOURCE_COLUMNS,
        low_memory=False,
    )


def collapse_education(values: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[0, 15, 17, 19, 20, 21, 24],
        labels=[
            "Less than high school",
            "High school or GED",
            "Some college",
            "Associate degree",
            "Bachelor's degree",
            "Graduate degree",
        ],
    )


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    frame = pd.DataFrame({"value": values, "weight": weights}).dropna()
    frame = frame.loc[frame["weight"] > 0]
    return float(np.average(frame["value"], weights=frame["weight"]))


def weighted_median(values: pd.Series, weights: pd.Series) -> float:
    frame = pd.DataFrame({"value": values, "weight": weights}).dropna()
    frame = frame.loc[frame["weight"] > 0].sort_values("value")
    cutoff = frame["weight"].sum() / 2
    return float(frame.loc[frame["weight"].cumsum() >= cutoff, "value"].iloc[0])


def clean_person_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    data = raw.copy()
    for column in [
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
        "PWGTP",
        "ESR",
        "STATE",
        "SPORDER",
    ]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    duplicate_count = int(data.duplicated(["SERIALNO", "SPORDER"]).sum())
    data = data.drop_duplicates(["SERIALNO", "SPORDER"], keep="first")
    data = data.loc[
        (data["STATE"] == WASHINGTON_STATE_CODE)
        & (data["AGEP"] >= 16)
        & data["ESR"].isin(EMPLOYED_ESR_CODES)
    ].copy()

    data["YEAR"] = pd.to_numeric(
        data["SERIALNO"].astype(str).str[:4], errors="coerce"
    ).astype("Int64")
    data["PERSON_ID"] = (
        data["SERIALNO"].astype(str)
        + "-"
        + data["SPORDER"].astype("Int64").astype(str)
    )
    data["EDUCATION_LEVEL"] = collapse_education(data["SCHL"])
    for column, mapping in CATEGORY_LABELS.items():
        data[f"{column}_LABEL"] = data[column].map(mapping).fillna("Missing or unknown")

    data["COMMUTE_STATUS"] = np.select(
        [data["JWTRNS"].eq(11), data["JWTRNS"].notna()],
        ["Works from home", "Commutes"],
        default="Not reported or not applicable",
    )
    wage_median = weighted_median(data["WAGP"], data["PWGTP"])
    data["HIGH_EARNER"] = (data["WAGP"] > wage_median).astype("int8")

    core_fields = [
        "AGEP",
        "WAGP",
        "WKHP",
        "WKWN",
        "SEX",
        "SCHL",
        "MAR",
        "COW",
        "RAC1P",
        "PWGTP",
    ]
    data["COMPLETE_CORE_CASE"] = data[core_fields].notna().all(axis=1)
    data["COMPLETE_COMMUTE_CASE"] = data[[*core_fields, "JWTRNS", "JWMNP"]].notna().all(axis=1)
    data = data.reset_index(drop=True)

    audit = {
        "rows_raw": int(len(raw)),
        "duplicate_person_records_removed": duplicate_count,
        "rows_after_population_filter": int(len(data)),
        "complete_core_cases": int(data["COMPLETE_CORE_CASE"].sum()),
        "complete_commute_cases": int(data["COMPLETE_COMMUTE_CASE"].sum()),
        "work_from_home_records": int(data["JWTRNS"].eq(11).sum()),
        "weighted_population": int(data["PWGTP"].sum()),
        "weighted_wage_median": wage_median,
        "missing_by_feature": {
            column: int(data[column].isna().sum())
            for column in [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
        },
        "population_filter": "STATE=53, AGEP>=16, ESR in {1,2,4,5}",
        "commute_policy": (
            "JWTRNS=11 means works from home. Other blanks stay missing because the "
            "question may not apply or may not have been reported."
        ),
    }
    return data, audit


def descriptive_tables(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = data[NUMERIC_FEATURES].describe(percentiles=[0.25, 0.5, 0.75]).T
    numeric.index.name = "variable"
    numeric = numeric.reset_index()
    numeric["weighted_mean"] = [
        weighted_mean(data[column], data["PWGTP"]) for column in NUMERIC_FEATURES
    ]
    numeric["weighted_median"] = [
        weighted_median(data[column], data["PWGTP"]) for column in NUMERIC_FEATURES
    ]

    category_frames = []
    for column in [
        "SEX_LABEL",
        "EDUCATION_LEVEL",
        "MAR_LABEL",
        "COW_LABEL",
        "RAC1P_LABEL",
        "JWTRNS_LABEL",
    ]:
        frame = (
            data.assign(category=data[column].astype("string").fillna("Missing or unknown"))
            .groupby("category", dropna=False)
            .agg(sample_count=("PERSON_ID", "size"), weighted_population=("PWGTP", "sum"))
            .reset_index()
        )
        frame["weighted_percent"] = (
            100 * frame["weighted_population"] / frame["weighted_population"].sum()
        )
        frame.insert(0, "variable", column)
        category_frames.append(frame)
    return numeric.round(2), pd.concat(category_frames, ignore_index=True).round(2)


def data_dictionary(data: pd.DataFrame) -> pd.DataFrame:
    meanings = {
        "AGEP": "Age",
        "WAGP": "Annual wage and salary income",
        "WKHP": "Usual hours worked per week",
        "JWMNP": "Travel time to work in minutes",
        "WKWN": "Weeks worked in the past 12 months",
        "SEX": "Sex",
        "SCHL": "Educational attainment",
        "MAR": "Marital status",
        "COW": "Class of worker",
        "RAC1P": "Race",
        "JWTRNS": "Commute mode",
    }
    rows = []
    for column in [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]:
        rows.append(
            {
                "variable": column,
                "kind": "numerical" if column in NUMERIC_FEATURES else "categorical",
                "meaning": meanings[column],
                "missing_count": int(data[column].isna().sum()),
                "unique_non_missing": int(data[column].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)
