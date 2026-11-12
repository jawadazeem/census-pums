from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path(os.getenv("ACS_DATA_PATH", ROOT / "data" / "raw"))

NUMERIC_FEATURES = ["AGEP", "WAGP", "WKHP", "JWMNP", "WKWN"]
CATEGORICAL_FEATURES = ["SEX", "SCHL", "MAR", "COW", "RAC1P", "JWTRNS"]
SOURCE_COLUMNS = [
    "RT", "SERIALNO", "SPORDER", "STATE", "PWGTP", "ESR",
    *NUMERIC_FEATURES, *CATEGORICAL_FEATURES,
]

EMPLOYED_ESR_CODES = {1, 2, 4, 5}

CATEGORY_LABELS = {
    "SEX": {1: "Male", 2: "Female"},
    "MAR": {1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated", 5: "Never married"},
    "COW": {
        1: "Private for-profit", 2: "Private nonprofit", 3: "Local government",
        4: "State government", 5: "Federal government",
        6: "Self-employed, not incorporated", 7: "Self-employed, incorporated",
        8: "Unpaid family worker",
    },
    "RAC1P": {
        1: "White alone", 2: "Black or African American alone",
        3: "American Indian alone", 4: "Alaska Native alone",
        5: "American Indian and Alaska Native", 6: "Asian alone",
        7: "Native Hawaiian or Pacific Islander alone", 8: "Some other race alone",
        9: "Two or more races",
    },
    "JWTRNS": {
        1: "Car, truck, or van", 2: "Bus", 3: "Streetcar or trolley",
        4: "Subway or elevated rail", 5: "Railroad", 6: "Ferryboat",
        7: "Taxicab", 8: "Motorcycle", 9: "Bicycle", 10: "Walked",
        11: "Worked from home", 12: "Other method",
    },
}


def resolve_input_path(input_path: str | Path = DEFAULT_INPUT) -> Path:
    path = Path(input_path)
    if path.is_file():
        return path
    expected = path / "psam_p53.csv"
    if expected.is_file():
        return expected
    raise FileNotFoundError(
        f"Could not find psam_p53.csv under {path}. "
        "Pass the CSV path or set ACS_DATA_PATH."
    )


def load_workers(input_path: str | Path = DEFAULT_INPUT) -> pd.DataFrame:
    """Read the ACS person file and return the cleaned worker DataFrame."""
    path = resolve_input_path(input_path)
    header = pd.read_csv(path, nrows=0).columns
    missing = sorted(set(SOURCE_COLUMNS) - set(header))
    if missing:
        raise ValueError("Missing ACS columns: " + ", ".join(missing))

    raw = pd.read_csv(path, usecols=SOURCE_COLUMNS, low_memory=False)
    if set(raw["RT"].dropna().astype(str)) != {"P"}:
        raise ValueError("Expected an ACS person file with RT=P.")
    if set(pd.to_numeric(raw["STATE"], errors="coerce").dropna().astype(int)) != {53}:
        raise ValueError("Expected the Washington file with STATE=53.")
    return clean_workers(raw)


def clean_workers(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.copy()
    for column in [
        *NUMERIC_FEATURES, *CATEGORICAL_FEATURES, "PWGTP", "ESR", "STATE", "SPORDER"
    ]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.drop_duplicates(["SERIALNO", "SPORDER"])
    data = data.loc[(data["AGEP"] >= 16) & data["ESR"].isin(EMPLOYED_ESR_CODES)].copy()

    data["YEAR"] = pd.to_numeric(data["SERIALNO"].astype(str).str[:4], errors="coerce")
    data["PERSON_ID"] = data["SERIALNO"].astype(str) + "-" + data["SPORDER"].astype(str)
    data["EDUCATION_LEVEL"] = collapse_education(data["SCHL"])
    for column, labels in CATEGORY_LABELS.items():
        data[f"{column}_LABEL"] = data[column].map(labels).fillna("Missing or unknown")

    data["COMMUTE_STATUS"] = np.select(
        [data["JWTRNS"].eq(11), data["JWTRNS"].notna()],
        ["Works from home", "Commutes"],
        default="Not reported or not applicable",
    )
    data["COMMUTE_MINUTES"] = data["JWMNP"]
    data.loc[data["JWTRNS"].eq(11), "COMMUTE_MINUTES"] = 0
    median_wage = weighted_median(data["WAGP"], data["PWGTP"])
    data["HIGH_EARNER"] = (data["WAGP"] > median_wage).astype(int)
    return data.reset_index(drop=True)


def collapse_education(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[0, 15, 17, 19, 20, 21, 24],
        labels=[
            "Less than high school", "High school or GED", "Some college",
            "Associate degree", "Bachelor's degree", "Graduate degree",
        ],
    )


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    frame = pd.DataFrame({"value": values, "weight": weights}).dropna()
    return float(np.average(frame["value"], weights=frame["weight"]))


def weighted_median(values: pd.Series, weights: pd.Series) -> float:
    frame = pd.DataFrame({"value": values, "weight": weights}).dropna()
    frame = frame.loc[frame["weight"] > 0].sort_values("value")
    cutoff = frame["weight"].sum() / 2
    return float(frame.loc[frame["weight"].cumsum() >= cutoff, "value"].iloc[0])
