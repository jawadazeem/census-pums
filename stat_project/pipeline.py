from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    clean_person_data,
    data_dictionary,
    descriptive_tables,
    load_source,
    validate_input_file,
)


def run_milestone_1(
    input_path: Path, output_dir: Path, random_state: int
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = validate_input_file(input_path)
    cleaned, audit = clean_person_data(load_source(metadata))
    numeric, categorical = descriptive_tables(cleaned)

    write_json(output_dir / "input_metadata.json", metadata)
    write_json(output_dir / "cleaning_audit.json", audit)
    cleaned.to_csv(output_dir / "cleaned_person_data.csv", index=False)
    numeric.to_csv(output_dir / "descriptive_numeric.csv", index=False)
    categorical.to_csv(output_dir / "categorical_frequencies.csv", index=False)
    data_dictionary(cleaned).to_csv(output_dir / "data_dictionary.csv", index=False)
    write_observations(output_dir, audit, numeric)

    manifest = make_manifest(
        through="milestone1",
        input_path=str(metadata["path"]),
        output_dir=output_dir,
        random_state=random_state,
        cleaned_rows=len(cleaned),
    )
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def make_manifest(
    through: str,
    input_path: str,
    output_dir: Path,
    random_state: int,
    cleaned_rows: int,
) -> dict[str, object]:
    files = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    if "run_manifest.json" not in files:
        files.append("run_manifest.json")
        files.sort()
    return {
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "through": through,
        "input": input_path,
        "output": str(output_dir.resolve()),
        "random_state": random_state,
        "cleaned_rows": cleaned_rows,
        "files": files,
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, default=json_value),
        encoding="utf-8",
    )


def json_value(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_observations(
    output_dir: Path, audit: dict[str, object], numeric: pd.DataFrame
) -> None:
    wage = numeric.set_index("variable").loc["WAGP"]
    text = f"""# Milestone 1 observations

The source contains {audit['rows_raw']:,} person records. Filtering to employed Washington residents age 16 or older leaves {audit['rows_after_population_filter']:,} records, with no duplicate person identifiers.

All core project fields are complete. Commute time is blank for {audit['missing_by_feature']['JWMNP']:,} records, including {audit['work_from_home_records']:,} people explicitly coded as working from home. Another {audit['missing_by_feature']['JWTRNS']:,} records have no commute mode, so those values remain missing rather than being relabeled.

Annual wages are right-skewed. The unweighted mean is ${wage['mean']:,.2f}, compared with a median of ${wage['50%']:,.2f}. The ACS-weighted mean is ${wage['weighted_mean']:,.2f}, and the weighted median is ${wage['weighted_median']:,.2f}.
"""
    (output_dir / "milestone_1_observations.md").write_text(text, encoding="utf-8")
