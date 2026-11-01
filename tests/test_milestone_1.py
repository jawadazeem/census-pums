import unittest
from pathlib import Path

import pandas as pd

from stat_project.data import (
    DataValidationError,
    clean_person_data,
    collapse_education,
    resolve_input_path,
    validate_input_file,
    weighted_median,
)


TEST_DATA = Path(__file__).parent / "_tmp"
TEST_DATA.mkdir(exist_ok=True)


class InputValidationTests(unittest.TestCase):
    def test_resolves_person_file_from_folder(self):
        folder = TEST_DATA / "folder_input"
        folder.mkdir(exist_ok=True)
        expected = folder / "psam_p53.csv"
        expected.write_text("RT,STATE\nP,53\n", encoding="utf-8")

        self.assertEqual(resolve_input_path(folder), expected)

    def test_rejects_housing_file(self):
        path = TEST_DATA / "psam_h25.csv"
        pd.DataFrame(
            {"RT": ["H"], "SERIALNO": ["2023HU1"], "STATE": [25]}
        ).to_csv(path, index=False)

        with self.assertRaisesRegex(DataValidationError, "missing required person fields"):
            validate_input_file(path)

    def test_accepts_washington_person_file(self):
        path = TEST_DATA / "psam_p53.csv"
        pd.DataFrame([person_row()]).to_csv(path, index=False)

        metadata = validate_input_file(path)

        self.assertEqual(metadata["record_type"], "P")
        self.assertEqual(metadata["state"], 53)
        self.assertEqual(metadata["years"], [2023])


class CleaningTests(unittest.TestCase):
    def test_collapses_education_codes(self):
        values = pd.Series([1, 16, 18, 20, 21, 22, 24, pd.NA])

        result = collapse_education(values).astype("string").tolist()

        self.assertEqual(
            result,
            [
                "Less than high school",
                "High school or GED",
                "Some college",
                "Associate degree",
                "Bachelor's degree",
                "Graduate degree",
                "Graduate degree",
                pd.NA,
            ],
        )

    def test_weighted_median_uses_person_weights(self):
        self.assertEqual(
            weighted_median(pd.Series([10, 20, 30]), pd.Series([1, 1, 8])),
            30,
        )

    def test_filter_and_commute_handling(self):
        rows = pd.DataFrame(
            [
                person_row(SPORDER=1, AGEP=35, WAGP=40_000, JWTRNS=1, JWMNP=30),
                person_row(SPORDER=2, AGEP=42, WAGP=90_000, JWTRNS=11, JWMNP=pd.NA),
                person_row(SPORDER=3, AGEP=15),
                person_row(SPORDER=4, AGEP=51, ESR=6),
            ]
        )

        cleaned, audit = clean_person_data(rows)

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(audit["rows_after_population_filter"], 2)
        self.assertEqual(cleaned.loc[1, "COMMUTE_STATUS"], "Works from home")
        self.assertTrue(pd.isna(cleaned.loc[1, "JWMNP"]))


def person_row(**overrides):
    row = {
        "RT": "P",
        "SERIALNO": "2023P0001",
        "SPORDER": 1,
        "STATE": 53,
        "PWGTP": 1,
        "ESR": 1,
        "AGEP": 30,
        "WAGP": 50_000,
        "WKHP": 40,
        "JWMNP": 20,
        "WKWN": 52,
        "SEX": 1,
        "SCHL": 21,
        "MAR": 1,
        "COW": 1,
        "RAC1P": 1,
        "JWTRNS": 1,
    }
    row.update(overrides)
    return row


if __name__ == "__main__":
    unittest.main()
