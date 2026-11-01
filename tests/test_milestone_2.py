import unittest

import pandas as pd

from stat_project.statistics import hypothesis_tests


class StatisticalTests(unittest.TestCase):
    def test_hypothesis_table_contains_both_required_tests(self):
        data = pd.DataFrame(
            {
                "SEX": [1, 1, 1, 2, 2, 2],
                "WAGP": [80_000, 70_000, 90_000, 50_000, 45_000, 55_000],
            }
        )

        results = hypothesis_tests(data)

        self.assertEqual(
            results["test"].tolist(),
            ["Welch two-sample t-test", "Two-sided F-test for variance"],
        )
        self.assertTrue(results["p_value"].notna().all())


if __name__ == "__main__":
    unittest.main()
