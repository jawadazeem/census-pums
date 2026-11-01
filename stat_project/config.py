from pathlib import Path


DEFAULT_INPUT = Path(r"C:\Users\jawad.azeem\Downloads\csv_pwa")
DEFAULT_OUTPUT = Path("outputs/milestones-1-2")
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
