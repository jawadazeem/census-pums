# Washington Person Data Audit

## Files found

- `C:\Users\jawad.azeem\Downloads\csv_pwa\ACS2019_2023_PUMS_README.pdf`
- `C:\Users\jawad.azeem\Downloads\csv_pwa\psam_p53.csv`

## CSV findings

- File size: 250,978,600 bytes
- Raw rows: 384,503
- Columns: 286
- Record type: every row has `RT=P`, meaning person records
- State: every row has `STATE=53`, meaning Washington
- Source years represented in serial numbers: 2019 through 2023
- Available survey weight: `PWGTP`, the person weight
- All selected project variables are present
- Rows after filtering to employed respondents age 16+: 180,301
- Duplicate `SERIALNO` + `SPORDER` identifiers after filtering: 0
- Missing selected values after filtering: `JWMNP` 36,552 and `JWTRNS` 4,090; all other selected variables have zero missing values
- Explicit work-from-home records (`JWTRNS=11`): 32,462
- Unweighted wage mean and median: $69,702.80 and $50,000
- ACS-weighted wage mean and median: $68,084.70 and $49,400

## Conclusion

This is the correct source file for the proposed Washington workforce analysis. Its filtered row count and missing-value counts reproduce the figures in the proposal.

Commute blanks require two interpretations: 32,462 people explicitly worked from home and correctly have no commute minutes, while 4,090 filtered records have no commute mode. The pipeline preserves that distinction rather than treating every commute blank as work from home.

