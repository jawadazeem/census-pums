# Washington ACS Backend

This is the active backend for Milestones 1 and 2 of the CS/STAT 3654 project. The later regression, PCA, classification, and clustering work is preserved separately in:

`C:\Users\jawad.azeem\Documents\misc\stat-project-backend-complete-reference`

## Data

The verified source is `C:\Users\jawad.azeem\Downloads\csv_pwa\psam_p53.csv`, the 2019-2023 ACS 5-Year PUMS person file for Washington. It has 384,503 rows. Filtering to employed Washington respondents age 16 or older produces 180,301 rows.

## Milestone 1

Milestone 1 validates and cleans the source, decodes categories, handles commute blanks, and exports descriptive statistics.

```powershell
python -m pip install -r requirements.txt
python backend_pipeline.py --through milestone1 --output .\outputs\milestone-1
```

The Git tag `milestone-1` identifies the November 1 release.

## Milestone 2

Milestone 2 reruns Milestone 1, then adds covariance, correlation, distance metrics, normality testing, transformations, Q-Q data, a t-test, an F-test, bias notes, and written observations.

```powershell
python -m pip install -r requirements-milestone2.txt
python backend_pipeline.py --through milestone2 --output .\outputs\milestones-1-2
```

## Other commands

```powershell
# Validate the source without running the pipeline
python backend_pipeline.py --validate-only

# Run all current tests
python -m unittest discover -s tests -v
```

## Structure

- `backend_pipeline.py` is the command-line entry point.
- `stat_project/data.py` owns validation, cleaning, labels, weighting, and descriptive tables.
- `stat_project/statistics.py` owns Milestone 2 calculations.
- `stat_project/pipeline.py` writes Milestone 1 outputs.
- `stat_project/milestone2.py` adds Milestone 2 outputs.
- `stat_project/cli.py` selects the milestone cutoff.

The active code intentionally stops at Milestone 2. Later modeling work should return only after these results have been reviewed.
