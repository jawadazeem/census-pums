# Washington ACS Backend - Milestone 1

This release contains the data foundation for the CS/STAT 3654 project. Later exploratory work is preserved in:

`C:\Users\jawad.azeem\Documents\misc\stat-project-backend-complete-reference`

## Data

The verified source is `C:\Users\jawad.azeem\Downloads\csv_pwa\psam_p53.csv`, the 2019-2023 ACS 5-Year PUMS person file for Washington. It has 384,503 rows. Filtering to employed Washington respondents age 16 or older produces 180,301 rows.

## Run Milestone 1

```powershell
python -m pip install -r requirements.txt
python backend_pipeline.py --through milestone1 --output .\outputs\milestone-1
```

The run validates and cleans the source, decodes categories, handles commute blanks, and exports descriptive statistics.

## Validate and test

```powershell
python backend_pipeline.py --validate-only
python -m unittest discover -s tests -p "test_milestone_1.py" -v
```

## Structure

- `backend_pipeline.py` is the command-line entry point.
- `stat_project/data.py` owns validation, cleaning, labels, weighting, and descriptive tables.
- `stat_project/pipeline.py` writes Milestone 1 outputs.
- `stat_project/cli.py` handles command-line arguments.

Milestone 2 statistical analysis is planned for November 8.
