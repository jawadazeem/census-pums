# Washington Workforce Backend

The backend loads and cleans the ACS person data once. The Dash frontend imports these functions directly inside callbacks. There is no backend server to run.

```text
backend/
  data.py
  statistics.py
  services.py
```

## Prepare the VM

Activate the Conda environment you already use, then install the backend packages:

```bash
conda activate YOUR_ENVIRONMENT
conda install -c conda-forge pandas numpy scipy
python scripts/extract_data.py
```

Check that the backend loads the complete cleaned DataFrame:

```bash
python -c "from backend.services import load_dashboard_data; workers = load_dashboard_data(); print(workers.shape)"
```

Expected row count:

```text
(180301, ...)
```

For a short instructor-facing demonstration, run:

```bash
python -m backend
```

This prints each processing step, five cleaned rows, numerical summaries, education counts, and a few statistical results. It does not start a server or create frontend files.

## Use from Dash

The frontend engineer can import the backend directly:

```python
from backend.services import load_dashboard_data, filter_workers

workers = load_dashboard_data()

def update_wage_chart(age_range, education):
    filtered = filter_workers(
        workers,
        age_range=age_range,
        education_levels=education,
    )
    return px.histogram(
        filtered,
        x="WAGP",
        color="EDUCATION_LEVEL",
    )
```

`backend.statistics` contains covariance, correlation, distance, normality, t-test, and F-test functions for callbacks or report work.

The dataset is stored as three archive parts under `data/raw/`. `extract_data.py` joins them, verifies the archive, and restores the original CSV.
