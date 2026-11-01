# Census Washington Workforce

The backend loads and cleans the ACS person data once. The Dash frontend imports these functions directly inside callbacks.

```text
backend/
  data.py
  statistics.py
  services.py
```

## Setup

Activate the Conda environment you already use, then install the backend packages:

```bash
conda activate YOUR_ENVIRONMENT
conda install -c conda-forge pandas numpy scipy
python scripts/extract_data.py
```

For a short demonstration, run:

```bash
python -m backend
```

This prints each processing step, five cleaned rows, numerical summaries, education counts, and a few statistical results. It does not start a server or create frontend files.

`backend.statistics` contains covariance, correlation, distance, normality, t-test, and F-test functions for callbacks or report work.

The dataset is stored as three archive parts under `data/raw/`. `extract_data.py` joins them, verifies the archive, and restores the original CSV.
