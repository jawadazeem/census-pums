# Washington Workforce Dashboard

This is one Python Dash application. The backend loads and cleans the ACS person data. Dash callbacks filter that DataFrame and return Plotly figures.

```text
backend/
  data.py
  statistics.py
  services.py

frontend/
  app.py
  callbacks.py
  layouts.py
```

## Setup

```bash
bash scripts/setup.sh
```

## Run

```bash
conda run -n washington-workforce python -m frontend.app
```

Open http://127.0.0.1:8050.

The original ACS CSV is stored as three archive parts under `data/raw/`. The setup script joins them, verifies the archive, and extracts the original CSV.

