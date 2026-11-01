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
python -m frontend.app
```

Open port 8050 in Codespaces.

The original ACS CSV is stored as `data/raw/psam_p53.csv.zip` because the uncompressed file exceeds GitHub's 100 MB limit. The setup script extracts it and verifies its SHA-256 hash.

