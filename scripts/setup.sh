#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$repo_root/scripts/extract_data.py"
python -m pip install --upgrade pip
python -m pip install -r "$repo_root/requirements.txt"
python -c "from backend.data import validate_input_file; print(validate_input_file('$repo_root/data/raw'))"

echo "Setup complete."

