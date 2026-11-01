#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
environment_name="washington-workforce"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is not available on PATH. Initialize Conda for this shell first." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "$environment_name"; then
  conda env update --name "$environment_name" --file "$repo_root/environment.yml" --prune
else
  conda env create --name "$environment_name" --file "$repo_root/environment.yml"
fi

conda run --name "$environment_name" python "$repo_root/scripts/extract_data.py"
conda run --name "$environment_name" python -c \
  "from backend.data import validate_input_file; print(validate_input_file('$repo_root/data/raw'))"

echo "Setup complete. Run: conda run -n $environment_name python -m frontend.app"

