"""Backend analysis for the Washington ACS PUMS project."""

import sys
from pathlib import Path


local_dependencies = Path(__file__).resolve().parent.parent / ".deps"
if local_dependencies.is_dir():
    sys.path.insert(0, str(local_dependencies))

