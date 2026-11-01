#!/usr/bin/env python3
"""Run the first two milestones of the Washington ACS project."""

from __future__ import annotations

import sys
from pathlib import Path


LOCAL_DEPS = Path(__file__).resolve().parent / ".deps"
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))

from stat_project.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
