from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DEFAULT_INPUT, DEFAULT_OUTPUT
from .data import DataValidationError, validate_input_file
from .pipeline import run_milestone_1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Milestone 1 for the Washington ACS project."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Person CSV or containing folder (default: {DEFAULT_INPUT})",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--through",
        choices=["milestone1"],
        default="milestone1",
        help="Last milestone to run",
    )
    parser.add_argument("--random-state", type=int, default=3654)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.validate_only:
            print(json.dumps(validate_input_file(args.input), indent=2))
            return 0
        manifest = run_milestone_1(args.input, args.output, args.random_state)
    except (DataValidationError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2))
    return 0
