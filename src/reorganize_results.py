from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .results_migration import migrate_results_tree, migrate_run_directory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reorganize result folders into meta/tables/plots layout.")
    parser.add_argument("--results-dir", default="results", help="Results root directory")
    parser.add_argument("--run-name", help="Optional single run directory name to reorganize")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir = Path(args.results_dir)
    if args.run_name:
        migrate_run_directory(results_dir / args.run_name)
    else:
        migrate_results_tree(results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
