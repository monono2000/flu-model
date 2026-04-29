from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import SimulationConfig, load_config, save_config
from .data_loader import load_inputs
from .metrics import build_simulation_tables, save_tables
from .result_layout import ensure_output_layout
from .simulation import run_simulation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Influenza simulation CLI")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to a YAML config file")
    parser.add_argument("--mode", choices=["fixed", "calendar", "legacy_batch"], help="Override run mode")
    parser.add_argument("--regime", choices=["I", "II", "III", "IV"], help="Override regime for fixed mode")
    parser.add_argument("--days", type=int, help="Override number of days for fixed mode")
    parser.add_argument("--run-name", help="Override output run directory name")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument(
        "--compare-initial-conditions",
        action="store_true",
        help="Run both same_prevalence and equal_absolute_seed under one result directory",
    )
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    default_config_path: str | None = None,
    default_mode: str | None = None,
) -> int:
    args = parse_args(argv)
    config_path = default_config_path or args.config
    project_root = Path.cwd()
    config = load_config(config_path, project_root=project_root)
    apply_cli_overrides(config, args, default_mode=default_mode)

    if config.run.compare_initial_conditions:
        if config.run.mode == "legacy_batch":
            raise ValueError("compare_initial_conditions does not support legacy_batch mode.")

        from .initial_condition_batch import run_initial_condition_batch

        require_calendar = config.run.mode == "calendar"
        print("[1/4] Loading inputs")
        inputs = load_inputs(config, require_calendar=require_calendar)
        print("[2/4] Running initial condition comparison")
        output_dir = run_initial_condition_batch(config, inputs)
        print("[3/4] Writing comparison outputs")
        print("[4/4] Done")
        print(f"Done: {output_dir}")
        return 0

    if config.run.mode == "legacy_batch":
        from .legacy import run_legacy_batch

        print("[1/3] Loading legacy inputs")
        inputs = load_inputs(config, require_calendar=False)
        print("[2/3] Running legacy batch")
        output_dir = run_legacy_batch(config, inputs)
        print(f"[3/3] Legacy outputs written to: {output_dir}")
        return 0

    require_calendar = config.run.mode == "calendar"
    print("[1/4] Loading inputs")
    inputs = load_inputs(config, require_calendar=require_calendar)
    print("[2/4] Running simulation")
    result = run_simulation(config, inputs)
    print("[3/4] Building result tables")
    tables = build_simulation_tables(result)
    run_dir = config.paths.results_dir / config.run.run_name
    save_tables(tables, run_dir)
    layout = ensure_output_layout(run_dir)
    save_config(config, layout.meta_dir / "config_used.yaml")

    if config.run.create_plots:
        from .plotting import create_all_plots

        print("[4/4] Generating plots")
        create_all_plots(result, tables, run_dir)
    else:
        print("[4/4] Skipping plot generation")

    print(f"Done: {run_dir}")
    return 0


def apply_cli_overrides(
    config: SimulationConfig,
    args: argparse.Namespace,
    default_mode: str | None = None,
) -> None:
    if default_mode is not None and args.mode is None:
        config.run.mode = default_mode
    elif args.mode is not None:
        config.run.mode = args.mode

    if args.regime is not None:
        config.run.fixed_regime = args.regime
    if args.days is not None:
        config.run.days = int(args.days)
    if args.run_name is not None:
        config.run.run_name = args.run_name
    if args.no_plots:
        config.run.create_plots = False
    if args.compare_initial_conditions:
        config.run.compare_initial_conditions = True

    config.apply_counterfactual_flags()
    config.validate()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
