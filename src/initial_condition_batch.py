from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd

from .config import SimulationConfig, save_config
from .data_loader import LoadedInputs
from .metrics import build_simulation_tables, save_tables
from .result_layout import ensure_output_layout, write_run_manifest
from .simulation import SimulationResult, run_simulation


INITIAL_CONDITION_BATCH_MODES = ("same_prevalence", "equal_absolute_seed")


def run_initial_condition_batch(config: SimulationConfig, inputs: LoadedInputs) -> Path:
    root_dir = config.paths.results_dir / config.run.run_name
    layout = ensure_output_layout(root_dir)
    save_config(config, layout.meta_dir / "batch_config_used.yaml")

    overall_rows: list[dict] = []
    region_group_frames: list[pd.DataFrame] = []
    age_group_frames: list[pd.DataFrame] = []
    run_sections: dict[str, str] = {}

    for mode in INITIAL_CONDITION_BATCH_MODES:
        scenario_config = build_initial_condition_batch_config(config, mode)
        scenario_dir = root_dir / mode
        result = run_simulation(scenario_config, inputs)
        tables = build_simulation_tables(result)
        save_tables(tables, scenario_dir)

        scenario_layout = ensure_output_layout(scenario_dir)
        save_config(scenario_config, scenario_layout.meta_dir / "config_used.yaml")

        if scenario_config.run.create_plots:
            from .plotting import create_all_plots

            create_all_plots(result, tables, scenario_dir)

        overall_rows.append(build_initial_condition_overall_summary(mode, result, tables.overall_daily_metrics))
        region_group_frames.append(_with_initial_condition_column(tables.region_group_final_summary, mode))
        age_group_frames.append(_with_initial_condition_column(tables.age_group_summary, mode))
        run_sections[mode] = f"{mode}/"

    pd.DataFrame(overall_rows).to_csv(
        layout.tables_dir / "initial_condition_overall_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(region_group_frames, ignore_index=True).to_csv(
        layout.tables_dir / "region_group_final_summary_by_initial_condition.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(age_group_frames, ignore_index=True).to_csv(
        layout.tables_dir / "age_group_summary_by_initial_condition.csv",
        index=False,
        encoding="utf-8-sig",
    )

    batch_summary = {
        "initial_condition_modes": list(INITIAL_CONDITION_BATCH_MODES),
        "run_directories": run_sections,
    }
    (layout.meta_dir / "batch_summary.json").write_text(
        json.dumps(batch_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_run_manifest(
        layout,
        run_type="initial_condition_batch",
        primary_files={
            "batch_config": "meta/batch_config_used.yaml",
            "overall_summary": "tables/initial_condition_overall_summary.csv",
            "region_group_summary": "tables/region_group_final_summary_by_initial_condition.csv",
        },
        extra_sections={
            "tables": {
                "age_group_summary": "tables/age_group_summary_by_initial_condition.csv",
            },
            "runs": run_sections,
        },
    )
    return layout.root


def build_initial_condition_batch_config(
    base_config: SimulationConfig,
    initial_condition_mode: str,
) -> SimulationConfig:
    if initial_condition_mode not in INITIAL_CONDITION_BATCH_MODES:
        raise ValueError(f"Unsupported initial condition batch mode: {initial_condition_mode}")

    scenario_config = copy.deepcopy(base_config)
    scenario_config.initial_conditions.mode = initial_condition_mode
    scenario_config.run.run_name = initial_condition_mode
    scenario_config.run.compare_initial_conditions = False
    scenario_config.apply_counterfactual_flags()
    scenario_config.validate()
    return scenario_config


def build_initial_condition_overall_summary(
    initial_condition_mode: str,
    result: SimulationResult,
    overall_daily_metrics: pd.DataFrame,
) -> dict:
    final_row = overall_daily_metrics.iloc[-1]
    peak_row = overall_daily_metrics.loc[
        overall_daily_metrics["total_active_infected"].idxmax()
    ]
    initial_seed_total = float(result.state_history["I0"][0].sum() + result.state_history["E0"][0].sum())
    return {
        "initial_condition_mode": initial_condition_mode,
        "seed_compartment": result.config.initial_conditions.seed_compartment,
        "initial_seed_total": initial_seed_total,
        "peak_day": peak_row["date_or_day"],
        "peak_active_infected": float(peak_row["total_active_infected"]),
        "peak_active_infected_per_100k": float(peak_row["total_active_infected_per_100k"]),
        "final_cumulative_first_infections": float(final_row["cumulative_first_infections"]),
        "final_cumulative_reinfections": float(final_row["cumulative_reinfections"]),
        "final_cumulative_total_infection_episodes": float(final_row["cumulative_total_infection_episodes"]),
        "final_total_active_infected": float(final_row["total_active_infected"]),
        "final_total_active_infected_per_100k": float(final_row["total_active_infected_per_100k"]),
        "final_elderly_ever_infected_rate": float(final_row["elderly_ever_infected_rate"]),
        "final_cross_region_flow_share": float(final_row["cross_region_flow_share"]),
    }


def _with_initial_condition_column(frame: pd.DataFrame, initial_condition_mode: str) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "initial_condition_mode", initial_condition_mode)
    return output
