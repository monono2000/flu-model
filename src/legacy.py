from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .constants import AGE_GROUPS, LEGACY_SCENARIOS, REGIONS
from .data_loader import LoadedInputs
from .metrics import safe_rate
from .plotting import save_grouped_bar_chart_image, save_line_chart_image
from .simulation import SimulationResult, run_simulation

if TYPE_CHECKING:
    from .config import SimulationConfig


def run_legacy_batch(config: "SimulationConfig", inputs: LoadedInputs) -> Path:
    output_dir = config.paths.legacy_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_order = [scenario for scenario in config.legacy.scenario_order if scenario in LEGACY_SCENARIOS]
    scenario_results: dict[str, SimulationResult] = {}
    for scenario in scenario_order:
        scenario_config = copy.deepcopy(config)
        scenario_config.run.mode = "fixed"
        scenario_config.run.fixed_regime = scenario
        scenario_results[scenario] = run_simulation(scenario_config, inputs)

    summary_df = build_legacy_summary_table(scenario_results)
    summary_df.to_csv(output_dir / "summary_table.csv", index=False, encoding="utf-8-sig")

    save_legacy_matrices(inputs, output_dir, scenario_order)
    plot_total_infectious_legacy(scenario_results, output_dir)
    plot_age_cumulative_legacy(scenario_results, output_dir)
    return output_dir


def build_legacy_summary_table(scenario_results: dict[str, SimulationResult]) -> pd.DataFrame:
    rows = []
    scenario_name_map = {
        "I": "Semester Weekday",
        "II": "Vacation Weekday",
        "III": "Weekend",
        "IV": "Lunar New Year",
    }
    for region in REGIONS:
        for scenario in scenario_results:
            result = scenario_results[scenario]
            summary = summarize_legacy_region(result, region)
            rows.append(
                {
                    "City": region,
                    "Scenario": scenario,
                    "Scenario_Name": scenario_name_map[scenario],
                    "Peak_Day": summary["peak_day"],
                    "Peak_I_per_100k": round(summary["peak_I_per_100k"], 2),
                    "Cumulative_Rate_%": round(summary["cumulative_rate_total"], 2),
                    "Elderly_Share_%": round(summary["elderly_share"], 2),
                    "Age_0_18_%": round(float(summary["cumulative_rate_by_age"][0]), 2),
                    "Age_19_49_%": round(float(summary["cumulative_rate_by_age"][1]), 2),
                    "Age_50_64_%": round(float(summary["cumulative_rate_by_age"][2]), 2),
                    "Age_65plus_%": round(float(summary["cumulative_rate_by_age"][3]), 2),
                }
            )
    return pd.DataFrame(rows)


def save_legacy_matrices(inputs: LoadedInputs, output_dir: Path, scenario_order: list[str]) -> None:
    for scenario in scenario_order:
        pd.DataFrame(
            inputs.age_matrices[scenario],
            index=AGE_GROUPS,
            columns=AGE_GROUPS,
        ).to_csv(output_dir / f"age_matrix_{scenario}.csv", encoding="utf-8-sig")
        pd.DataFrame(
            inputs.region_matrices[scenario],
            index=REGIONS,
            columns=REGIONS,
        ).to_csv(output_dir / f"region_matrix_{scenario}.csv", encoding="utf-8-sig")


def plot_total_infectious_legacy(scenario_results: dict[str, SimulationResult], output_dir: Path) -> None:
    for scenario, result in scenario_results.items():
        x = result.day_numbers
        active_history = result.state_history["I0"] + result.state_history["I1"]
        series = []
        for region_idx, region in enumerate(REGIONS):
            per_100k = np.divide(
                active_history[:, region_idx, :].sum(axis=1),
                result.population[region_idx].sum(),
                out=np.zeros_like(x, dtype=float),
                where=result.population[region_idx].sum() > 0.0,
            ) * 100000.0
            color = "#1f77b4" if region == "Region_A" else "#c0392b"
            series.append((region, per_100k, color))

        save_line_chart_image(
            output_path=output_dir / f"total_infectious_{scenario}.png",
            title=f"Legacy Fixed Regime {scenario}: Region_A vs Region_B",
            x_values=x,
            series=series,
            y_label="Infectious per 100,000",
            x_label="Day",
        )


def plot_age_cumulative_legacy(scenario_results: dict[str, SimulationResult], output_dir: Path) -> None:
    for scenario, result in scenario_results.items():
        region_a_summary = summarize_legacy_region(result, "Region_A")
        region_b_summary = summarize_legacy_region(result, "Region_B")
        save_grouped_bar_chart_image(
            output_path=output_dir / f"age_cumulative_{scenario}.png",
            title=f"Legacy Fixed Regime {scenario}: Age-specific cumulative infection rate",
            categories=list(AGE_GROUPS),
            series=[
                ("Region_A", np.asarray(region_a_summary["cumulative_rate_by_age"]), "#1f77b4"),
                ("Region_B", np.asarray(region_b_summary["cumulative_rate_by_age"]), "#c0392b"),
            ],
            y_label="Cumulative infection rate (%)",
            highlight_last_category=True,
        )


def summarize_legacy_region(result: SimulationResult, region: str) -> dict:
    region_idx = REGIONS.index(region)
    active_history = result.state_history["I0"][:, region_idx, :] + result.state_history["I1"][:, region_idx, :]
    region_active_total = active_history.sum(axis=1)
    region_population = float(result.population[region_idx].sum())
    peak_idx = int(np.argmax(region_active_total))

    cumulative_first_final = result.population[region_idx] - result.state_history["S0"][-1, region_idx]
    cumulative_total = float(cumulative_first_final.sum())
    elderly_share = safe_rate(cumulative_first_final[-1], cumulative_total, scale=100.0)

    return {
        "peak_day": peak_idx,
        "peak_I_per_100k": safe_rate(region_active_total[peak_idx], region_population, scale=100000.0),
        "cumulative_rate_total": safe_rate(cumulative_total, region_population, scale=100.0),
        "cumulative_rate_by_age": np.divide(
            cumulative_first_final,
            result.population[region_idx],
            out=np.zeros_like(cumulative_first_final),
            where=result.population[region_idx] > 0.0,
        )
        * 100.0,
        "elderly_share": elderly_share,
    }
