from __future__ import annotations

import copy
import json
import math
from itertools import product
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SimulationConfig, save_config
from .constants import AGE_GROUPS
from .data_loader import load_inputs
from .metrics import build_simulation_tables, save_tables
from .observations import filter_weekly_window, load_observed_influenza_weekly, melt_weekly_rates
from .simulation import run_simulation


@dataclass
class CalibrationSearchSpace:
    beta0_values: list[float] = field(
        default_factory=lambda: [0.04, 0.05, 0.055, 0.06, 0.07]
    )
    same_prevalence_values: list[float] = field(
        default_factory=lambda: [0.0001, 0.0003, 0.0005, 0.001, 0.002]
    )
    multiplier_I_values: list[float] = field(default_factory=lambda: [0.8, 1.0, 1.2])
    multiplier_II_values: list[float] = field(default_factory=lambda: [1.0, 1.2, 1.4, 1.6, 1.8])
    multiplier_III_values: list[float] = field(default_factory=lambda: [0.8, 1.0, 1.2, 1.4])
    multiplier_IV_values: list[float] = field(default_factory=lambda: [1.0, 1.2, 1.4, 1.6])


@dataclass
class SusceptibilitySearchSpace:
    age_0_18_values: list[float] = field(default_factory=lambda: [0.7, 0.85, 1.0, 1.15, 1.3])
    age_19_49_values: list[float] = field(default_factory=lambda: [0.8, 0.95, 1.1, 1.25])
    age_50_64_values: list[float] = field(default_factory=lambda: [0.8, 0.95, 1.1, 1.25])
    age_65_plus_values: list[float] = field(default_factory=lambda: [0.8, 1.0, 1.2, 1.4])


def aggregate_simulation_to_weekly_age_rates(result: SimulationResult) -> pd.DataFrame:
    population_by_age = result.population.sum(axis=0)
    daily_new = result.new_first_infections + result.new_reinfections
    rows: list[dict] = []
    for day_idx, label in enumerate(result.daily_labels):
        date_value = pd.to_datetime(label)
        week_start = (date_value - pd.Timedelta(days=date_value.weekday())).strftime("%Y-%m-%d")
        for age_idx, age_group in enumerate(AGE_GROUPS):
            daily_rate = (
                float(daily_new[day_idx, :, age_idx].sum()) / float(population_by_age[age_idx]) * 100000.0
                if population_by_age[age_idx] > 0.0
                else 0.0
            )
            rows.append(
                {
                    "week_start": week_start,
                    "age_group": age_group,
                    "daily_episode_rate_per_100k": daily_rate,
                }
            )
    weekly_df = pd.DataFrame(rows)
    weekly_df = (
        weekly_df.groupby(["week_start", "age_group"], as_index=False)["daily_episode_rate_per_100k"]
        .sum()
        .rename(columns={"daily_episode_rate_per_100k": "model_rate"})
    )
    return weekly_df


def score_calibration(
    observed_long: pd.DataFrame,
    model_long: pd.DataFrame,
) -> float:
    merged = observed_long.merge(model_long, on=["week_start", "age_group"], how="inner")
    total_score = 0.0
    for age_group in AGE_GROUPS:
        subset = merged[merged["age_group"] == age_group]
        observed = subset["observed_rate"].to_numpy(dtype=float)
        modeled = subset["model_rate"].to_numpy(dtype=float)
        if observed.size == 0 or modeled.size == 0 or modeled.max() <= 0.0:
            return 1.0e18

        observed_norm = observed / observed.max()
        modeled_norm = modeled / modeled.max()
        total_score += float(np.mean((observed_norm - modeled_norm) ** 2))
        total_score += 0.20 * abs(int(np.argmax(observed)) - int(np.argmax(modeled)))
        total_score += 0.03 * abs(math.log((modeled.max() + 1.0e-9) / (observed.max() + 1.0e-9)))
    return total_score


def prepare_observed_long(
    observed_csv_path: str | Path,
    compare_start_date: str,
    compare_end_date: str,
) -> pd.DataFrame:
    observed_weekly = load_observed_influenza_weekly(observed_csv_path)
    observed_window = filter_weekly_window(
        observed_weekly,
        start_date=compare_start_date,
        end_date=compare_end_date,
    )
    return melt_weekly_rates(observed_window, value_name="observed_rate")


def evaluate_config_against_observed(
    trial_config: SimulationConfig,
    observed_long: pd.DataFrame,
    inputs,
) -> dict:
    trial_result = run_simulation(trial_config, inputs)
    model_long = aggregate_simulation_to_weekly_age_rates(trial_result)
    score = score_calibration(observed_long=observed_long, model_long=model_long)
    return {
        "score": score,
        "config": trial_config,
        "result": trial_result,
        "model_long": model_long,
        "observed_long": observed_long,
    }


def run_calendar_grid_calibration(
    base_config: SimulationConfig,
    observed_csv_path: str | Path,
    search_space: CalibrationSearchSpace | None = None,
    compare_start_date: str = "2025-12-01",
    compare_end_date: str = "2026-02-23",
) -> dict:
    search_space = search_space or CalibrationSearchSpace()
    if base_config.run.mode != "calendar":
        raise ValueError("관측 보정은 calendar mode 설정을 기준으로 실행해야 합니다.")

    observed_long = prepare_observed_long(
        observed_csv_path=observed_csv_path,
        compare_start_date=compare_start_date,
        compare_end_date=compare_end_date,
    )
    inputs = load_inputs(base_config, require_calendar=True)

    best_record: dict | None = None
    for beta0 in search_space.beta0_values:
        for same_prevalence in search_space.same_prevalence_values:
            for mult_i in search_space.multiplier_I_values:
                for mult_ii in search_space.multiplier_II_values:
                    for mult_iii in search_space.multiplier_III_values:
                        for mult_iv in search_space.multiplier_IV_values:
                            trial_config = copy.deepcopy(base_config)
                            trial_config.model.beta0 = beta0
                            trial_config.model.beta_multiplier = {
                                "I": mult_i,
                                "II": mult_ii,
                                "III": mult_iii,
                                "IV": mult_iv,
                            }
                            trial_config.initial_conditions.mode = "same_prevalence"
                            trial_config.initial_conditions.same_prevalence = same_prevalence
                            trial_config.initial_conditions.same_prevalence_by_age = {}
                            record = evaluate_config_against_observed(
                                trial_config=trial_config,
                                observed_long=observed_long,
                                inputs=inputs,
                            )
                            if best_record is None or record["score"] < best_record["score"]:
                                best_record = record

    if best_record is None:
        raise RuntimeError("grid calibration 결과를 찾지 못했습니다.")
    best_record["compare_start_date"] = compare_start_date
    best_record["compare_end_date"] = compare_end_date
    return best_record


def refine_susceptibility_search(
    seed_record: dict,
    observed_csv_path: str | Path,
    compare_start_date: str,
    compare_end_date: str,
    search_space: SusceptibilitySearchSpace | None = None,
) -> dict:
    search_space = search_space or SusceptibilitySearchSpace()
    base_config = seed_record["config"]
    observed_long = prepare_observed_long(
        observed_csv_path=observed_csv_path,
        compare_start_date=compare_start_date,
        compare_end_date=compare_end_date,
    )
    inputs = load_inputs(base_config, require_calendar=True)

    best_record = seed_record
    for age_0_18, age_19_49, age_50_64, age_65_plus in product(
        search_space.age_0_18_values,
        search_space.age_19_49_values,
        search_space.age_50_64_values,
        search_space.age_65_plus_values,
    ):
        trial_config = copy.deepcopy(base_config)
        trial_config.model.susceptibility = {
            "0-18": float(age_0_18),
            "19-49": float(age_19_49),
            "50-64": float(age_50_64),
            "65+": float(age_65_plus),
        }
        record = evaluate_config_against_observed(
            trial_config=trial_config,
            observed_long=observed_long,
            inputs=inputs,
        )
        if record["score"] < best_record["score"]:
            best_record = record

    best_record["compare_start_date"] = compare_start_date
    best_record["compare_end_date"] = compare_end_date
    return best_record


def save_calibration_outputs(
    run_dir: str | Path,
    best_record: dict,
) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    result = best_record["result"]
    tables = build_simulation_tables(result)
    save_tables(tables, run_dir)
    save_config(best_record["config"], run_dir / "calibrated_config.yaml")
    best_record["observed_long"].to_csv(run_dir / "target_weekly_long.csv", index=False, encoding="utf-8-sig")
    best_record["model_long"].to_csv(run_dir / "model_weekly_long.csv", index=False, encoding="utf-8-sig")
    comparison = best_record["observed_long"].merge(
        best_record["model_long"],
        on=["week_start", "age_group"],
        how="inner",
    )
    comparison.to_csv(run_dir / "target_vs_model_weekly.csv", index=False, encoding="utf-8-sig")
    summary = {
        "is_synthetic_target": True,
        "score": best_record["score"],
        "beta0": best_record["config"].model.beta0,
        "same_prevalence": best_record["config"].initial_conditions.same_prevalence,
        "beta_multiplier": best_record["config"].model.beta_multiplier,
        "susceptibility": best_record["config"].model.susceptibility,
        "compare_start_date": best_record.get("compare_start_date"),
        "compare_end_date": best_record.get("compare_end_date"),
        "submission_note": (
            "This calibration uses a manually prepared synthetic weekly target series. "
            "It is not official surveillance data and is intended only for demonstration "
            "and course-project experimentation."
        ),
        "note": (
            "관측 CSV는 2025-W36~2026-W16의 주간 연령별 발병률로 해석했고, "
            "비교 구간은 calibration 실행 시 지정한 주차 창을 사용했다. "
            "관측치는 surveillance rate, 모델은 simulated infection episode rate이므로 "
            "shape 중심 비교로 보정했다."
        ),
    }
    (run_dir / "calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
