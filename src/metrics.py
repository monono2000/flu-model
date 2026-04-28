from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import AGE_GROUPS, AGE_TOKEN_MAP, NODE_ORDER, REGIONS
from .simulation import SimulationResult


@dataclass
class SimulationTables:
    states_long: pd.DataFrame
    node_daily_metrics: pd.DataFrame
    overall_daily_metrics: pd.DataFrame
    flow_long: pd.DataFrame
    summary_metrics: dict
    region_daily_metrics: pd.DataFrame
    age_group_summary: pd.DataFrame


def build_simulation_tables(result: SimulationResult) -> SimulationTables:
    states_long = build_states_long(result)
    node_daily_metrics = build_node_daily_metrics(result)
    overall_daily_metrics = build_overall_daily_metrics(result)
    flow_long = build_flow_long(result)
    summary_metrics = build_summary_metrics(result)
    region_daily_metrics = build_region_daily_metrics(result)
    age_group_summary = build_age_group_summary(result)
    return SimulationTables(
        states_long=states_long,
        node_daily_metrics=node_daily_metrics,
        overall_daily_metrics=overall_daily_metrics,
        flow_long=flow_long,
        summary_metrics=summary_metrics,
        region_daily_metrics=region_daily_metrics,
        age_group_summary=age_group_summary,
    )


def build_states_long(result: SimulationResult) -> pd.DataFrame:
    rows: list[dict] = []
    active_history = result.state_history["I0"] + result.state_history["I1"]
    susceptible_history = result.state_history["S0"] + result.state_history["S1"]

    for time_idx, label in enumerate(result.state_labels):
        for region_idx, region in enumerate(REGIONS):
            for age_idx, age_group in enumerate(AGE_GROUPS):
                row = {
                    "date_or_day": label,
                    "region": region,
                    "age_group": age_group,
                    "S0": float(result.state_history["S0"][time_idx, region_idx, age_idx]),
                    "E0": float(result.state_history["E0"][time_idx, region_idx, age_idx]),
                    "I0": float(result.state_history["I0"][time_idx, region_idx, age_idx]),
                    "R": float(result.state_history["R"][time_idx, region_idx, age_idx]),
                    "S1": float(result.state_history["S1"][time_idx, region_idx, age_idx]),
                    "E1": float(result.state_history["E1"][time_idx, region_idx, age_idx]),
                    "I1": float(result.state_history["I1"][time_idx, region_idx, age_idx]),
                    "active_I": float(active_history[time_idx, region_idx, age_idx]),
                    "susceptible_total": float(susceptible_history[time_idx, region_idx, age_idx]),
                    "population": float(result.population[region_idx, age_idx]),
                }
                rows.append(row)
    return pd.DataFrame(rows)


def build_node_daily_metrics(result: SimulationResult) -> pd.DataFrame:
    rows: list[dict] = []
    active_history = result.state_history["I0"][1:] + result.state_history["I1"][1:]
    cumulative_first = result.population[np.newaxis, :, :] - result.state_history["S0"][1:]
    cumulative_reinfection = np.cumsum(result.new_reinfections, axis=0)

    for time_idx, label in enumerate(result.daily_labels):
        for region_idx, region in enumerate(REGIONS):
            for age_idx, age_group in enumerate(AGE_GROUPS):
                population = result.population[region_idx, age_idx]
                new_first = float(result.new_first_infections[time_idx, region_idx, age_idx])
                new_reinf = float(result.new_reinfections[time_idx, region_idx, age_idx])
                cum_first = float(cumulative_first[time_idx, region_idx, age_idx])
                cum_reinf = float(cumulative_reinfection[time_idx, region_idx, age_idx])
                active = float(active_history[time_idx, region_idx, age_idx])
                row = {
                    "date_or_day": label,
                    "region": region,
                    "age_group": age_group,
                    "new_first_infections": new_first,
                    "new_reinfections": new_reinf,
                    "new_infections_total": new_first + new_reinf,
                    "new_recoveries": float(result.new_recoveries[time_idx, region_idx, age_idx]),
                    "active_infected": active,
                    "active_infected_per_100k": safe_rate(active, population, scale=100000.0),
                    "cumulative_first_infections": cum_first,
                    "cumulative_reinfections": cum_reinf,
                    "cumulative_total_infection_episodes": cum_first + cum_reinf,
                    "ever_infected_rate": safe_rate(cum_first, population),
                    "infection_episode_rate": safe_rate(cum_first + cum_reinf, population),
                }
                rows.append(row)

    return pd.DataFrame(rows)


def build_overall_daily_metrics(result: SimulationResult) -> pd.DataFrame:
    total_population = float(result.population.sum())
    elderly_population = float(result.population[:, -1].sum())
    active_history = result.state_history["I0"][1:] + result.state_history["I1"][1:]
    cumulative_first = result.population[np.newaxis, :, :] - result.state_history["S0"][1:]
    cumulative_reinfection = np.cumsum(result.new_reinfections, axis=0)

    total_new_first = result.new_first_infections.sum(axis=(1, 2))
    total_new_reinfection = result.new_reinfections.sum(axis=(1, 2))
    total_active = active_history.sum(axis=(1, 2))
    total_recovered = result.state_history["R"][1:].sum(axis=(1, 2))
    total_flow = result.flow_total.sum(axis=(1, 2, 3, 4))
    cross_region_flow = compute_cross_region_flow(result.flow_total)

    elderly_new_first = result.new_first_infections[:, :, -1].sum(axis=1)
    elderly_new_reinfection = result.new_reinfections[:, :, -1].sum(axis=1)
    elderly_active = active_history[:, :, -1].sum(axis=1)
    elderly_ever_rate = (
        cumulative_first[:, :, -1].sum(axis=1) / elderly_population if elderly_population > 0 else np.zeros_like(elderly_active)
    )
    elderly_reinfection_share = np.divide(
        elderly_new_reinfection,
        elderly_new_first + elderly_new_reinfection,
        out=np.zeros_like(elderly_new_reinfection),
        where=(elderly_new_first + elderly_new_reinfection) > 0.0,
    )

    rows = []
    for time_idx, label in enumerate(result.daily_labels):
        total_new = float(total_new_first[time_idx] + total_new_reinfection[time_idx])
        rows.append(
            {
                "date_or_day": label,
                "total_new_first_infections": float(total_new_first[time_idx]),
                "total_new_reinfections": float(total_new_reinfection[time_idx]),
                "total_new_infections": total_new,
                "total_active_infected": float(total_active[time_idx]),
                "total_active_infected_per_100k": safe_rate(total_active[time_idx], total_population, scale=100000.0),
                "total_recovered": float(total_recovered[time_idx]),
                "total_recovered_per_100k": safe_rate(total_recovered[time_idx], total_population, scale=100000.0),
                "cross_region_flow_share": safe_rate(cross_region_flow[time_idx], total_flow[time_idx]),
                "elderly_new_infections": float(elderly_new_first[time_idx] + elderly_new_reinfection[time_idx]),
                "elderly_active_infected": float(elderly_active[time_idx]),
                "elderly_active_infected_per_100k": safe_rate(elderly_active[time_idx], elderly_population, scale=100000.0),
                "elderly_ever_infected_rate": float(elderly_ever_rate[time_idx]),
                "elderly_reinfection_share": float(elderly_reinfection_share[time_idx]),
            }
        )

    return pd.DataFrame(rows)


def build_flow_long(result: SimulationResult) -> pd.DataFrame:
    rows: list[dict] = []
    for time_idx, label in enumerate(result.daily_labels):
        for source_region_idx, source_region in enumerate(REGIONS):
            for source_age_idx, source_age_group in enumerate(AGE_GROUPS):
                for target_region_idx, target_region in enumerate(REGIONS):
                    for target_age_idx, target_age_group in enumerate(AGE_GROUPS):
                        rows.append(
                            {
                                "date_or_day": label,
                                "source_region": source_region,
                                "source_age_group": source_age_group,
                                "target_region": target_region,
                                "target_age_group": target_age_group,
                                "flow_first": float(
                                    result.flow_first[
                                        time_idx,
                                        source_region_idx,
                                        source_age_idx,
                                        target_region_idx,
                                        target_age_idx,
                                    ]
                                ),
                                "flow_reinfection": float(
                                    result.flow_reinfection[
                                        time_idx,
                                        source_region_idx,
                                        source_age_idx,
                                        target_region_idx,
                                        target_age_idx,
                                    ]
                                ),
                                "flow_total": float(
                                    result.flow_total[
                                        time_idx,
                                        source_region_idx,
                                        source_age_idx,
                                        target_region_idx,
                                        target_age_idx,
                                    ]
                                ),
                            }
                        )
    return pd.DataFrame(rows)


def build_summary_metrics(result: SimulationResult) -> dict:
    active_history = result.state_history["I0"] + result.state_history["I1"]
    cumulative_first_final = result.population - result.state_history["S0"][-1]
    cumulative_reinfection_final = np.cumsum(result.new_reinfections, axis=0)[-1] if result.new_reinfections.size else np.zeros_like(result.population)
    cumulative_episode_final = cumulative_first_final + cumulative_reinfection_final

    region_population = result.population.sum(axis=1)
    peak_day_by_region = {}
    peak_value_by_region = {}
    for region_idx, region in enumerate(REGIONS):
        region_active = active_history[:, region_idx, :].sum(axis=1)
        peak_idx = int(np.argmax(region_active))
        peak_day_by_region[region] = result.state_labels[peak_idx]
        peak_value_by_region[region] = {
            "active_infected": float(region_active[peak_idx]),
            "active_infected_per_100k": safe_rate(region_active[peak_idx], region_population[region_idx], scale=100000.0),
        }

    overall_active = active_history.sum(axis=(1, 2))
    overall_peak_idx = int(np.argmax(overall_active))
    cross_region_total = float(compute_cross_region_flow(result.flow_total).sum())
    total_flow = float(result.flow_total.sum())
    total_reinfection = float(result.new_reinfections.sum())
    total_infections = float(result.new_first_infections.sum() + result.new_reinfections.sum())
    elderly_population = float(result.population[:, -1].sum())
    elderly_active = active_history[:, :, -1].sum(axis=1)
    elderly_peak_idx = int(np.argmax(elderly_active))

    cumulative_ever_rate = {}
    cumulative_episode_rate = {}
    for region_idx, region in enumerate(REGIONS):
        for age_idx, age_group in enumerate(AGE_GROUPS):
            node_id = f"{region}_{AGE_TOKEN_MAP[age_group]}"
            population = result.population[region_idx, age_idx]
            cumulative_ever_rate[node_id] = safe_rate(
                cumulative_first_final[region_idx, age_idx],
                population,
            )
            cumulative_episode_rate[node_id] = safe_rate(
                cumulative_episode_final[region_idx, age_idx],
                population,
            )

    elderly_first = float(cumulative_first_final[:, -1].sum())
    elderly_reinfection = float(cumulative_reinfection_final[:, -1].sum())
    elderly_episodes = elderly_first + elderly_reinfection
    elderly_summary = {
        "elderly_population": elderly_population,
        "elderly_peak_day": result.state_labels[elderly_peak_idx],
        "elderly_peak_active_infected": float(elderly_active[elderly_peak_idx]),
        "elderly_peak_active_infected_per_100k": safe_rate(
            elderly_active[elderly_peak_idx], elderly_population, scale=100000.0
        ),
        "elderly_ever_infected_rate": safe_rate(elderly_first, elderly_population),
        "elderly_episode_rate": safe_rate(elderly_episodes, elderly_population),
        "elderly_reinfection_share": safe_rate(elderly_reinfection, elderly_episodes),
        "elderly_episode_share_of_total": safe_rate(elderly_episodes, total_infections),
    }

    return {
        "peak_day_by_region": peak_day_by_region,
        "peak_value_by_region": peak_value_by_region,
        "overall_peak_day": result.state_labels[overall_peak_idx],
        "overall_peak_value": {
            "active_infected": float(overall_active[overall_peak_idx]),
            "active_infected_per_100k": safe_rate(
                overall_active[overall_peak_idx],
                float(result.population.sum()),
                scale=100000.0,
            ),
        },
        "cumulative_ever_infected_rate_by_region_age": cumulative_ever_rate,
        "cumulative_episode_rate_by_region_age": cumulative_episode_rate,
        "reinfection_share": safe_rate(total_reinfection, total_infections),
        "cross_region_flow_share_total": safe_rate(cross_region_total, total_flow),
        "elderly_burden_summary": elderly_summary,
    }


def build_region_daily_metrics(result: SimulationResult) -> pd.DataFrame:
    active_history = result.state_history["I0"][1:] + result.state_history["I1"][1:]
    cumulative_first = result.population[np.newaxis, :, :] - result.state_history["S0"][1:]
    cumulative_reinfection = np.cumsum(result.new_reinfections, axis=0)

    rows: list[dict] = []
    for time_idx, label in enumerate(result.daily_labels):
        for region_idx, region in enumerate(REGIONS):
            population = float(result.population[region_idx].sum())
            new_first = float(result.new_first_infections[time_idx, region_idx, :].sum())
            new_reinf = float(result.new_reinfections[time_idx, region_idx, :].sum())
            active = float(active_history[time_idx, region_idx, :].sum())
            cum_first = float(cumulative_first[time_idx, region_idx, :].sum())
            cum_reinf = float(cumulative_reinfection[time_idx, region_idx, :].sum())
            rows.append(
                {
                    "date_or_day": label,
                    "region": region,
                    "new_first_infections": new_first,
                    "new_reinfections": new_reinf,
                    "new_infections_total": new_first + new_reinf,
                    "active_infected": active,
                    "active_infected_per_100k": safe_rate(active, population, scale=100000.0),
                    "ever_infected_rate": safe_rate(cum_first, population),
                    "infection_episode_rate": safe_rate(cum_first + cum_reinf, population),
                    "reinfection_share": safe_rate(new_reinf, new_first + new_reinf),
                }
            )
    return pd.DataFrame(rows)


def build_age_group_summary(result: SimulationResult) -> pd.DataFrame:
    cumulative_first_final = result.population - result.state_history["S0"][-1]
    cumulative_reinfection_final = np.cumsum(result.new_reinfections, axis=0)[-1] if result.new_reinfections.size else np.zeros_like(result.population)

    rows = []
    for age_idx, age_group in enumerate(AGE_GROUPS):
        population = float(result.population[:, age_idx].sum())
        cum_first = float(cumulative_first_final[:, age_idx].sum())
        cum_reinf = float(cumulative_reinfection_final[:, age_idx].sum())
        rows.append(
            {
                "age_group": age_group,
                "population": population,
                "ever_infected_rate": safe_rate(cum_first, population),
                "infection_episode_rate": safe_rate(cum_first + cum_reinf, population),
                "reinfection_share": safe_rate(cum_reinf, cum_first + cum_reinf),
                "is_elderly": age_group == "65+",
            }
        )
    return pd.DataFrame(rows)


def summarize_legacy_city(result: SimulationResult, region: str) -> dict:
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


def save_tables(tables: SimulationTables, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    tables.states_long.to_csv(run_dir / "states_long.csv", index=False, encoding="utf-8-sig")
    tables.node_daily_metrics.to_csv(run_dir / "node_daily_metrics.csv", index=False, encoding="utf-8-sig")
    tables.overall_daily_metrics.to_csv(run_dir / "overall_daily_metrics.csv", index=False, encoding="utf-8-sig")
    tables.flow_long.to_csv(run_dir / "flow_long.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary_metrics.json").write_text(
        json.dumps(tables.summary_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def compute_cross_region_flow(flow_total: np.ndarray) -> np.ndarray:
    cross_region_flow = np.zeros(flow_total.shape[0], dtype=float)
    for source_region_idx in range(len(REGIONS)):
        for target_region_idx in range(len(REGIONS)):
            if source_region_idx == target_region_idx:
                continue
            cross_region_flow += flow_total[:, source_region_idx, :, target_region_idx, :].sum(axis=(1, 2))
    return cross_region_flow


def safe_rate(numerator: float | np.ndarray, denominator: float | np.ndarray, scale: float = 1.0):
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(np.asarray(numerator, dtype=float), dtype=float),
        where=np.asarray(denominator, dtype=float) > 0.0,
    ) * scale

