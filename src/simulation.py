from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig
from .constants import REGIMES
from .data_loader import LoadedInputs
from .model import COMPARTMENTS, PreparedRegimeMatrices, create_initial_state, prepare_regime_matrices, step_model


@dataclass
class SimulationResult:
    config: SimulationConfig
    population: np.ndarray
    daily_labels: list[str]
    state_labels: list[str]
    daily_regimes: list[str]
    day_numbers: np.ndarray
    state_history: dict[str, np.ndarray]
    new_first_infections: np.ndarray
    new_reinfections: np.ndarray
    new_recoveries: np.ndarray
    lambda_values: np.ndarray
    flow_first: np.ndarray
    flow_reinfection: np.ndarray
    flow_total: np.ndarray


def run_simulation(config: SimulationConfig, inputs: LoadedInputs) -> SimulationResult:
    daily_labels, state_labels, daily_regimes = build_schedule(config, inputs)
    horizon = len(daily_labels)
    population = inputs.population.copy()

    prepared_matrices: dict[str, PreparedRegimeMatrices] = {
        regime: prepare_regime_matrices(inputs.age_matrices[regime], inputs.region_matrices[regime])
        for regime in REGIMES
    }

    state = create_initial_state(population, config.initial_conditions)
    state_history = {
        compartment: np.zeros((horizon + 1, population.shape[0], population.shape[1]), dtype=float)
        for compartment in COMPARTMENTS
    }
    for compartment in COMPARTMENTS:
        state_history[compartment][0] = getattr(state, compartment)

    new_first_infections = np.zeros((horizon, population.shape[0], population.shape[1]), dtype=float)
    new_reinfections = np.zeros_like(new_first_infections)
    new_recoveries = np.zeros_like(new_first_infections)
    lambda_values = np.zeros_like(new_first_infections)
    flow_shape = (horizon, population.shape[0], population.shape[1], population.shape[0], population.shape[1])
    flow_first = np.zeros(flow_shape, dtype=float)
    flow_reinfection = np.zeros(flow_shape, dtype=float)
    flow_total = np.zeros(flow_shape, dtype=float)

    for day_idx, regime in enumerate(daily_regimes):
        step_result = step_model(
            state=state,
            population=population,
            regime_matrices=prepared_matrices[regime],
            model_config=config.model,
            regime=regime,
        )
        state = step_result.next_state

        new_first_infections[day_idx] = step_result.new_first_infections
        new_reinfections[day_idx] = step_result.new_reinfections
        new_recoveries[day_idx] = step_result.new_recoveries
        lambda_values[day_idx] = step_result.lambda_values
        flow_first[day_idx] = step_result.flow_first
        flow_reinfection[day_idx] = step_result.flow_reinfection
        flow_total[day_idx] = step_result.flow_total

        for compartment in COMPARTMENTS:
            state_history[compartment][day_idx + 1] = getattr(state, compartment)

    day_numbers = np.arange(horizon + 1, dtype=int)
    return SimulationResult(
        config=config,
        population=population,
        daily_labels=daily_labels,
        state_labels=state_labels,
        daily_regimes=daily_regimes,
        day_numbers=day_numbers,
        state_history=state_history,
        new_first_infections=new_first_infections,
        new_reinfections=new_reinfections,
        new_recoveries=new_recoveries,
        lambda_values=lambda_values,
        flow_first=flow_first,
        flow_reinfection=flow_reinfection,
        flow_total=flow_total,
    )


def build_schedule(config: SimulationConfig, inputs: LoadedInputs) -> tuple[list[str], list[str], list[str]]:
    if config.run.mode == "fixed":
        daily_labels = [f"day_{day_idx}" for day_idx in range(1, config.run.days + 1)]
        daily_regimes = [config.run.fixed_regime] * config.run.days
    elif config.run.mode == "calendar":
        if inputs.calendar is None:
            raise ValueError("calendar mode인데 winter_calendar 데이터가 로드되지 않았습니다.")
        daily_labels = inputs.calendar["date"].tolist()
        daily_regimes = inputs.calendar["regime"].tolist()
    else:
        raise ValueError(f"legacy_batch는 run_simulation이 아니라 별도 루틴으로 실행해야 합니다: {config.run.mode}")

    state_labels = ["initial_state", *daily_labels]
    return daily_labels, state_labels, daily_regimes
