from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ModelConfig
from .constants import AGE_GROUPS
from .flow_decomposition import decompose_daily_flows


COMPARTMENTS = ("S0", "E0", "I0", "R", "S1", "E1", "I1")


@dataclass
class StateVector:
    S0: np.ndarray
    E0: np.ndarray
    I0: np.ndarray
    R: np.ndarray
    S1: np.ndarray
    E1: np.ndarray
    I1: np.ndarray

    def copy(self) -> "StateVector":
        return StateVector(**{name: getattr(self, name).copy() for name in COMPARTMENTS})

    def total_population(self) -> np.ndarray:
        return sum((getattr(self, name) for name in COMPARTMENTS), start=np.zeros_like(self.S0))


@dataclass
class PreparedRegimeMatrices:
    age_matrix: np.ndarray
    region_matrix: np.ndarray
    k_age: np.ndarray
    p_age: np.ndarray
    p_reg: np.ndarray


@dataclass
class StepResult:
    next_state: StateVector
    lambda_values: np.ndarray
    raw_contrib: np.ndarray
    flow_first: np.ndarray
    flow_reinfection: np.ndarray
    flow_total: np.ndarray
    new_first_infections: np.ndarray
    new_reinfections: np.ndarray
    new_recoveries: np.ndarray
    waning: np.ndarray


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = sanitize_array(matrix)
    row_sum = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum != 0.0)


def normalize_cols(matrix: np.ndarray) -> np.ndarray:
    matrix = sanitize_array(matrix)
    col_sum = matrix.sum(axis=0, keepdims=True)
    return np.divide(matrix, col_sum, out=np.zeros_like(matrix), where=col_sum != 0.0)


def sanitize_array(array: np.ndarray) -> np.ndarray:
    return np.nan_to_num(array.astype(float), nan=0.0, posinf=0.0, neginf=0.0)


def prepare_regime_matrices(age_matrix: np.ndarray, region_matrix: np.ndarray) -> PreparedRegimeMatrices:
    age_matrix = sanitize_array(age_matrix)
    region_matrix = sanitize_array(region_matrix)
    return PreparedRegimeMatrices(
        age_matrix=age_matrix,
        region_matrix=region_matrix,
        k_age=age_matrix.sum(axis=1),
        p_age=normalize_rows(age_matrix),
        p_reg=normalize_cols(region_matrix),
    )


def create_initial_state(
    population: np.ndarray,
    mode_config,
    regions: tuple[str, ...],
    region_groups: tuple[str, ...] | None = None,
) -> StateVector:
    population = sanitize_array(population)
    initial_seed = np.zeros_like(population)

    if mode_config.mode == "same_prevalence":
        age_prevalence = np.array(
            [
                mode_config.same_prevalence_by_age.get(age_group, mode_config.same_prevalence)
                for age_group in AGE_GROUPS
            ],
            dtype=float,
        )
        initial_seed = population * age_prevalence[np.newaxis, :]
    elif mode_config.mode == "seed_by_region_age":
        for region_idx, region in enumerate(regions):
            age_payload = mode_config.seed_by_region_age.get(region, {})
            for age_idx, age_group in enumerate(AGE_GROUPS):
                initial_seed[region_idx, age_idx] = float(age_payload.get(age_group, 0.0))
    elif mode_config.mode == "equal_seed":
        initial_seed = np.full_like(population, float(mode_config.equal_seed_count))
    elif mode_config.mode == "equal_absolute_seed":
        initial_seed = build_group_equal_seed(
            population=population,
            equal_seed_count=float(mode_config.equal_seed_count),
            group_labels=region_groups or regions,
        )
    else:
        raise ValueError(f"지원하지 않는 초기조건 모드입니다: {mode_config.mode}")

    initial_seed = np.clip(initial_seed, 0.0, population * 0.99)
    S0 = population - initial_seed
    E0 = np.zeros_like(population)
    I0 = np.zeros_like(population)
    if mode_config.seed_compartment == "I":
        I0 = initial_seed
    else:
        E0 = initial_seed

    zeros = np.zeros_like(population)
    return StateVector(S0=S0, E0=E0, I0=I0, R=zeros.copy(), S1=zeros.copy(), E1=zeros.copy(), I1=zeros.copy())


def build_group_equal_seed(
    population: np.ndarray,
    equal_seed_count: float,
    group_labels: tuple[str, ...],
) -> np.ndarray:
    if len(group_labels) != population.shape[0]:
        raise ValueError("group_labels length must match the population region axis.")

    initial_seed = np.zeros_like(population)
    unique_groups = tuple(dict.fromkeys(group_labels))
    for age_idx in range(population.shape[1]):
        for group in unique_groups:
            indices = [idx for idx, label in enumerate(group_labels) if label == group]
            group_population = float(population[indices, age_idx].sum())
            if group_population <= 0.0:
                continue
            initial_seed[indices, age_idx] = (
                equal_seed_count * population[indices, age_idx] / group_population
            )
    return initial_seed


def step_model(
    state: StateVector,
    population: np.ndarray,
    regime_matrices: PreparedRegimeMatrices,
    model_config: ModelConfig,
    regime: str,
    time_beta_multiplier: float = 1.0,
) -> StepResult:
    susceptibility = np.array(
        [model_config.susceptibility[age_group] for age_group in AGE_GROUPS],
        dtype=float,
    )
    beta_multiplier = float(model_config.beta_multiplier[regime]) * float(time_beta_multiplier)
    lambda_values, raw_contrib = compute_force_of_infection(
        state=state,
        population=population,
        regime_matrices=regime_matrices,
        beta0=float(model_config.beta0),
        beta_multiplier=beta_multiplier,
        susceptibility=susceptibility,
    )

    new_first_infections, new_reinfections = compute_new_exposures(
        S0=state.S0,
        S1=state.S1,
        lambda_values=lambda_values,
        infection_update=model_config.infection_update,
        reinfection_scale=float(model_config.reinfection_susceptibility_scale),
        enable_reinfection=model_config.enable_reinfection,
    )

    sigma = 1.0 / float(model_config.latent_period_days)
    gamma = 1.0 / float(model_config.infectious_period_days)
    omega = float(model_config.waning_rate_per_day) if model_config.enable_reinfection else 0.0

    new_I0 = np.clip(sigma * state.E0, 0.0, state.E0)
    new_I1 = np.clip(sigma * state.E1, 0.0, state.E1)
    new_R0 = np.clip(gamma * state.I0, 0.0, state.I0)
    new_R1 = np.clip(gamma * state.I1, 0.0, state.I1)
    waning = np.clip(omega * state.R, 0.0, state.R)

    next_state = StateVector(
        S0=state.S0 - new_first_infections,
        E0=state.E0 + new_first_infections - new_I0,
        I0=state.I0 + new_I0 - new_R0,
        R=state.R + new_R0 + new_R1 - waning,
        S1=state.S1 + waning - new_reinfections,
        E1=state.E1 + new_reinfections - new_I1,
        I1=state.I1 + new_I1 - new_R1,
    )
    next_state = enforce_population_conservation(next_state, population)

    flow_first, flow_reinfection, flow_total = decompose_daily_flows(
        raw_contrib=raw_contrib,
        new_first_infections=new_first_infections,
        new_reinfections=new_reinfections,
    )
    new_recoveries = new_R0 + new_R1
    return StepResult(
        next_state=next_state,
        lambda_values=lambda_values,
        raw_contrib=raw_contrib,
        flow_first=flow_first,
        flow_reinfection=flow_reinfection,
        flow_total=flow_total,
        new_first_infections=new_first_infections,
        new_reinfections=new_reinfections,
        new_recoveries=new_recoveries,
        waning=waning,
    )


def compute_force_of_infection(
    state: StateVector,
    population: np.ndarray,
    regime_matrices: PreparedRegimeMatrices,
    beta0: float,
    beta_multiplier: float,
    susceptibility: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    infectious_total = state.I0 + state.I1
    infectious_fraction = np.divide(
        infectious_total,
        population,
        out=np.zeros_like(infectious_total),
        where=population > 0.0,
    )

    source_region_count, source_age_count = infectious_fraction.shape
    raw_contrib = np.zeros(
        (source_region_count, source_age_count, source_region_count, source_age_count),
        dtype=float,
    )
    lambda_values = np.zeros_like(infectious_fraction)

    for target_region in range(source_region_count):
        for target_age in range(source_age_count):
            for source_region in range(source_region_count):
                region_weight = regime_matrices.p_reg[source_region, target_region]
                if region_weight <= 0.0:
                    continue
                for source_age in range(source_age_count):
                    raw_contrib[source_region, source_age, target_region, target_age] = (
                        regime_matrices.k_age[target_age]
                        * regime_matrices.p_age[target_age, source_age]
                        * region_weight
                        * infectious_fraction[source_region, source_age]
                    )
            lambda_values[target_region, target_age] = (
                beta0
                * beta_multiplier
                * susceptibility[target_age]
                * raw_contrib[:, :, target_region, target_age].sum()
            )

    return sanitize_array(lambda_values), sanitize_array(raw_contrib)


def compute_new_exposures(
    S0: np.ndarray,
    S1: np.ndarray,
    lambda_values: np.ndarray,
    infection_update: str,
    reinfection_scale: float,
    enable_reinfection: bool,
) -> tuple[np.ndarray, np.ndarray]:
    lambda_values = np.clip(sanitize_array(lambda_values), 0.0, None)
    if infection_update == "probability":
        new_first_infections = S0 * (1.0 - np.exp(-lambda_values))
        if enable_reinfection:
            reinfection_lambda = np.clip(reinfection_scale * lambda_values, 0.0, None)
            new_reinfections = S1 * (1.0 - np.exp(-reinfection_lambda))
        else:
            new_reinfections = np.zeros_like(S1)
    else:
        new_first_infections = S0 * lambda_values
        new_reinfections = S1 * reinfection_scale * lambda_values if enable_reinfection else np.zeros_like(S1)

    new_first_infections = np.clip(new_first_infections, 0.0, S0)
    new_reinfections = np.clip(new_reinfections, 0.0, S1)
    return sanitize_array(new_first_infections), sanitize_array(new_reinfections)


def enforce_population_conservation(state: StateVector, population: np.ndarray) -> StateVector:
    payload = {name: np.clip(sanitize_array(getattr(state, name)), 0.0, None) for name in COMPARTMENTS}
    total = sum((payload[name] for name in COMPARTMENTS), start=np.zeros_like(population))
    residual = population - total
    payload["S0"] = payload["S0"] + residual
    payload["S0"] = np.clip(payload["S0"], 0.0, None)

    total = sum((payload[name] for name in COMPARTMENTS), start=np.zeros_like(population))
    if not np.allclose(total, population, atol=1.0e-8):
        raise ValueError("상태 갱신 후 총인구 보존이 깨졌습니다.")
    return StateVector(**payload)
