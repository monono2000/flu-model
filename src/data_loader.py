from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .constants import AGE_GROUPS, REGIMES, REGIONS


@dataclass
class LoadedInputs:
    regions: tuple[str, ...]
    region_groups: tuple[str, ...]
    population: np.ndarray
    age_matrices: dict[str, np.ndarray]
    region_matrices: dict[str, np.ndarray]
    calendar: pd.DataFrame | None = None


def load_inputs(config: SimulationConfig, require_calendar: bool = False) -> LoadedInputs:
    regions, region_groups, population = load_population(config.paths.population_path())
    age_matrices = {
        regime: load_square_matrix(config.paths.age_contact_path(regime), AGE_GROUPS)
        for regime in REGIMES
    }
    region_matrices = {
        regime: load_region_matrix(
            config.paths.region_contact_path(regime),
            regions=regions,
            region_groups=region_groups,
            population=population,
        )
        for regime in REGIMES
    }

    if config.counterfactual.no_cross_region:
        region_matrices = {
            regime: apply_no_cross_region(matrix)
            for regime, matrix in region_matrices.items()
        }

    calendar = None
    if require_calendar:
        calendar = load_winter_calendar(config.paths.winter_calendar_path(), config.paths.sample_calendar_path())
        if config.counterfactual.no_holiday:
            calendar = calendar.copy()
            calendar.loc[calendar["regime"] == "IV", "regime"] = "II"

    return LoadedInputs(
        regions=regions,
        region_groups=region_groups,
        population=population,
        age_matrices=age_matrices,
        region_matrices=region_matrices,
        calendar=calendar,
    )


def load_population(path: Path) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"필수 인구 파일이 없습니다: {path}")

    population_df = pd.read_csv(path)
    required_columns = {"region", "age_group", "population"}
    missing_columns = required_columns.difference(population_df.columns)
    if missing_columns:
        raise ValueError(f"population_by_region_age.csv에 필요한 열이 없습니다: {sorted(missing_columns)}")

    region_order = tuple(dict.fromkeys(population_df["region"].astype(str).tolist()))
    region_groups = tuple(_region_group_for(population_df, region) for region in region_order)

    population = np.zeros((len(region_order), len(AGE_GROUPS)), dtype=float)
    for region_idx, region in enumerate(region_order):
        for age_idx, age_group in enumerate(AGE_GROUPS):
            matched = population_df[
                (population_df["region"] == region) & (population_df["age_group"] == age_group)
            ]
            if matched.empty:
                raise ValueError(
                    f"population_by_region_age.csv에서 ({region}, {age_group}) 행을 찾을 수 없습니다."
                )
            population[region_idx, age_idx] = float(matched.iloc[0]["population"])

    return region_order, region_groups, population


def load_square_matrix(path: Path, labels: tuple[str, ...]) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"필수 행렬 파일이 없습니다: {path}")

    raw_df = pd.read_csv(path)
    if labels[0] in raw_df.columns and len(raw_df.columns) == len(labels):
        raw_df.index = list(labels)
        matrix_df = raw_df.loc[:, list(labels)]
    else:
        matrix_df = pd.read_csv(path, index_col=0)
        if matrix_df.shape[0] != len(labels) or matrix_df.shape[1] != len(labels):
            raise ValueError(f"행렬 크기가 예상과 다릅니다: {path}")
        if not set(labels).issubset(matrix_df.columns):
            matrix_df.columns = list(labels)
        if not set(labels).issubset(matrix_df.index):
            matrix_df.index = list(labels)
        matrix_df = matrix_df.loc[list(labels), list(labels)]

    matrix = matrix_df.astype(float).to_numpy()
    if np.isnan(matrix).any() or np.isinf(matrix).any():
        raise ValueError(f"행렬 파일에 NaN/Inf가 포함되어 있습니다: {path}")
    return matrix


def load_region_matrix(
    path: Path,
    regions: tuple[str, ...],
    region_groups: tuple[str, ...],
    population: np.ndarray,
) -> np.ndarray:
    raw_df = pd.read_csv(path, index_col=0)
    raw_df.index = raw_df.index.astype(str)
    raw_df.columns = raw_df.columns.astype(str)

    if set(regions).issubset(raw_df.index) and set(regions).issubset(raw_df.columns):
        matrix_df = raw_df.loc[list(regions), list(regions)]
        return matrix_df.astype(float).to_numpy()

    coarse_groups = tuple(dict.fromkeys(region_groups))
    if set(coarse_groups).issubset(raw_df.index) and set(coarse_groups).issubset(raw_df.columns):
        coarse_df = raw_df.loc[list(coarse_groups), list(coarse_groups)]
        return expand_region_matrix(
            coarse_matrix=coarse_df.astype(float).to_numpy(),
            coarse_groups=coarse_groups,
            region_groups=region_groups,
            population=population,
        )

    if len(regions) == len(REGIONS):
        return load_square_matrix(path, REGIONS)
    raise ValueError(f"지역 행렬 레이블을 해석할 수 없습니다: {path}")


def expand_region_matrix(
    coarse_matrix: np.ndarray,
    coarse_groups: tuple[str, ...],
    region_groups: tuple[str, ...],
    population: np.ndarray,
) -> np.ndarray:
    region_totals = population.sum(axis=1)
    group_index = {group: idx for idx, group in enumerate(coarse_groups)}
    group_totals = {
        group: float(
            region_totals[[idx for idx, value in enumerate(region_groups) if value == group]].sum()
        )
        for group in coarse_groups
    }

    expanded = np.zeros((len(region_groups), len(region_groups)), dtype=float)
    for source_idx, source_group in enumerate(region_groups):
        source_weight = (
            float(region_totals[source_idx] / group_totals[source_group])
            if group_totals[source_group] > 0.0
            else 0.0
        )
        for target_idx, target_group in enumerate(region_groups):
            expanded[source_idx, target_idx] = (
                coarse_matrix[group_index[source_group], group_index[target_group]] * source_weight
            )
    return expanded


def load_winter_calendar(path: Path, sample_path: Path) -> pd.DataFrame:
    if not path.exists():
        if not sample_path.exists():
            create_sample_winter_calendar(sample_path)
        raise FileNotFoundError(
            "winter_calendar.csv가 없습니다. "
            f"샘플 템플릿을 확인하세요: {sample_path}"
        )

    calendar_df = pd.read_csv(path)
    required_columns = {"date", "regime"}
    missing_columns = required_columns.difference(calendar_df.columns)
    if missing_columns:
        raise ValueError(f"winter_calendar.csv에 필요한 열이 없습니다: {sorted(missing_columns)}")

    calendar_df = calendar_df.copy()
    calendar_df["date"] = pd.to_datetime(calendar_df["date"], errors="raise")
    if calendar_df["date"].duplicated().any():
        raise ValueError("winter_calendar.csv에 중복 날짜가 있습니다.")
    invalid_regimes = sorted(set(calendar_df["regime"]) - set(REGIMES))
    if invalid_regimes:
        raise ValueError(
            f"winter_calendar.csv의 regime 값은 I/II/III/IV만 허용됩니다. 오류 값: {invalid_regimes}"
        )

    calendar_df = calendar_df.sort_values("date").reset_index(drop=True)
    calendar_df["date"] = calendar_df["date"].dt.strftime("%Y-%m-%d")
    return calendar_df


def create_sample_winter_calendar(sample_path: Path) -> None:
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2025-12-01", "2026-02-28", freq="D")
    sample_df = pd.DataFrame({"date": dates, "regime": "I"})

    weekend_mask = sample_df["date"].dt.dayofweek >= 5
    vacation_mask = (sample_df["date"] >= pd.Timestamp("2025-12-29")) & (
        sample_df["date"] <= pd.Timestamp("2026-02-27")
    )
    lunar_mask = (sample_df["date"] >= pd.Timestamp("2026-02-16")) & (
        sample_df["date"] <= pd.Timestamp("2026-02-18")
    )
    extra_holiday_mask = sample_df["date"].isin(
        [pd.Timestamp("2025-12-25"), pd.Timestamp("2026-01-01")]
    )

    sample_df.loc[weekend_mask, "regime"] = "III"
    sample_df.loc[extra_holiday_mask, "regime"] = "III"
    sample_df.loc[vacation_mask & (~weekend_mask), "regime"] = "II"
    sample_df.loc[lunar_mask, "regime"] = "IV"
    sample_df["date"] = sample_df["date"].dt.strftime("%Y-%m-%d")
    sample_df.to_csv(sample_path, index=False, encoding="utf-8-sig")


def apply_no_cross_region(region_matrix: np.ndarray) -> np.ndarray:
    adjusted = region_matrix.copy()
    for row_idx in range(adjusted.shape[0]):
        for col_idx in range(adjusted.shape[1]):
            if row_idx != col_idx:
                adjusted[row_idx, col_idx] = 0.0
    return adjusted


def _region_group_for(population_df: pd.DataFrame, region: str) -> str:
    if "region_group" not in population_df.columns:
        return region
    matched = (
        population_df.loc[population_df["region"] == region, "region_group"]
        .dropna()
        .astype(str)
        .unique()
    )
    if matched.size == 0:
        return region
    if matched.size > 1:
        raise ValueError(f"region_group 값이 지역별로 일관되지 않습니다: {region}")
    return str(matched[0])
