from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .constants import AGE_GROUPS
from .observations import load_observed_influenza_weekly


def derive_overall_weekly_curve_from_csv(
    weekly_csv_path: str | Path,
    age_population_weights: Mapping[str, float] | Sequence[float],
    encoding: str = "cp949",
    age_weighting: str = "population",
) -> pd.DataFrame:
    weekly_df = load_observed_influenza_weekly(weekly_csv_path, encoding=encoding)
    weights = resolve_age_weights(age_population_weights, age_weighting=age_weighting)
    overall_rate = np.average(
        weekly_df.loc[:, list(AGE_GROUPS)].to_numpy(dtype=float),
        axis=1,
        weights=weights,
    )
    return pd.DataFrame(
        {
            "week_start": weekly_df["week_start"].tolist(),
            "overall_rate": overall_rate,
        }
    )


def derive_daily_beta_multipliers_from_csv(
    weekly_csv_path: str | Path,
    daily_labels: Sequence[str],
    age_population_weights: Mapping[str, float] | Sequence[float],
    encoding: str = "cp949",
    normalize_to: str = "mean",
    age_weighting: str = "population",
    power: float = 1.0,
) -> np.ndarray:
    weekly_curve = derive_overall_weekly_curve_from_csv(
        weekly_csv_path=weekly_csv_path,
        age_population_weights=age_population_weights,
        encoding=encoding,
        age_weighting=age_weighting,
    )
    raw_values = weekly_curve["overall_rate"].to_numpy(dtype=float)
    if normalize_to == "mean":
        baseline = float(np.mean(raw_values))
    elif normalize_to == "max":
        baseline = float(np.max(raw_values))
    else:
        raise ValueError(f"unsupported normalize_to for time beta csv: {normalize_to}")
    if baseline <= 0.0:
        raise ValueError("time beta csv baseline must be positive")

    weekly_curve = weekly_curve.copy()
    weekly_curve["beta_multiplier"] = np.power(raw_values / baseline, power)
    lookup = {
        row["week_start"]: float(row["beta_multiplier"])
        for row in weekly_curve.to_dict(orient="records")
    }
    available_week_starts = weekly_curve["week_start"].tolist()

    multipliers = []
    for label in daily_labels:
        date_value = dt.date.fromisoformat(label)
        week_start = (date_value - dt.timedelta(days=date_value.weekday())).isoformat()
        multipliers.append(_lookup_week_value(lookup, available_week_starts, week_start))
    return np.array(multipliers, dtype=float)


def aggregate_model_daily_to_weekly(daily_labels: Sequence[str], values: Sequence[float]) -> pd.DataFrame:
    weekly_rows = []
    for label, value in zip(daily_labels, values):
        date_value = dt.date.fromisoformat(label)
        week_start = (date_value - dt.timedelta(days=date_value.weekday())).isoformat()
        weekly_rows.append({"week_start": week_start, "value": float(value)})

    weekly_df = pd.DataFrame(weekly_rows)
    return weekly_df.groupby("week_start", as_index=False)["value"].sum()


def resolve_age_weights(
    age_population_weights: Mapping[str, float] | Sequence[float],
    age_weighting: str,
) -> np.ndarray:
    if age_weighting == "equal":
        return np.ones(len(AGE_GROUPS), dtype=float)
    if isinstance(age_population_weights, Mapping):
        weights = np.array([float(age_population_weights[age_group]) for age_group in AGE_GROUPS], dtype=float)
    else:
        weights = np.array(list(age_population_weights), dtype=float)
    if weights.shape != (len(AGE_GROUPS),):
        raise ValueError("age population weights must have length 4")
    if np.all(weights <= 0.0):
        raise ValueError("age population weights must contain a positive value")
    return weights


def _lookup_week_value(
    lookup: dict[str, float],
    available_week_starts: Sequence[str],
    week_start: str,
) -> float:
    if week_start in lookup:
        return lookup[week_start]
    if week_start < available_week_starts[0]:
        return lookup[available_week_starts[0]]
    return lookup[available_week_starts[-1]]
