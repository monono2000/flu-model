from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import AGE_GROUPS
from .observations import filter_weekly_window, load_observed_influenza_weekly


def derive_age_susceptibility_from_csv(
    weekly_csv_path: str | Path,
    encoding: str = "utf-8",
    start_date: str = "2025-12-01",
    end_date: str = "2026-02-23",
    metric: str = "mean",
    preseason_start_date: str = "2025-09-01",
    preseason_end_date: str = "2025-11-24",
    normalize_to: str = "mean",
) -> dict[str, float]:
    weekly_df = load_observed_influenza_weekly(weekly_csv_path, encoding=encoding)
    window_df = filter_weekly_window(weekly_df, start_date=start_date, end_date=end_date)
    if window_df.empty:
        raise ValueError("susceptibility 계산 구간에 해당하는 주간 데이터가 없습니다.")

    if metric == "mean":
        raw_values = _series_from_window(window_df, reducer="mean")
    elif metric == "peak":
        raw_values = _series_from_window(window_df, reducer="max")
    elif metric == "preseason_ratio":
        preseason_df = filter_weekly_window(
            weekly_df,
            start_date=preseason_start_date,
            end_date=preseason_end_date,
        )
        if preseason_df.empty:
            raise ValueError("preseason_ratio 계산용 기준 구간 데이터가 없습니다.")
        preseason_mean = _series_from_window(preseason_df, reducer="mean")
        winter_mean = _series_from_window(window_df, reducer="mean")
        raw_values = winter_mean.divide(preseason_mean.where(preseason_mean > 0.0))
    else:
        raise ValueError(f"지원하지 않는 susceptibility metric입니다: {metric}")

    raw_values = raw_values.astype(float)
    if normalize_to == "mean":
        baseline = float(raw_values.mean())
    else:
        baseline = float(raw_values[normalize_to])
    if baseline <= 0.0:
        raise ValueError("susceptibility 정규화 기준값이 0보다 커야 합니다.")

    normalized = raw_values / baseline
    return {age_group: float(normalized[age_group]) for age_group in AGE_GROUPS}


def _series_from_window(window_df: pd.DataFrame, reducer: str) -> pd.Series:
    age_df = window_df.loc[:, list(AGE_GROUPS)]
    if reducer == "mean":
        return age_df.mean(axis=0)
    if reducer == "max":
        return age_df.max(axis=0)
    raise ValueError(f"지원하지 않는 reducer입니다: {reducer}")
