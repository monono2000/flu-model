from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd


OBSERVED_AGE_ROWS = (
    "0\uc138",
    "1-6\uc138",
    "7-12\uc138",
    "13-18\uc138",
    "19-49\uc138",
    "50-64\uc138",
    "65\uc138 \uc774\uc0c1",
)

MODEL_AGE_GROUPS = ("0-18", "19-49", "50-64", "65+")


def load_observed_influenza_weekly(
    path: str | Path,
    start_iso_year: int = 2025,
    start_iso_week: int = 36,
    encoding: str = "cp949",
) -> pd.DataFrame:
    path = Path(path)
    rows: list[tuple[str, list[float]]] = []
    with path.open("r", encoding=encoding) as handle:
        for row in csv.reader(handle):
            values: list[float] = []
            for cell in row[1:]:
                cell = cell.strip()
                if not cell:
                    continue
                try:
                    values.append(float(cell))
                except ValueError:
                    continue
            rows.append((row[0].strip(), values))

    if len(rows) != len(OBSERVED_AGE_ROWS):
        raise ValueError(
            f"\uad00\uce21 CSV \uc5f0\ub839 \ud589 \uc218\uac00 \uc608\uc0c1\uacfc \ub2e4\ub985\ub2c8\ub2e4. "
            f"expected={len(OBSERVED_AGE_ROWS)}, actual={len(rows)}"
        )

    for expected_label, (actual_label, values) in zip(OBSERVED_AGE_ROWS, rows):
        if expected_label != actual_label:
            raise ValueError(
                "\uad00\uce21 CSV \uc5f0\ub839 \ud589 \uc21c\uc11c\uac00 \uc608\uc0c1\uacfc \ub2e4\ub985\ub2c8\ub2e4. "
                f"expected={expected_label}, actual={actual_label}"
            )
        if len(values) != 33:
            raise ValueError(
                "\uad00\uce21 CSV\ub294 33\uac1c \uc8fc\uac04 \uac12\uc774 \ud544\uc694\ud569\ub2c8\ub2e4. "
                f"row={actual_label}, actual={len(values)}"
            )

    weekly_index = build_iso_week_index(
        start_iso_year=start_iso_year,
        start_iso_week=start_iso_week,
        count=33,
    )
    age_0_18 = np.average(
        np.array([rows[idx][1] for idx in range(4)], dtype=float),
        axis=0,
        weights=[1.0, 6.0, 6.0, 6.0],
    )
    return pd.DataFrame(
        {
            "week_start": weekly_index,
            "0-18": age_0_18,
            "19-49": rows[4][1],
            "50-64": rows[5][1],
            "65+": rows[6][1],
        }
    )


def build_iso_week_index(start_iso_year: int, start_iso_week: int, count: int) -> list[str]:
    weeks: list[str] = []
    current_year = start_iso_year
    current_week = start_iso_week
    for _ in range(count):
        weeks.append(dt.date.fromisocalendar(current_year, current_week, 1).isoformat())
        current_week += 1
        try:
            dt.date.fromisocalendar(current_year, current_week, 1)
        except ValueError:
            current_year += 1
            current_week = 1
    return weeks


def filter_weekly_window(
    weekly_df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    mask = (weekly_df["week_start"] >= start_date) & (weekly_df["week_start"] <= end_date)
    return weekly_df.loc[mask].reset_index(drop=True)


def melt_weekly_rates(weekly_df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    return weekly_df.melt(
        id_vars="week_start",
        value_vars=list(MODEL_AGE_GROUPS),
        var_name="age_group",
        value_name=value_name,
    )
