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
    raw_rows: list[list[str]] = []
    with path.open("r", encoding=encoding) as handle:
        for row in csv.reader(handle):
            raw_rows.append([cell.strip() for cell in row])

    week_labels: list[str] | None = None
    if _looks_like_header_row(raw_rows):
        week_labels = [cell for cell in raw_rows[0][1:] if cell]
        raw_rows = raw_rows[1:]

    if len(raw_rows) != len(OBSERVED_AGE_ROWS):
        raise ValueError(
            f"\uad00\uce21 CSV \uc5f0\ub839 \ud589 \uc218\uac00 \uc608\uc0c1\uacfc \ub2e4\ub985\ub2c8\ub2e4. "
            f"expected={len(OBSERVED_AGE_ROWS)}, actual={len(raw_rows)}"
        )

    rows: list[tuple[str, list[float]]] = []
    weekly_value_count: int | None = None
    for expected_label, raw_row in zip(OBSERVED_AGE_ROWS, raw_rows):
        actual_label = raw_row[0].strip()
        if expected_label != actual_label:
            raise ValueError(
                "\uad00\uce21 CSV \uc5f0\ub839 \ud589 \uc21c\uc11c\uac00 \uc608\uc0c1\uacfc \ub2e4\ub985\ub2c8\ub2e4. "
                f"expected={expected_label}, actual={actual_label}"
            )
        values = _parse_numeric_cells(raw_row[1:])
        if weekly_value_count is None:
            weekly_value_count = len(values)
        if len(values) != weekly_value_count:
            raise ValueError(
                "\uad00\uce21 CSV \uc8fc\uac04 \uac12 \uac1c\uc218\uac00 \ud589\ub9c8\ub2e4 \ub3d9\uc77c\ud574\uc57c \ud569\ub2c8\ub2e4. "
                f"row={actual_label}, actual={len(values)}, expected={weekly_value_count}"
            )
        rows.append((actual_label, values))

    if weekly_value_count is None or weekly_value_count == 0:
        raise ValueError("\uad00\uce21 CSV\uc5d0 \uc8fc\uac04 \uac12\uc774 \uc5c6\uc2b5\ub2c8\ub2e4.")

    if week_labels is not None:
        if len(week_labels) != weekly_value_count:
            raise ValueError(
                "\ud5e4\ub354 \uc8fc\ucc28 \uac1c\uc218\uc640 \uc8fc\uac04 \uac12 \uac1c\uc218\uac00 \ub2e4\ub985\ub2c8\ub2e4. "
                f"header={len(week_labels)}, values={weekly_value_count}"
            )
        weekly_index = build_iso_week_index_from_labels(week_labels, start_iso_year=start_iso_year)
    else:
        if weekly_value_count != 33:
            raise ValueError(
                "\ud5e4\ub354\uac00 \uc5c6\ub294 \uad00\uce21 CSV\ub294 33\uac1c \uc8fc\uac04 \uac12\uc774 \ud544\uc694\ud569\ub2c8\ub2e4. "
                f"actual={weekly_value_count}"
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


def _looks_like_header_row(rows: list[list[str]]) -> bool:
    if len(rows) != len(OBSERVED_AGE_ROWS) + 1:
        return False
    if not rows or not rows[0]:
        return False
    if rows[0][0] in OBSERVED_AGE_ROWS:
        return False
    return tuple(row[0].strip() for row in rows[1:]) == OBSERVED_AGE_ROWS


def _parse_numeric_cells(cells: list[str]) -> list[float]:
    values: list[float] = []
    for cell in cells:
        if not cell:
            continue
        try:
            values.append(float(cell))
        except ValueError:
            continue
    return values


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


def build_iso_week_index_from_labels(week_labels: list[str], start_iso_year: int) -> list[str]:
    week_numbers = [_parse_week_label(label) for label in week_labels]
    weeks: list[str] = []
    current_year = start_iso_year
    previous_week: int | None = None
    for week_number in week_numbers:
        if previous_week is not None and week_number < previous_week:
            current_year += 1
        weeks.append(dt.date.fromisocalendar(current_year, week_number, 1).isoformat())
        previous_week = week_number
    return weeks


def _parse_week_label(label: str) -> int:
    digits = "".join(char for char in label if char.isdigit())
    if not digits:
        raise ValueError(f"week label does not contain a number: {label}")
    return int(digits)


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
