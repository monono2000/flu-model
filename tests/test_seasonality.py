from __future__ import annotations

import numpy as np

from src.data_loader import load_inputs
from src.observations import load_observed_influenza_weekly
from src.simulation import run_simulation
from tests.helpers import REPO_ROOT, build_test_config, write_calendar


def test_load_observed_influenza_weekly_accepts_header_row(tmp_path):
    source_text = (REPO_ROOT / "data" / "synthetic_target_influenza_2025_2026.csv").read_text(encoding="utf-8")
    header = "," + ",".join([f"{week}주" for week in list(range(36, 53)) + list(range(1, 17))]) + "\n"
    csv_path = tmp_path / "weekly_with_header.csv"
    csv_path.write_text(header + source_text, encoding="utf-8")

    weekly_df = load_observed_influenza_weekly(csv_path, encoding="utf-8")

    assert len(weekly_df) == 33
    assert tuple(weekly_df.columns) == ("week_start", "0-18", "19-49", "50-64", "65+")


def test_load_observed_influenza_weekly_accepts_partial_week_header(tmp_path):
    csv_lines = [
        ",49주,50주,51주,52주,1주,2주,3주",
        "0세,47.6,26.2,26.6,26.2,20.8,26.9,29.7",
        "1-6세,81.3,73.2,56.9,51.3,44.9,51,73.4",
        "7-12세,150,116,110.4,107.3,105.4,127.2,135.9",
        "13-18세,119.1,116.2,91.5,75,77.1,97.2,88.7",
        "19-49세,59.3,50.5,39,37.7,38.2,44.2,44.8",
        "50-64세,17.3,17.7,13.9,13.1,13.6,14.7,14.7",
        "65세 이상,10.5,10.6,9,10,9.3,9,9.5",
    ]
    csv_path = tmp_path / "partial_week_header.csv"
    csv_path.write_text("\n".join(csv_lines), encoding="utf-8")

    weekly_df = load_observed_influenza_weekly(csv_path, encoding="utf-8", start_iso_year=2025)

    assert weekly_df["week_start"].tolist() == [
        "2025-12-01",
        "2025-12-08",
        "2025-12-15",
        "2025-12-22",
        "2025-12-29",
        "2026-01-05",
        "2026-01-12",
    ]
    assert float(weekly_df.loc[0, "19-49"]) == 59.3


def test_run_simulation_applies_time_beta_multiplier_from_csv(tmp_path):
    config = build_test_config(tmp_path, mode="calendar", days=8)
    write_calendar(config.paths.data_dir, ["I"] * 8, start_date="2025-12-01")
    config.model.time_beta_from_csv.enabled = True
    config.model.time_beta_from_csv.source_csv = "data/synthetic_target_influenza_2025_2026.csv"
    config.model.time_beta_from_csv.encoding = "utf-8"
    config.model.time_beta_from_csv.normalize_to = "mean"
    config.model.time_beta_from_csv.age_weighting = "population"
    config.model.time_beta_from_csv.power = 1.0
    config.validate()

    inputs = load_inputs(config, require_calendar=True)
    result = run_simulation(config, inputs)

    assert len(result.daily_time_beta_multipliers) == 8
    assert np.allclose(result.daily_time_beta_multipliers[:7], result.daily_time_beta_multipliers[0])
    assert not np.isclose(result.daily_time_beta_multipliers[7], result.daily_time_beta_multipliers[0])
