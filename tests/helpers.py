from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from src.config import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_test_config(tmp_path: Path, mode: str = "fixed", days: int = 20, regime: str = "I"):
    data_dir = tmp_path / "data"
    shutil.copytree(REPO_ROOT / "data", data_dir, dirs_exist_ok=True)

    config = load_config(REPO_ROOT / "configs" / "default.yaml", project_root=REPO_ROOT)
    config.paths.data_dir = data_dir.resolve()
    config.paths.results_dir = (tmp_path / "results").resolve()
    config.paths.legacy_output_dir = (tmp_path / "outputs").resolve()
    config.run.mode = mode
    config.run.days = days
    config.run.fixed_regime = regime
    config.run.run_name = "test_run"
    config.validate()
    return config


def write_calendar(data_dir: Path, regimes: list[str], start_date: str = "2026-01-01") -> Path:
    dates = pd.date_range(start=start_date, periods=len(regimes), freq="D")
    calendar_df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "regime": regimes,
        }
    )
    calendar_path = data_dir / "winter_calendar.csv"
    calendar_df.to_csv(calendar_path, index=False, encoding="utf-8-sig")
    return calendar_path

