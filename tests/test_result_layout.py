from __future__ import annotations

import json
from pathlib import Path

from src.data_loader import load_inputs
from src.metrics import build_simulation_tables, save_tables
from src.result_layout import build_output_layout
from src.results_migration import migrate_run_directory
from src.simulation import run_simulation
from tests.helpers import build_test_config


def test_save_tables_uses_structured_layout(tmp_path):
    config = build_test_config(tmp_path, mode="fixed", days=5, regime="I")
    inputs = load_inputs(config, require_calendar=False)
    result = run_simulation(config, inputs)
    tables = build_simulation_tables(result)

    run_dir = tmp_path / "results" / "layout_case"
    save_tables(tables, run_dir)
    layout = build_output_layout(run_dir)

    assert (layout.meta_dir / "summary_metrics.json").exists()
    assert (layout.tables_dir / "states_long.csv").exists()
    assert (layout.tables_dir / "region_daily_metrics.csv").exists()
    assert (layout.tables_dir / "region_group_daily_metrics.csv").exists()
    assert (layout.tables_dir / "region_group_final_summary.csv").exists()
    assert (layout.tables_dir / "age_group_summary.csv").exists()
    assert (layout.root / "manifest.json").exists()

    manifest = json.loads((layout.root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["directories"]["meta"] == "meta/"
    assert manifest["primary_files"]["summary"] == "meta/summary_metrics.json"


def test_migrate_run_directory_moves_root_artifacts(tmp_path):
    run_dir = tmp_path / "results" / "legacy_style"
    run_dir.mkdir(parents=True)
    (run_dir / "config_used.yaml").write_text("run: test\n", encoding="utf-8")
    (run_dir / "summary_metrics.json").write_text("{}", encoding="utf-8")
    (run_dir / "overall_daily_metrics.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (run_dir / "timeseries_overview.png").write_bytes(b"png")
    snapshot_dir = run_dir / "network_snapshots"
    snapshot_dir.mkdir()
    (snapshot_dir / "001_start.png").write_bytes(b"png")

    migrate_run_directory(run_dir)
    layout = build_output_layout(run_dir)

    assert not (run_dir / "config_used.yaml").exists()
    assert not (run_dir / "overall_daily_metrics.csv").exists()
    assert (layout.meta_dir / "config_used.yaml").exists()
    assert (layout.meta_dir / "summary_metrics.json").exists()
    assert (layout.tables_dir / "overall_daily_metrics.csv").exists()
    assert (layout.plots_dir / "timeseries_overview.png").exists()
    assert (layout.snapshots_dir / "001_start.png").exists()
