from __future__ import annotations

import shutil
from pathlib import Path

from .result_layout import ensure_output_layout, write_run_manifest


META_FILENAMES = {
    "config_used.yaml",
    "calibrated_config.yaml",
    "summary_metrics.json",
    "calibration_summary.json",
}
TABLE_SUFFIXES = {".csv"}
PLOT_SUFFIXES = {".png", ".gif"}
RENAMED_TABLE_FILENAMES = {
    "observed_weekly_long.csv": "target_weekly_long.csv",
    "observed_vs_model_weekly.csv": "target_vs_model_weekly.csv",
}


def migrate_results_tree(results_dir: str | Path) -> None:
    results_path = Path(results_dir)
    if not results_path.exists():
        return
    for run_dir in results_path.iterdir():
        if run_dir.is_dir():
            migrate_run_directory(run_dir)


def migrate_run_directory(run_dir: str | Path) -> None:
    layout = ensure_output_layout(run_dir)
    root = layout.root

    for item in list(root.iterdir()):
        if item.name in {"meta", "tables", "plots"}:
            continue
        if item.name == "network_snapshots" and item.is_dir():
            _move_directory_contents(item, layout.snapshots_dir)
            _remove_if_empty(item)
            continue
        if item.is_file():
            target_path = _target_path_for_file(item, layout)
            if target_path is not None:
                _move_file(item, target_path)

    _normalize_existing_table_names(layout)
    run_type = "calibration" if (layout.meta_dir / "calibration_summary.json").exists() else "simulation"
    primary_files = _build_primary_files(layout, run_type)
    extra_sections = _build_extra_sections(layout)
    write_run_manifest(layout, run_type=run_type, primary_files=primary_files, extra_sections=extra_sections)


def _target_path_for_file(file_path: Path, layout) -> Path | None:
    if file_path.name == "manifest.json":
        return layout.root / "manifest.json"
    if file_path.name in META_FILENAMES:
        return layout.meta_dir / file_path.name
    if file_path.suffix.lower() in TABLE_SUFFIXES:
        return layout.tables_dir / RENAMED_TABLE_FILENAMES.get(file_path.name, file_path.name)
    if file_path.suffix.lower() in PLOT_SUFFIXES:
        return layout.plots_dir / file_path.name
    return None


def _move_directory_contents(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in list(source_dir.iterdir()):
        destination = target_dir / item.name
        if item.is_dir():
            _move_directory_contents(item, destination)
            _remove_if_empty(item)
        else:
            _move_file(item, destination)


def _move_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return
    if destination.exists():
        if destination.is_file():
            destination.unlink()
        else:
            shutil.rmtree(destination)
    try:
        source.replace(destination)
    except PermissionError:
        shutil.copy2(source, destination)
        try:
            source.unlink()
        except PermissionError:
            pass


def _remove_if_empty(path: Path) -> None:
    if path.exists() and path.is_dir() and not any(path.iterdir()):
        path.rmdir()


def _build_primary_files(layout, run_type: str) -> dict[str, str]:
    if run_type == "calibration":
        target_weekly_name = (
            "target_weekly_long.csv"
            if (layout.tables_dir / "target_weekly_long.csv").exists()
            else "observed_weekly_long.csv"
        )
        comparison_name = (
            "target_vs_model_weekly.csv"
            if (layout.tables_dir / "target_vs_model_weekly.csv").exists()
            else "observed_vs_model_weekly.csv"
        )
        return {
            "config": "meta/calibrated_config.yaml",
            "summary": "meta/calibration_summary.json",
            "simulation_summary": "meta/summary_metrics.json",
            "target_weekly": f"tables/{target_weekly_name}",
            "model_weekly": "tables/model_weekly_long.csv",
            "comparison": f"tables/{comparison_name}",
        }
    return {
        "config": "meta/config_used.yaml",
        "summary": "meta/summary_metrics.json",
        "overall_daily_metrics": "tables/overall_daily_metrics.csv",
        "node_daily_metrics": "tables/node_daily_metrics.csv",
    }


def _build_extra_sections(layout) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {}
    if layout.tables_dir.exists():
        sections["tables"] = {
            path.name: f"tables/{path.name}"
            for path in sorted(layout.tables_dir.iterdir())
            if path.is_file()
        }
    if layout.plots_dir.exists():
        plot_entries = {
            path.name: f"plots/{path.name}"
            for path in sorted(layout.plots_dir.iterdir())
            if path.is_file()
        }
        if layout.snapshots_dir.exists():
            plot_entries["network_snapshots"] = "plots/network_snapshots/"
        if plot_entries:
            sections["plots"] = plot_entries
    return sections


def _normalize_existing_table_names(layout) -> None:
    for old_name, new_name in RENAMED_TABLE_FILENAMES.items():
        old_path = layout.tables_dir / old_name
        new_path = layout.tables_dir / new_name
        if old_path.exists() and not new_path.exists():
            old_path.replace(new_path)
