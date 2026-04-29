from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunOutputLayout:
    root: Path
    meta_dir: Path
    tables_dir: Path
    plots_dir: Path
    snapshots_dir: Path


def build_output_layout(run_dir: str | Path) -> RunOutputLayout:
    root = Path(run_dir)
    return RunOutputLayout(
        root=root,
        meta_dir=root / "meta",
        tables_dir=root / "tables",
        plots_dir=root / "plots",
        snapshots_dir=root / "plots" / "network_snapshots",
    )


def ensure_output_layout(run_dir: str | Path) -> RunOutputLayout:
    layout = build_output_layout(run_dir)
    layout.root.mkdir(parents=True, exist_ok=True)
    layout.meta_dir.mkdir(parents=True, exist_ok=True)
    layout.tables_dir.mkdir(parents=True, exist_ok=True)
    layout.plots_dir.mkdir(parents=True, exist_ok=True)
    return layout


def write_run_manifest(
    layout: RunOutputLayout,
    run_type: str,
    primary_files: dict[str, str],
    extra_sections: dict[str, dict[str, str]] | None = None,
) -> None:
    manifest = {
        "layout_version": 2,
        "run_type": run_type,
        "directories": {
            "meta": "meta/",
            "tables": "tables/",
            "plots": "plots/",
        },
        "primary_files": primary_files,
    }
    if extra_sections:
        manifest["sections"] = extra_sections
    (layout.root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def update_manifest_sections(
    layout: RunOutputLayout,
    sections: dict[str, dict[str, str]],
) -> None:
    manifest_path = layout.root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sections:
        manifest["sections"] = sections
    else:
        manifest.pop("sections", None)

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
