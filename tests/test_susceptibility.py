from __future__ import annotations

import shutil

from src.config import load_config
from src.susceptibility import derive_age_susceptibility_from_csv
from tests.helpers import REPO_ROOT


def test_derive_age_susceptibility_from_csv_mean():
    susceptibility = derive_age_susceptibility_from_csv(
        REPO_ROOT / "data" / "synthetic_target_influenza_2025_2026.csv",
        encoding="utf-8",
        start_date="2025-12-01",
        end_date="2026-02-23",
        metric="mean",
        normalize_to="mean",
    )

    assert susceptibility["0-18"] > susceptibility["19-49"] > susceptibility["50-64"] > susceptibility["65+"]
    assert abs(sum(susceptibility.values()) / len(susceptibility) - 1.0) < 1.0e-10


def test_load_config_applies_susceptibility_from_csv(tmp_path):
    shutil.copytree(REPO_ROOT / "data", tmp_path / "data", dirs_exist_ok=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  susceptibility_from_csv:",
                "    enabled: true",
                "    source_csv: data/synthetic_target_influenza_2025_2026.csv",
                "    encoding: utf-8",
                '    start_date: "2025-12-01"',
                '    end_date: "2026-02-23"',
                "    metric: mean",
                "    normalize_to: mean",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path, project_root=tmp_path)

    assert config.model.susceptibility["0-18"] > config.model.susceptibility["19-49"]
    assert config.model.susceptibility["19-49"] > config.model.susceptibility["50-64"]
    assert config.model.susceptibility["50-64"] > config.model.susceptibility["65+"]
