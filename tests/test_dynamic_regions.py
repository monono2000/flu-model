from __future__ import annotations

import pandas as pd
import numpy as np

from src.data_loader import load_inputs
from src.simulation import run_simulation
from tests.helpers import build_test_config
from tests.helpers import write_calendar


def test_expand_coarse_region_matrix_to_fine_regions(tmp_path):
    config = build_test_config(tmp_path, mode="fixed", days=5, regime="I")
    config.paths.population_file = "population_fine_regions.csv"

    population_rows = []
    region_payload = {
        "Seoul_Jongno": ("Region_A", [100, 200, 300, 400]),
        "Seoul_Junggu": ("Region_A", [300, 400, 500, 600]),
        "Wonju_Munmak": ("Region_B", [100, 200, 300, 400]),
        "Wonju_Jijeong": ("Region_B", [100, 200, 300, 400]),
    }
    for region, (region_group, ages) in region_payload.items():
        for age_group, population in zip(["0-18", "19-49", "50-64", "65+"], ages, strict=True):
            population_rows.append(
                {
                    "region": region,
                    "region_group": region_group,
                    "age_group": age_group,
                    "population": population,
                }
            )

    pd.DataFrame(population_rows).to_csv(
        config.paths.data_dir / config.paths.population_file,
        index=False,
        encoding="utf-8-sig",
    )

    coarse_region_matrix = pd.DataFrame(
        [[4.0, 1.0], [2.0, 3.0]],
        index=["Region_A", "Region_B"],
        columns=["Region_A", "Region_B"],
    )
    for regime in ["I", "II", "III", "IV"]:
        coarse_region_matrix.to_csv(
            config.paths.data_dir / f"region_contact_period_{regime}.csv",
            encoding="utf-8-sig",
        )

    inputs = load_inputs(config, require_calendar=False)

    assert inputs.regions == ("Seoul_Jongno", "Seoul_Junggu", "Wonju_Munmak", "Wonju_Jijeong")
    assert inputs.region_groups == ("Region_A", "Region_A", "Region_B", "Region_B")

    matrix = inputs.region_matrices["I"]
    assert matrix.shape == (4, 4)
    assert matrix[:, 0].tolist() == [1.4285714285714286, 2.5714285714285716, 1.0, 1.0]
    assert matrix[:, 2].tolist() == [0.35714285714285715, 0.6428571428571429, 1.5, 1.5]


def test_equal_absolute_seed_distributes_by_group_age_in_standard_calendar_mode(tmp_path):
    config = build_test_config(tmp_path, mode="calendar", days=3, regime="I")
    config.paths.population_file = "population_fine_regions.csv"
    config.initial_conditions.mode = "equal_absolute_seed"
    config.initial_conditions.equal_seed_count = 100.0

    population_rows = []
    region_payload = {
        "Seoul_Jongno": ("Region_A", [100, 200, 300, 400]),
        "Seoul_Junggu": ("Region_A", [300, 400, 500, 600]),
        "Wonju_Munmak": ("Region_B", [100, 200, 300, 400]),
        "Wonju_Jijeong": ("Region_B", [100, 200, 300, 400]),
    }
    for region, (region_group, ages) in region_payload.items():
        for age_group, population in zip(["0-18", "19-49", "50-64", "65+"], ages, strict=True):
            population_rows.append(
                {
                    "region": region,
                    "region_group": region_group,
                    "age_group": age_group,
                    "population": population,
                }
            )

    pd.DataFrame(population_rows).to_csv(
        config.paths.data_dir / config.paths.population_file,
        index=False,
        encoding="utf-8-sig",
    )

    write_calendar(config.paths.data_dir, ["I", "I", "I"])
    inputs = load_inputs(config, require_calendar=True)
    result = run_simulation(config, inputs)

    initial_seed = result.state_history["I0"][0]
    assert np.allclose(initial_seed[0], [25.0, 33.333333333333336, 37.5, 40.0])
    assert np.allclose(initial_seed[1], [75.0, 66.66666666666667, 62.5, 60.0])
    assert np.allclose(initial_seed[2], [50.0, 50.0, 50.0, 50.0])
    assert np.allclose(initial_seed[3], [50.0, 50.0, 50.0, 50.0])

    for age_idx in range(initial_seed.shape[1]):
        seoul_total = float(initial_seed[[0, 1], age_idx].sum())
        wonju_total = float(initial_seed[[2, 3], age_idx].sum())
        assert np.isclose(seoul_total, 100.0)
        assert np.isclose(wonju_total, 100.0)
