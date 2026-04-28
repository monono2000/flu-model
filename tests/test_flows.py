from __future__ import annotations

import numpy as np

from src.data_loader import load_inputs
from src.metrics import compute_cross_region_flow
from src.simulation import run_simulation
from tests.helpers import build_test_config


def test_no_cross_region_check(tmp_path):
    config = build_test_config(tmp_path, mode="fixed", days=20, regime="II")
    config.counterfactual.no_cross_region = True

    inputs = load_inputs(config, require_calendar=False)
    result = run_simulation(config, inputs)

    cross_region_flow = compute_cross_region_flow(result.flow_total)
    assert np.allclose(cross_region_flow, 0.0, atol=1.0e-10)


def test_source_target_flow_consistency(tmp_path):
    config = build_test_config(tmp_path, mode="fixed", days=16, regime="I")
    inputs = load_inputs(config, require_calendar=False)
    result = run_simulation(config, inputs)

    source_summed_flow = result.flow_total.sum(axis=(1, 2))
    target_new_infections = result.new_first_infections + result.new_reinfections
    assert np.allclose(source_summed_flow, target_new_infections, atol=1.0e-10)


def test_region_a_seed_isolation_sanity(tmp_path):
    config = build_test_config(tmp_path, mode="fixed", days=20, regime="I")
    config.counterfactual.no_cross_region = True
    config.initial_conditions.mode = "seoul_seed"
    config.initial_conditions.seed_by_region_age = {
        "Region_A": {
            "0-18": 100.0,
            "19-49": 100.0,
            "50-64": 100.0,
            "65+": 100.0,
        },
        "Region_B": {
            "0-18": 0.0,
            "19-49": 0.0,
            "50-64": 0.0,
            "65+": 0.0,
        },
    }

    inputs = load_inputs(config, require_calendar=False)
    result = run_simulation(config, inputs)

    region_b_first = inputs.population[1] - result.state_history["S0"][:, 1, :]
    region_b_active = result.state_history["I0"][:, 1, :] + result.state_history["I1"][:, 1, :]
    assert np.allclose(region_b_first, 0.0, atol=1.0e-10)
    assert np.allclose(region_b_active, 0.0, atol=1.0e-10)
