from __future__ import annotations

import numpy as np

from src.data_loader import load_inputs
from src.legacy_compat import normalize_initial_condition_payload
from src.simulation import run_simulation
from tests.helpers import build_test_config, write_calendar


def test_population_conservation(tmp_path):
    config = build_test_config(tmp_path, mode="fixed", days=25, regime="I")
    inputs = load_inputs(config, require_calendar=False)
    result = run_simulation(config, inputs)

    total = sum(result.state_history[name] for name in ["S0", "E0", "I0", "R", "S1", "E1", "I1"])
    expected = inputs.population[np.newaxis, :, :]
    assert np.allclose(total, expected, atol=1.0e-8)


def test_zero_infection_stability(tmp_path):
    config = build_test_config(tmp_path, mode="fixed", days=20, regime="II")
    config.initial_conditions.mode = "same_prevalence"
    config.initial_conditions.same_prevalence = 0.0
    config.initial_conditions.same_prevalence_by_age = {}

    inputs = load_inputs(config, require_calendar=False)
    result = run_simulation(config, inputs)

    assert np.allclose(result.new_first_infections, 0.0)
    assert np.allclose(result.new_reinfections, 0.0)
    assert np.allclose(result.state_history["E0"], 0.0)
    assert np.allclose(result.state_history["I0"], 0.0)
    assert np.allclose(result.state_history["E1"], 0.0)
    assert np.allclose(result.state_history["I1"], 0.0)
    assert np.allclose(result.state_history["R"], 0.0)


def test_fixed_vs_calendar_consistency(tmp_path):
    fixed_config = build_test_config(tmp_path / "fixed", mode="fixed", days=18, regime="I")
    calendar_config = build_test_config(tmp_path / "calendar", mode="calendar", days=18, regime="I")
    write_calendar(calendar_config.paths.data_dir, ["I"] * 18)

    fixed_inputs = load_inputs(fixed_config, require_calendar=False)
    calendar_inputs = load_inputs(calendar_config, require_calendar=True)
    fixed_result = run_simulation(fixed_config, fixed_inputs)
    calendar_result = run_simulation(calendar_config, calendar_inputs)

    for compartment in ["S0", "E0", "I0", "R", "S1", "E1", "I1"]:
        assert np.allclose(
            fixed_result.state_history[compartment],
            calendar_result.state_history[compartment],
            atol=1.0e-10,
        )
    assert np.allclose(fixed_result.new_first_infections, calendar_result.new_first_infections, atol=1.0e-10)
    assert np.allclose(fixed_result.new_reinfections, calendar_result.new_reinfections, atol=1.0e-10)


def test_no_reinfection_reduction(tmp_path):
    config = build_test_config(tmp_path, mode="fixed", days=22, regime="II")
    config.model.waning_rate_per_day = 0.01
    config.counterfactual.no_reinfection = True
    config.apply_counterfactual_flags()
    config.validate()

    inputs = load_inputs(config, require_calendar=False)
    result = run_simulation(config, inputs)

    assert np.allclose(result.new_reinfections, 0.0)
    assert np.allclose(result.state_history["S1"], 0.0)
    assert np.allclose(result.state_history["E1"], 0.0)
    assert np.allclose(result.state_history["I1"], 0.0)


def test_legacy_equal_absolute_alias_maps_to_standard_mode():
    normalized = normalize_initial_condition_payload(
        {
            "mode": "legacy_equal_absolute",
            "legacy_equal_absolute": 100.0,
        }
    )

    assert normalized["mode"] == "equal_absolute_seed"
    assert normalized["equal_seed_count"] == 100.0
    assert "legacy_equal_absolute" not in normalized
