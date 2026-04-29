from __future__ import annotations

import pandas as pd

from src.data_loader import load_inputs
from src.initial_condition_batch import run_initial_condition_batch
from tests.helpers import build_test_config


def test_initial_condition_batch_writes_both_scenarios_and_comparison_tables(tmp_path):
    config = build_test_config(tmp_path, mode="fixed", days=8, regime="I")
    config.run.run_name = "initial_condition_comparison"
    config.run.create_plots = False
    config.run.compare_initial_conditions = True
    config.initial_conditions.same_prevalence = 0.0002
    config.initial_conditions.equal_seed_count = 25.0

    inputs = load_inputs(config, require_calendar=False)
    output_dir = run_initial_condition_batch(config, inputs)

    overall_summary_path = output_dir / "tables" / "initial_condition_overall_summary.csv"
    region_summary_path = output_dir / "tables" / "region_group_final_summary_by_initial_condition.csv"
    age_summary_path = output_dir / "tables" / "age_group_summary_by_initial_condition.csv"

    assert overall_summary_path.exists()
    assert region_summary_path.exists()
    assert age_summary_path.exists()
    assert (output_dir / "same_prevalence" / "meta" / "config_used.yaml").exists()
    assert (output_dir / "equal_absolute_seed" / "meta" / "config_used.yaml").exists()

    overall_df = pd.read_csv(overall_summary_path)
    region_df = pd.read_csv(region_summary_path)
    age_df = pd.read_csv(age_summary_path)

    assert set(overall_df["initial_condition_mode"]) == {"same_prevalence", "equal_absolute_seed"}
    assert set(region_df["initial_condition_mode"]) == {"same_prevalence", "equal_absolute_seed"}
    assert set(age_df["initial_condition_mode"]) == {"same_prevalence", "equal_absolute_seed"}
