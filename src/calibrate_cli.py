from __future__ import annotations

import argparse
from pathlib import Path

from .calibration import (
    CalibrationSearchSpace,
    SusceptibilitySearchSpace,
    refine_susceptibility_search,
    run_calendar_grid_calibration,
    save_calibration_outputs,
)
from .config import load_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="관측 인플루엔자 주간 발병률 기반 calendar calibration")
    parser.description = "Run calendar calibration against a synthetic weekly target series."
    parser.add_argument("--config", default="configs/calendar_baseline.yaml")
    parser.add_argument(
        "--target-csv",
        "--observed-csv",
        dest="target_csv",
        default="data/synthetic_target_influenza_2025_2026.csv",
        help="Path to a synthetic weekly target CSV. --observed-csv is kept as a compatibility alias.",
    )
    parser.add_argument("--run-name", default="synthetic_target_calibration")
    parser.add_argument("--compare-start", default="2025-12-01")
    parser.add_argument("--compare-end", default="2026-02-23")
    parser.add_argument("--refine-susceptibility", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path.cwd()
    config = load_config(args.config, project_root=root)
    config.run.mode = "calendar"

    print("[1/4] 관측 CSV 로드 및 grid calibration 시작")
    best_record = run_calendar_grid_calibration(
        base_config=config,
        observed_csv_path=args.target_csv,
        search_space=CalibrationSearchSpace(),
        compare_start_date=args.compare_start,
        compare_end_date=args.compare_end,
    )
    if args.refine_susceptibility:
        print("[1.5/4] 연령별 susceptibility refinement")
        best_record = refine_susceptibility_search(
            seed_record=best_record,
            observed_csv_path=args.target_csv,
            compare_start_date=args.compare_start,
            compare_end_date=args.compare_end,
            search_space=SusceptibilitySearchSpace(),
        )

    run_dir = config.paths.results_dir / args.run_name
    print("[2/4] calibration 결과 저장")
    save_calibration_outputs(run_dir, best_record)

    print("[3/4] 최적 파라미터")
    print(f"score={best_record['score']:.6f}")
    print(f"beta0={best_record['config'].model.beta0}")
    print(f"same_prevalence={best_record['config'].initial_conditions.same_prevalence}")
    print(f"beta_multiplier={best_record['config'].model.beta_multiplier}")
    print(f"susceptibility={best_record['config'].model.susceptibility}")

    print(f"[4/4] 완료: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
