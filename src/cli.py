from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import SimulationConfig, load_config, save_config
from .data_loader import load_inputs
from .metrics import build_simulation_tables, save_tables
from .simulation import run_simulation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2지역 4연령 인플루엔자 시뮬레이션 실행기")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML 설정 파일 경로")
    parser.add_argument("--mode", choices=["fixed", "calendar", "legacy_batch"], help="실행 모드 override")
    parser.add_argument("--regime", choices=["I", "II", "III", "IV"], help="fixed mode용 regime override")
    parser.add_argument("--days", type=int, help="fixed mode용 일수 override")
    parser.add_argument("--run-name", help="results 하위 저장 폴더명")
    parser.add_argument("--no-plots", action="store_true", help="기본 그림 생성을 생략")
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    default_config_path: str | None = None,
    default_mode: str | None = None,
) -> int:
    args = parse_args(argv)
    config_path = default_config_path or args.config
    project_root = Path.cwd()
    config = load_config(config_path, project_root=project_root)
    apply_cli_overrides(config, args, default_mode=default_mode)

    if config.run.mode == "legacy_batch":
        from .legacy import run_legacy_batch

        print("[1/3] legacy 입력 데이터 로드")
        inputs = load_inputs(config, require_calendar=False)
        print("[2/3] legacy batch 실행")
        output_dir = run_legacy_batch(config, inputs)
        print(f"[3/3] legacy 출력 저장 완료: {output_dir}")
        return 0

    require_calendar = config.run.mode == "calendar"
    print("[1/4] 입력 데이터 로드")
    inputs = load_inputs(config, require_calendar=require_calendar)
    print("[2/4] 시뮬레이션 실행")
    result = run_simulation(config, inputs)
    print("[3/4] 결과 테이블 생성")
    tables = build_simulation_tables(result)
    run_dir = config.paths.results_dir / config.run.run_name
    save_tables(tables, run_dir)
    save_config(config, run_dir / "config_used.yaml")

    if config.run.create_plots:
        from .plotting import create_all_plots

        print("[4/4] 기본 그림 생성")
        create_all_plots(result, tables, run_dir)
    else:
        print("[4/4] 그림 생성 생략")

    print(f"완료: {run_dir}")
    return 0


def apply_cli_overrides(
    config: SimulationConfig,
    args: argparse.Namespace,
    default_mode: str | None = None,
) -> None:
    if default_mode is not None and args.mode is None:
        config.run.mode = default_mode
    elif args.mode is not None:
        config.run.mode = args.mode

    if args.regime is not None:
        config.run.fixed_regime = args.regime
    if args.days is not None:
        config.run.days = int(args.days)
    if args.run_name is not None:
        config.run.run_name = args.run_name
    if args.no_plots:
        config.run.create_plots = False

    config.apply_counterfactual_flags()
    config.validate()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
