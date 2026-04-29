from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .constants import AGE_GROUPS, LEGACY_SCENARIOS, REGIMES
from .legacy_compat import (
    ALL_RUN_MODES,
    LEGACY_RUN_MODE,
    STANDARD_INITIAL_CONDITION_MODES,
    normalize_initial_condition_payload,
)


def _default_beta_multiplier() -> dict[str, float]:
    return {regime: 1.0 for regime in REGIMES}


def _default_susceptibility() -> dict[str, float]:
    return {age_group: 1.0 for age_group in AGE_GROUPS}


def _default_seed_by_region_age() -> dict[str, dict[str, float]]:
    return {
        "Region_A": {age_group: 100.0 for age_group in AGE_GROUPS},
        "Region_B": {age_group: 0.0 for age_group in AGE_GROUPS},
    }


@dataclass
class PathsConfig:
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    legacy_output_dir: Path = Path("outputs")
    population_file: str = "population_by_region_age.csv"
    age_contact_template: str = "age_contact_period_{regime}.csv"
    region_contact_template: str = "region_contact_period_{regime}.csv"
    winter_calendar_file: str = "winter_calendar.csv"
    sample_calendar_file: str = "sample_winter_calendar.csv"

    def resolve(self, project_root: Path) -> None:
        self.data_dir = _resolve_path(project_root, self.data_dir)
        self.results_dir = _resolve_path(project_root, self.results_dir)
        self.legacy_output_dir = _resolve_path(project_root, self.legacy_output_dir)

    def population_path(self) -> Path:
        return self.data_dir / self.population_file

    def age_contact_path(self, regime: str) -> Path:
        return self.data_dir / self.age_contact_template.format(regime=regime)

    def region_contact_path(self, regime: str) -> Path:
        return self.data_dir / self.region_contact_template.format(regime=regime)

    def winter_calendar_path(self) -> Path:
        return self.data_dir / self.winter_calendar_file

    def sample_calendar_path(self) -> Path:
        return self.data_dir / self.sample_calendar_file


@dataclass
class SusceptibilityCsvConfig:
    enabled: bool = False
    source_csv: str = ""
    encoding: str = "utf-8"
    start_date: str = "2025-12-01"
    end_date: str = "2026-02-23"
    metric: str = "mean"
    preseason_start_date: str = "2025-09-01"
    preseason_end_date: str = "2025-11-24"
    normalize_to: str = "mean"
    power: float = 1.0


@dataclass
class TimeBetaCsvConfig:
    enabled: bool = False
    source_csv: str = ""
    encoding: str = "cp949"
    normalize_to: str = "mean"
    age_weighting: str = "population"
    power: float = 1.0


@dataclass
class ModelConfig:
    beta0: float = 0.040
    beta_multiplier: dict[str, float] = field(default_factory=_default_beta_multiplier)
    susceptibility: dict[str, float] = field(default_factory=_default_susceptibility)
    susceptibility_from_csv: SusceptibilityCsvConfig = field(default_factory=SusceptibilityCsvConfig)
    time_beta_from_csv: TimeBetaCsvConfig = field(default_factory=TimeBetaCsvConfig)
    reinfection_susceptibility_scale: float = 1.0
    latent_period_days: float = 2.0
    infectious_period_days: float = 5.0
    waning_rate_per_day: float = 0.0
    infection_update: str = "probability"
    enable_reinfection: bool = True


@dataclass
class InitialConditionConfig:
    mode: str = "same_prevalence"
    seed_compartment: str = "I"
    same_prevalence: float = 1.0e-4
    same_prevalence_by_age: dict[str, float] = field(default_factory=dict)
    equal_seed_count: float = 100.0
    seed_by_region_age: dict[str, dict[str, float]] = field(default_factory=_default_seed_by_region_age)


@dataclass
class CounterfactualConfig:
    no_holiday: bool = False
    no_cross_region: bool = False
    no_reinfection: bool = False


@dataclass
class LegacyConfig:
    write_legacy_outputs: bool = True
    scenario_order: list[str] = field(default_factory=lambda: list(LEGACY_SCENARIOS))


@dataclass
class RunConfig:
    mode: str = "fixed"
    run_name: str = "default_run"
    fixed_regime: str = "I"
    days: int = 180
    create_plots: bool = True
    compare_initial_conditions: bool = False


@dataclass
class SimulationConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    initial_conditions: InitialConditionConfig = field(default_factory=InitialConditionConfig)
    counterfactual: CounterfactualConfig = field(default_factory=CounterfactualConfig)
    legacy: LegacyConfig = field(default_factory=LegacyConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def resolve_paths(self, project_root: Path) -> None:
        self.paths.resolve(project_root)

    def apply_counterfactual_flags(self) -> None:
        if self.counterfactual.no_reinfection:
            self.model.enable_reinfection = False
            self.model.waning_rate_per_day = 0.0

    def validate(self) -> None:
        if self.run.mode not in ALL_RUN_MODES:
            raise ValueError(f"Unsupported run.mode: {self.run.mode}")
        if self.run.mode in {LEGACY_RUN_MODE, "fixed"} and self.run.fixed_regime not in REGIMES:
            raise ValueError(f"Unsupported fixed regime: {self.run.fixed_regime}")
        if self.model.infection_update not in {"probability", "linear"}:
            raise ValueError("model.infection_update must be 'probability' or 'linear'.")
        if self.initial_conditions.mode not in STANDARD_INITIAL_CONDITION_MODES:
            raise ValueError(
                "initial_conditions.mode must be one of: "
                "same_prevalence, seed_by_region_age, equal_seed, equal_absolute_seed."
            )
        if self.initial_conditions.seed_compartment not in {"E", "I"}:
            raise ValueError("initial_conditions.seed_compartment must be 'E' or 'I'.")
        if self.initial_conditions.equal_seed_count < 0.0:
            raise ValueError("initial_conditions.equal_seed_count must be non-negative.")
        if self.model.latent_period_days <= 0 or self.model.infectious_period_days <= 0:
            raise ValueError("latent_period_days and infectious_period_days must be positive.")
        if any(regime not in self.model.beta_multiplier for regime in REGIMES):
            raise ValueError("beta_multiplier must provide values for I, II, III, and IV.")
        if any(age_group not in self.model.susceptibility for age_group in AGE_GROUPS):
            raise ValueError("susceptibility must provide values for all four age groups.")
        self._validate_susceptibility_csv()
        self._validate_time_beta_csv()

    def _validate_susceptibility_csv(self) -> None:
        csv_config = self.model.susceptibility_from_csv
        if not csv_config.enabled:
            return
        if not csv_config.source_csv:
            raise ValueError("model.susceptibility_from_csv.source_csv is required when enabled.")
        if csv_config.metric not in {"mean", "peak", "preseason_ratio"}:
            raise ValueError(
                "model.susceptibility_from_csv.metric must be one of: mean, peak, preseason_ratio."
            )
        if csv_config.normalize_to not in {"mean", *AGE_GROUPS}:
            raise ValueError(
                "model.susceptibility_from_csv.normalize_to must be 'mean' or one of the four age groups."
            )
        if csv_config.power <= 0.0:
            raise ValueError("model.susceptibility_from_csv.power must be positive.")

    def _validate_time_beta_csv(self) -> None:
        csv_config = self.model.time_beta_from_csv
        if not csv_config.enabled:
            return
        if self.run.mode != "calendar":
            raise ValueError("model.time_beta_from_csv can only be used with calendar mode.")
        if not csv_config.source_csv:
            raise ValueError("model.time_beta_from_csv.source_csv is required when enabled.")
        if csv_config.normalize_to not in {"mean", "max"}:
            raise ValueError("model.time_beta_from_csv.normalize_to must be 'mean' or 'max'.")
        if csv_config.age_weighting not in {"population", "equal"}:
            raise ValueError("model.time_beta_from_csv.age_weighting must be 'population' or 'equal'.")
        if csv_config.power <= 0.0:
            raise ValueError("model.time_beta_from_csv.power must be positive.")


def load_config(config_path: str | Path, project_root: str | Path | None = None) -> SimulationConfig:
    config_path = Path(config_path)
    project_root = Path(project_root) if project_root is not None else Path.cwd()
    default_payload = asdict(SimulationConfig())
    user_payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    merged = _deep_merge(default_payload, user_payload)

    initial_condition_payload = normalize_initial_condition_payload(merged["initial_conditions"])
    model_payload = dict(merged["model"])
    model_payload["susceptibility_from_csv"] = SusceptibilityCsvConfig(
        **model_payload.get("susceptibility_from_csv", {})
    )
    model_payload["time_beta_from_csv"] = TimeBetaCsvConfig(
        **model_payload.get("time_beta_from_csv", {})
    )

    config = SimulationConfig(
        paths=PathsConfig(**merged["paths"]),
        model=ModelConfig(**model_payload),
        initial_conditions=InitialConditionConfig(**initial_condition_payload),
        counterfactual=CounterfactualConfig(**merged["counterfactual"]),
        legacy=LegacyConfig(**merged["legacy"]),
        run=RunConfig(**merged["run"]),
    )
    config.resolve_paths(project_root)
    _apply_susceptibility_from_csv(config, project_root)
    config.apply_counterfactual_flags()
    config.validate()
    return config


def save_config(config: SimulationConfig, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_serializable(asdict(config))
    output_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _resolve_data_input_path(project_root: Path, data_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    project_path = (project_root / path).resolve()
    if project_path.exists():
        return project_path
    return (data_dir / path).resolve()


def _apply_susceptibility_from_csv(config: SimulationConfig, project_root: Path) -> None:
    csv_config = config.model.susceptibility_from_csv
    if not csv_config.enabled:
        return

    from .susceptibility import derive_age_susceptibility_from_csv

    csv_path = _resolve_data_input_path(project_root, config.paths.data_dir, csv_config.source_csv)
    config.model.susceptibility = derive_age_susceptibility_from_csv(
        weekly_csv_path=csv_path,
        encoding=csv_config.encoding,
        start_date=csv_config.start_date,
        end_date=csv_config.end_date,
        metric=csv_config.metric,
        preseason_start_date=csv_config.preseason_start_date,
        preseason_end_date=csv_config.preseason_end_date,
        normalize_to=csv_config.normalize_to,
        power=csv_config.power,
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    return value
