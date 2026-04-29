from __future__ import annotations

from typing import Any


LEGACY_RUN_MODE = "legacy_batch"
STANDARD_RUN_MODES = {"fixed", "calendar"}
ALL_RUN_MODES = STANDARD_RUN_MODES | {LEGACY_RUN_MODE}

STANDARD_INITIAL_CONDITION_MODES = {
    "same_prevalence",
    "seed_by_region_age",
    "equal_seed",
    "equal_absolute_seed",
}
LEGACY_INITIAL_CONDITION_MODE_ALIASES = {
    "seoul_seed": "seed_by_region_age",
    "legacy_equal_absolute": "equal_absolute_seed",
}


def normalize_initial_condition_mode(mode: str) -> str:
    return LEGACY_INITIAL_CONDITION_MODE_ALIASES.get(mode, mode)


def normalize_initial_condition_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    mode = normalized.get("mode")
    if isinstance(mode, str):
        normalized["mode"] = normalize_initial_condition_mode(mode)
    if "legacy_equal_absolute" in normalized:
        normalized["equal_seed_count"] = normalized["legacy_equal_absolute"]
        normalized.pop("legacy_equal_absolute", None)
    return normalized
