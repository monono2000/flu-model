from __future__ import annotations

import numpy as np


def decompose_daily_flows(
    raw_contrib: np.ndarray,
    new_first_infections: np.ndarray,
    new_reinfections: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """source->target 일일 신규감염 흐름을 분해한다."""
    flow_first = np.zeros_like(raw_contrib)
    flow_reinfection = np.zeros_like(raw_contrib)

    _, _, target_region_count, target_age_count = raw_contrib.shape
    for target_region in range(target_region_count):
        for target_age in range(target_age_count):
            raw_slice = raw_contrib[:, :, target_region, target_age]
            raw_total = float(raw_slice.sum())
            if raw_total <= 0.0:
                continue
            weight = raw_slice / raw_total
            flow_first[:, :, target_region, target_age] = (
                weight * new_first_infections[target_region, target_age]
            )
            flow_reinfection[:, :, target_region, target_age] = (
                weight * new_reinfections[target_region, target_age]
            )

    flow_total = flow_first + flow_reinfection
    return flow_first, flow_reinfection, flow_total

