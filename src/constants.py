from __future__ import annotations

REGIONS = ("Region_A", "Region_B")
AGE_GROUPS = ("0-18", "19-49", "50-64", "65+")
REGIMES = ("I", "II", "III", "IV")
LEGACY_SCENARIOS = ("I", "II", "IV")

AGE_TOKEN_MAP = {
    "0-18": "0_18",
    "19-49": "19_49",
    "50-64": "50_64",
    "65+": "65_plus",
}

NODE_ORDER = tuple((region, age_group) for region in REGIONS for age_group in AGE_GROUPS)
NODE_IDS = tuple(f"{region}_{AGE_TOKEN_MAP[age_group]}" for region, age_group in NODE_ORDER)

REGION_INDEX = {region: idx for idx, region in enumerate(REGIONS)}
AGE_INDEX = {age_group: idx for idx, age_group in enumerate(AGE_GROUPS)}
NODE_INDEX = {node_id: idx for idx, node_id in enumerate(NODE_IDS)}

REGIME_COLORS = {
    "I": "#4c78a8",
    "II": "#f58518",
    "III": "#54a24b",
    "IV": "#e45756",
}
