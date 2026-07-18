import os
from typing import List

CORS_ALLOWED_ORIGINS = os.environ.get(
    "ORBIT_CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
).split(",")

SIM_SPEED_FACTOR = float(os.environ.get("ORBIT_SIM_SPEED_FACTOR", "0.4"))
DEFAULT_TICK_INTERVAL_S = float(os.environ.get("ORBIT_DEFAULT_TICK_INTERVAL_S", "1.0"))

OVERRIDE_TICKS = 15
SUGGESTION_TTL_TICKS = 20
COPILOT_SUGGESTIONS_MAX_SIZE = 100

# Canonical corridor paths
DOWN_PATH: List[str] = [f"edge-{k}-{k+1}" for k in range(195)] + ["edge-195-999"]
UP_PATH:   List[str] = ["edge-195-999"] + [f"edge-{k}-{k+1}" for k in reversed(range(195))]

# Fleet config defaults
TRAIN_TYPES = ["Vande Bharat", "Rajdhani", "Superfast", "Express",
               "Local", "Suburban", "Passenger", "Freight (WAG-9)"]
PRIORITY_MAP = {
    "Vande Bharat": 10, "Rajdhani": 10, "Superfast": 8,
    "Express": 6, "Local": 5, "Suburban": 5,
    "Passenger": 3, "Freight (WAG-9)": 2,
    "Freight (Wag-9)": 2, "Goods": 1, "Freight": 2, "Wag-9 Goods": 1,
}

from ai.config import SECTION_LENGTH_KM  # single source of truth for corridor distance

DEADLINE_BUFFER_MULTIPLIER = 2.5  # matches this repo's own Phase-1 schedule
                                   # design intent — see ai/config.py's
                                   # GOODS_100/SF_101 comments: "~2.5x
                                   # realistic travel time"

def compute_deadline(start_time: int, max_speed: int) -> int:
    """
    Derive an achievable deadline (sim-steps, ≈ minutes) from real train
    physics instead of a flat guess. 1 sim-step ≈ 1 real minute, so:
        raw_transit_steps = SECTION_LENGTH_KM / max_speed * 60
    Then apply a generous-but-tight buffer (consistent with this repo's
    own achievable-schedule design) so it's realistic, not just the
    bare minimum transit time.
    """
    if max_speed <= 0:
        raise ValueError("max_speed must be a positive number of km/h")
    raw_transit_steps = (SECTION_LENGTH_KM / max_speed) * 60
    return start_time + round(raw_transit_steps * DEADLINE_BUFFER_MULTIPLIER)

DEFAULT_DEADLINE = compute_deadline(0, 110)  # matches the max_speed=110
                                              # already assumed by the
                                              # fallback blocks below
