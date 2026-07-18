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
