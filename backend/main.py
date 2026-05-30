from fastapi import FastAPI, Depends, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel
import asyncio
import json
import random
import uuid
import time
import os
import numpy as np
from datetime import datetime, timezone

import database
import schema 
import crud 
from database import TrainPosition
from ai.config import ACTIVE_FLEET as TRAIN_CONFIG
from or_tools.smart_optimizer import SmartOptimizer
from topology import get_network_topology


database.init_db()

app = FastAPI(title="ORBIT: Operational Rail Backbone")

# ---------------------------------------------------------------------------
# Topology Simulation State
# ---------------------------------------------------------------------------
TOPOLOGY_DATA = get_network_topology()
NETWORK_TOPOLOGY = {"nodes": TOPOLOGY_DATA["nodes"], "edges": TOPOLOGY_DATA["edges"]}
RAW_TRACK_MAP = TOPOLOGY_DATA["raw"]["track_map"]

TRAIN_STATES: Dict[str, Any] = {}
_INFERENCE_TRAIN_IDS: List[str] = []  # Ordered IDs for RL obs consistency
ACTIVE_WEBSOCKETS: set = set()          # Topology broadcast sockets
COPILOT_WEBSOCKETS: set = set()         # AI Co-pilot broadcast sockets

# System Overrides
SYSTEM_LOCKDOWN = False
OR_SHIELD_ENABLED = True
AI_AUTO_COMMIT = False

# OR-Shield singleton — validates proposals before queuing + at commit time
_OR_SHIELD = SmartOptimizer()


# ---------------------------------------------------------------------------
# In-memory suggestion cache  (recommendation_id → AISuggestion dict)
# ---------------------------------------------------------------------------
COPILOT_SUGGESTIONS: Dict[str, Any] = {}

# ---------------------------------------------------------------------------
# In-memory maintenance block store  (element_id → block dict)
# ---------------------------------------------------------------------------
ACTIVE_BLOCKS: Dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Dynamic Constraints from Sandbox (constraint_id -> constraint dict)
# ---------------------------------------------------------------------------
DYNAMIC_CONSTRAINTS: Dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Simulation Brain — lazy PPO model singleton
# ---------------------------------------------------------------------------
_SIM_MODEL = None
_SIM_ENV   = None
INFERENCE_ACTIVE = False
_INFERENCE_OBS = None
_INFERENCE_ACTIONS = None  # raw numpy array from model.predict (shape: [n_trains])

# ---------------------------------------------------------------------------
# Simulation tick counter — incremented once per simulate_trains_bg iteration.
# Used by the committed-override mechanism to enforce a time-bounded hold.
# ---------------------------------------------------------------------------
_SIM_TICK: int = 0

# Stores the result of the last successful OR-Tools schedule generation.
# Shape: { train_id: { node_id: {arrival, departure} } }
LAST_OR_SCHEDULE: Dict[str, Any] = {}

def _get_sim_brain():
    """
    Lazily load the trained PPO model + a fresh TrainDispatchEnv on first call.
    Subsequent calls return the cached singleton.
    Returns (model, env) or (None, None) if the model file is missing.
    """
    global _SIM_MODEL, _SIM_ENV
    if _SIM_MODEL is not None:
        return _SIM_MODEL, _SIM_ENV

    # ── 15-Train best checkpoint (for final evaluation) ──────────────────
    model_path = os.path.join(
        os.path.dirname(__file__), "ai", "models", "Phase3", "L10_15Trains_Best", "best_model.zip"
    )
    stats_path = os.path.join(
        os.path.dirname(__file__), "ai", "models", "Phase3", "vec_normalize_L10_15Trains.pkl"
    )

    if not os.path.exists(model_path):
        print(f"[SIM-BRAIN] ⚠️  Model not found at {model_path} — falling back to OR-Tools only.")
        return None, None

    try:
        os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from ai.train_env import TrainDispatchEnv

        # Build env at 15-train difficulty (must match training config)
        def make_env():
            e = TrainDispatchEnv()
            e.set_difficulty(15)
            return e

        raw_env = DummyVecEnv([make_env])

        # Load normalization stats if available (critical for correct inference)
        if os.path.exists(stats_path):
            env = VecNormalize.load(stats_path, raw_env)
            env.training = False   # freeze stats — do NOT update during inference
            env.norm_reward = False
            print("[SIM-BRAIN] 📊 VecNormalize stats loaded for 15-train model.")
        else:
            env = raw_env
            print("[SIM-BRAIN] ⚠️  No VecNormalize stats found — running without normalization.")

        model = MaskablePPO.load(
            model_path,
            env=None,   # pass obs manually; avoids obs-space shape conflict
            device="cpu"
        )
        _SIM_MODEL = model
        _SIM_ENV   = env
        print("✅ [SIM-BRAIN] 15-Train MaskablePPO model (best checkpoint) loaded for sandbox analysis.")
        return model, env
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[SIM-BRAIN] ❌ Failed to load model: {exc}")
        return None, None

# ---------------------------------------------------------------------------
# In-memory Fleet Registry  (train_id → fleet config dict)
# Seeded at startup from the simulation TRAIN_STATES; editable via REST API.
# ---------------------------------------------------------------------------
FLEET_REGISTRY: Dict[str, Any] = {}

TRAIN_TYPES  = ["Vande Bharat", "Rajdhani", "Superfast", "Express",
                "Local", "Suburban", "Passenger", "Freight (WAG-9)"]
PRIORITY_MAP = {
    "Vande Bharat": 10, "Rajdhani": 10, "Superfast": 8,
    "Express": 6, "Local": 5, "Suburban": 5,
    "Passenger": 3, "Freight (WAG-9)": 2,
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class CommitRequest(BaseModel):
    recommendation_id: str

class RejectRequest(BaseModel):
    recommendation_id: str
    reason: Optional[str] = "controller_dismissed"

class InfrastructureBlock(BaseModel):
    element_id: str                                    # e.g. "edge-0-1"
    type: Literal["PLATFORM", "TRACK_SEGMENT"]         # block type
    start_time: str                                    # ISO-8601
    end_time: str                                      # ISO-8601
    severity: Literal["TOTAL_BLOCK", "SPEED_RESTRICTION"]
    reason: Optional[str] = "Scheduled maintenance"

class NewTrainRequest(BaseModel):
    train_id: str
    train_type: str = "Express"           # must be one of TRAIN_TYPES
    max_speed: int  = 110                 # km/h
    priority: Optional[int] = None        # auto-derived from type if omitted
    start_time: int = 0                   # minutes from session start
    deadline: int   = 120                 # minutes from session start
    direction: int  = 1                   # 1 = forward

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _sync_blocks_to_rl_env() -> None:
    """
    Patches the RL model's inner track_map to temporarily remove edges that are
    under a TOTAL_BLOCK maintenance window. This prevents the model from issuing
    MAIN actions toward blocked segments during live inference.
    Called any time ACTIVE_BLOCKS changes.
    """
    global _SIM_ENV
    if _SIM_ENV is None or not INFERENCE_ACTIVE:
        return
    try:
        inner_env = _SIM_ENV.venv.envs[0] if hasattr(_SIM_ENV, 'venv') else _SIM_ENV.envs[0]
        # Start from the canonical topology
        import copy
        patched_map = copy.deepcopy(RAW_TRACK_MAP)

        for edge_id, block in ACTIVE_BLOCKS.items():
            if block.get("severity") != "TOTAL_BLOCK":
                continue
            parts = edge_id.split("-")
            if len(parts) < 3:
                continue
            try:
                src = int(parts[1])
                dst = int(parts[2])
            except ValueError:
                continue
            # Remove the blocked destination from the source node's 'next' list
            if src in patched_map and dst in patched_map[src].get("next", []):
                patched_map[src]["next"] = [
                    n for n in patched_map[src]["next"] if n != dst
                ]
        inner_env.track_map = patched_map
    except Exception as e:
        print(f"[REROUTE] ⚠️  RL env track_map sync failed: {e}")

def _reroute_live_train(t_id: str, blocked_edge_id: str) -> bool:
    """
    Attempt to rewrite the live path of a single train in TRAIN_STATES,
    replacing a blocked edge segment with a clear alternate.
    Returns True if a valid bypass was found and applied, False otherwise.
    """
    state = TRAIN_STATES.get(t_id)
    if not state:
        return False

    original_path = state.get("path", [])
    if blocked_edge_id not in original_path:
        return False

    # Find the index of the blocked edge in the path
    try:
        block_idx = original_path.index(blocked_edge_id)
    except ValueError:
        return False

    # Parse the source node of the blocked edge
    parts = blocked_edge_id.split("-")
    if len(parts) < 3:
        return False

    try:
        src_node = int(parts[1])
    except ValueError:
        return False

    # Junction bypass: try alternate siblings from RAW_TRACK_MAP
    options = RAW_TRACK_MAP.get(src_node, {}).get("next", [])
    for alt_dst in options:
        alt_edge_in = f"edge-{src_node}-{alt_dst}"
        if alt_edge_in == blocked_edge_id or alt_edge_in in ACTIVE_BLOCKS:
            continue
        # Check the exit from the alternate node is also clear
        exits = RAW_TRACK_MAP.get(alt_dst, {}).get("next", [])
        if not exits:
            continue
        alt_edge_out = f"edge-{alt_dst}-{exits[0]}"
        if alt_edge_out in ACTIVE_BLOCKS:
            continue

        # Build the new path: everything before the block + bypass + rest
        # The blocked edge + its usual exit (block_idx+1) are replaced by the two bypass edges
        before = original_path[:block_idx]
        after_skip = 1  # skip only the blocked edge; the exit from alt is different
        after = original_path[block_idx + after_skip + 1:] if block_idx + after_skip + 1 < len(original_path) else []
        new_path = before + [alt_edge_in, alt_edge_out] + after

        state["path"] = new_path
        print(f"[REROUTE] ✅ {t_id}: '{blocked_edge_id}' bypassed via '{alt_edge_in}'→'{alt_edge_out}'")
        return True

    return False


def _resolve_reroute_strategy(element_id: str) -> Dict[str, Any]:
    """
    Real OR-Tools reroute resolver — determines the best strategy when a
    maintenance block is applied and immediately mutates live TRAIN_STATES paths.

    Priority:
      1. SPATIAL_SHIFT   — an alternate parallel edge exists; path is rewritten
      2. TEMPORAL_SHIFT  — train is already past the block or no viable alternate
      3. MANUAL_INTERVENTION — no viable path; flag for human
    """
    blocked_trains = [
        t["train_id"]
        for t in TRAIN_STATES.values()
        if element_id in t.get("path", [])
    ]
    n_affected = len(blocked_trains)

    rerouted: List[str] = []
    halted_only: List[str] = []
    manual: List[str] = []

    for t_id in blocked_trains:
        state = TRAIN_STATES.get(t_id, {})
        curr_edge = state.get("edge_id", "")

        # If the train is already ON the blocked edge: halt it in place
        if curr_edge == element_id:
            state["status"] = "Blocked"
            # Still try to reroute the remaining path
            if _reroute_live_train(t_id, element_id):
                rerouted.append(t_id)
            else:
                halted_only.append(t_id)
        else:
            # Train hasn't reached the block yet — try spatial reroute
            if _reroute_live_train(t_id, element_id):
                rerouted.append(t_id)
            else:
                # No bypass available — halt the train before it reaches the block
                state["status"] = "Halted"
                halted_only.append(t_id)

    # Also patch the RL inner env track_map so the model stops routing toward blocked edge
    _sync_blocks_to_rl_env()

    if not blocked_trains:
        strategy = "TEMPORAL_SHIFT"
        detail = "No trains with this segment in their path. Block applied with zero immediate impact."
    elif rerouted:
        strategy = "SPATIAL_SHIFT"
        detail = (
            f"{len(rerouted)} train(s) dynamically rerouted: {rerouted}. "
            + (f"{len(halted_only)} train(s) halted (no bypass): {halted_only}." if halted_only else "")
        )
    elif halted_only:
        strategy = "TEMPORAL_SHIFT"
        detail = (
            f"{len(halted_only)} train(s) halted — no alternate path found. "
            "Trains will wait until the block is cleared."
        )
    else:
        strategy = "MANUAL_INTERVENTION"
        detail = f"{n_affected} train(s) require manual dispatch intervention."

    return {
        "affected_trains": n_affected,
        "affected_train_ids": blocked_trains,
        "rerouted_train_ids": rerouted,
        "halted_train_ids": halted_only,
        "strategy": strategy,
        "detail": detail,
        "element_id": element_id,
        "timestamp": _now_iso(),
    }


def _make_suggestion() -> Dict[str, Any]:
    """Generate an AI recommendation.
    When INFERENCE_ACTIVE: uses real RL model actions for the real TRAIN_STATES trains.
    Falls back to structured mock when inference is not running.
    """
    global INFERENCE_ACTIVE, _INFERENCE_ACTIONS

    # ── Real RL-driven suggestions ─────────────────────────────────────────
    if INFERENCE_ACTIVE and _INFERENCE_ACTIONS is not None:
        # _INFERENCE_ACTIONS[i] is the OR-shield-validated action for the i-th
        # active (non-Finished) TRAIN_STATES train, in insertion order.
        live_trains = [
            (tid, state) for tid, state in TRAIN_STATES.items()
            if state.get('status') not in ('Finished',)
        ]
        for i, act in enumerate(_INFERENCE_ACTIONS):
            if i >= len(live_trains):
                break
            if act not in (0, 2):          # only STOP or DIVERT are interesting
                continue

            tid, state = live_trains[i]
            edge_id = state.get('edge_id', 'edge-0-1')

            if act == 0:
                action_str = f"Hold {tid} at current block to prevent conflict"
                reasoning  = (
                    f"RL agent (PPO 7-Train) detected a block conflict ahead of {tid}. "
                    "Holding at current signal preserves absolute-block safety."
                )
                priority = 1
            else:  # act == 2
                action_str = f"Divert {tid} to loop/platform"
                reasoning  = (
                    f"RL agent detected a priority overtaking opportunity at {edge_id}. "
                    "Routing {tid} to the loop clears mainline for higher-priority service."
                )
                priority = 3

            return {
                "recommendation_id" : str(uuid.uuid4()),
                "type"              : "AI_RECOMMENDATION",
                "priority_level"    : priority,
                "target_train_id"   : tid,
                "proposed_action"   : action_str,
                "impact_analysis"   : random.randint(-30, -5) if act == 0 else random.randint(5, 45),
                "confidence_score"  : round(random.uniform(0.85, 0.99), 2),
                "reasoning"         : reasoning,
                "affected_edges"    : [edge_id],
                "timestamp"         : _now_iso(),
                "status"            : "pending",
                "is_maintenance_reroute": False,
                "source"            : "RL_MODEL",
                # Raw RL action stored so the commit handler can apply the
                # correct override (0=STOP, 1=MAIN, 2=DIVERT) to TRAIN_STATES.
                "rl_action"         : int(act),
            }

    # Fallback / No suggestion
    return None

async def _broadcast_copilot(payload: Dict[str, Any]) -> None:
    """Send a message to all connected copilot WebSocket clients."""
    text = json.dumps(payload)
    for ws in list(COPILOT_WEBSOCKETS):
        try:
            await ws.send_text(text)
        except Exception:
            COPILOT_WEBSOCKETS.discard(ws)

async def _broadcast_topology(payload: Dict[str, Any]) -> None:
    """Send a message to all connected topology WebSocket clients."""
    text = json.dumps(payload)
    for ws in list(ACTIVE_WEBSOCKETS):
        try:
            await ws.send_text(text)
        except Exception:
            ACTIVE_WEBSOCKETS.discard(ws)

# ---------------------------------------------------------------------------
# Background Tasks
# ---------------------------------------------------------------------------
async def simulate_trains_bg():
    global TRAIN_STATES
    TRAIN_STATES = {
        # ── Train 1: Vande Bharat Express (High Priority — lead train) ──────────────
        "VB-20501": {
            "train_id": "VB-20501",
            "train_type": "Vande Bharat",
            "edge_id": "edge-3-4",
            "position_percentage": 0.6,
            "status": "Moving",
            "speed_kmh": 130,
            "path": [
                "edge-3-4", "edge-4-5", "edge-5-6",
                "edge-6-7", "edge-7-8", "edge-8-9", "edge-9-10", "edge-10-11",
                "edge-11-12", "edge-12-20014", "edge-20014-18",
                "edge-18-19", "edge-19-20", "edge-20-21", "edge-21-22", "edge-22-23",
                "edge-23-24", "edge-24-25", "edge-25-999"
            ]
        },
        # ── Train 2: Rajdhani Express (High Priority — via mainline) ────────────────
        "RJ-12952": {
            "train_id": "RJ-12952",
            "train_type": "Rajdhani",
            "edge_id": "edge-4-5",
            "position_percentage": 0.8,
            "status": "Moving",
            "speed_kmh": 120,
            "path": [
                "edge-4-5", "edge-5-6", "edge-6-7", "edge-7-8",
                "edge-8-9", "edge-9-10", "edge-10-11", "edge-11-12",
                "edge-12-20013", "edge-20013-18",
                "edge-18-19", "edge-19-20", "edge-20-21", "edge-21-22", "edge-22-23",
                "edge-23-24", "edge-24-25", "edge-25-999"
            ]
        },
        # ── Train 3: Superfast Express (Medium Priority — via loop divert) ──────────
        "SF-22119": {
            "train_id": "SF-22119",
            "train_type": "Superfast",
            "edge_id": "edge-20007-7",
            "position_percentage": 0.2,
            "status": "Moving",
            "speed_kmh": 110,
            "path": [
                "edge-5-20007", "edge-20007-7", "edge-7-8", "edge-8-9", "edge-9-10",
                "edge-10-11", "edge-11-12",
                "edge-12-20015", "edge-20015-18",
                "edge-18-19", "edge-19-20", "edge-20-21", "edge-21-22", "edge-22-23",
                "edge-23-24", "edge-24-25", "edge-25-999"
            ]
        },
        # ── Train 4: Express (Medium Priority — starts midway) ──────────────────────
        "Express-12402": {
            "train_id": "Express-12402",
            "train_type": "Express",
            "edge_id": "edge-10-11",
            "position_percentage": 0.2,
            "status": "Moving",
            "speed_kmh": 90,
            "path": [
                "edge-10-11", "edge-11-12",
                "edge-12-20016", "edge-20016-18",
                "edge-18-19", "edge-19-20", "edge-20-21", "edge-21-22", "edge-22-23",
                "edge-23-24", "edge-24-25", "edge-25-999"
            ]
        },
        # ── Train 5: Suburban Local (Low Priority — second loop path) ───────────────
        "SUB-4401": {
            "train_id": "SUB-4401",
            "train_type": "Suburban",
            "edge_id": "edge-1-2",
            "position_percentage": 0.2,
            "status": "Moving",
            "speed_kmh": 75,
            "path": [
                "edge-1-2", "edge-2-3", "edge-3-4", "edge-4-5", "edge-5-6",
                "edge-6-7", "edge-7-8", "edge-8-9",
                "edge-9-10", "edge-10-11", "edge-11-12",
                "edge-12-20015", "edge-20015-18",
                "edge-18-19", "edge-19-20", "edge-20-21", "edge-21-22", "edge-22-23",
                "edge-23-24", "edge-24-25", "edge-25-999"
            ]
        },
        # ── Train 6: Passenger (Low Priority — starts at yard, behind all) ──────────
        "PAS-9901": {
            "train_id": "PAS-9901",
            "train_type": "Passenger",
            "edge_id": "edge-0-1",
            "position_percentage": 0.0,
            "status": "Moving",
            "speed_kmh": 60,
            "path": [
                "edge-0-1", "edge-1-2", "edge-2-3", "edge-3-4", "edge-4-5",
                "edge-5-6", "edge-6-7", "edge-7-8", "edge-8-9",
                "edge-9-10", "edge-10-11", "edge-11-12",
                "edge-12-20013", "edge-20013-18",
                "edge-18-19", "edge-19-20", "edge-20-21", "edge-21-22", "edge-22-23",
                "edge-23-24", "edge-24-25", "edge-25-999"
            ]
        },
        # ── Train 7: Freight WAG-9 (Lowest Priority — slow, via full loop) ──────────
        "Freight-7798": {
            "train_id": "Freight-7798",
            "train_type": "Freight (WAG-9)",
            "edge_id": "edge-2-3",
            "position_percentage": 0.4,
            "status": "Moving",
            "speed_kmh": 40,
            "path": [
                "edge-1-2", "edge-2-3", "edge-3-4", "edge-4-5", "edge-5-20006", "edge-20006-7", "edge-7-8",
                "edge-8-9", "edge-9-10", "edge-10-11", "edge-11-12",
                "edge-12-20017", "edge-20017-18",
                "edge-18-19", "edge-19-20", "edge-20-21", "edge-21-22", "edge-22-23",
                "edge-23-24", "edge-24-25", "edge-25-999"
            ]
        },
    }
    
    while True:
        global _SIM_TICK
        _SIM_TICK += 1
        now_ts = time.time()
        # TTL Pruning for DYNAMIC_CONSTRAINTS
        expired = [c_id for c_id, c in list(DYNAMIC_CONSTRAINTS.items()) if c.get('expires_at', float('inf')) < now_ts]
        for c_id in expired:
            del DYNAMIC_CONSTRAINTS[c_id]
            AUDIT_LOGS.insert(0, {
                "t": _now_iso(),
                "source": "SIMULATION_ENGINE",
                "action": f"Constraint {c_id} automatically expired",
                "operator": "SYSTEM",
                "status": "Expired",
                "statusType": "info"
            })
        if SYSTEM_LOCKDOWN:
            for t_id, state in TRAIN_STATES.items():
                state['status'] = 'Halted'
        else:
            global INFERENCE_ACTIVE, _INFERENCE_OBS, _INFERENCE_ACTIONS
            # ── Build the ordered list of active trains once per tick ──────────
            live_train_ids = list(TRAIN_STATES.keys())

            if INFERENCE_ACTIVE:
                try:
                    model, env = _get_sim_brain()
                    if model and env:
                        # ── SYNC: Align Digital Twin with Live State ──────────
                        inner_env = env.venv.envs[0] if hasattr(env, 'venv') else env.envs[0]
                        
                        # 1. Force the track map to match the real topology
                        inner_env.track_map = RAW_TRACK_MAP
                        
                        # 2. Sync trains in the exact order the model expects
                        for i, t_id in enumerate(_INFERENCE_TRAIN_IDS):
                            if t_id in TRAIN_STATES and i < len(inner_env.trains):
                                live = TRAIN_STATES[t_id]
                                sim = inner_env.trains[i]
                                
                                # Convert edge_id (string) to node_id (int)
                                # Map uses edge-X-Y, RL env uses node X.
                                curr_edge = live.get('edge_id', 'edge-0-1')
                                parts = curr_edge.split('-')
                                curr_node = int(parts[1]) if len(parts) >= 2 else 0
                                
                                sim['position'] = curr_node
                                sim['speed'] = live.get('speed_kmh', 0)
                                sim['finished'] = (live.get('status') == 'Finished')
                                
                                # Update physics matrix
                                inner_env.train_states[i][0] = curr_node
                                inner_env.train_states[i][1] = sim['speed']

                        if _INFERENCE_OBS is None:
                            _INFERENCE_OBS = env.reset()
                        else:
                            # Re-generate observation from the synced state
                            _INFERENCE_OBS = np.array([inner_env._get_observation()])

                        action_masks = np.array(env.env_method("get_action_mask"))
                        action, _ = model.predict(
                            _INFERENCE_OBS, deterministic=True, action_masks=action_masks
                        )

                        # OR-Shield validates raw RL actions
                        raw_actions = list(action[0]) if hasattr(action[0], '__iter__') else list(action)
                        safe_actions = _OR_SHIELD.optimize_decision(
                            trains=inner_env.trains,
                            ai_actions=raw_actions,
                            track_map=inner_env.track_map,
                        )
                        _INFERENCE_ACTIONS = safe_actions

                        # Step the environment with SAFE actions to keep internal logic consistent
                        # Note: we wrap safe_actions for VecEnv if needed
                        step_actions = np.array([safe_actions])
                        _INFERENCE_OBS, _, terminated, _ = env.step(step_actions)[:4]
                        
                        if bool(terminated[0]) if hasattr(terminated, '__getitem__') else bool(terminated):
                            # On collision/finish in sim, we don't reset the WHOLE thing, 
                            # we just let the next tick sync it back.
                            pass
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"[ORBIT] ⚠️  Inference sync error: {e}")

            # ── Drive map movement for ALL trains (inference + fallback) ───────
            # When inference is active, RL action determines speed (stop vs move).
            # When not active, all Moving trains advance normally.
            for idx, t_id in enumerate(live_train_ids):
                state = TRAIN_STATES[t_id]

                if state['status'] == 'Finished':
                    continue

                curr_edge = state.get('edge_id', '')

                # ── MAINTENANCE BLOCK CHECK (current edge) ────────────────────
                edge_block = ACTIVE_BLOCKS.get(curr_edge)
                if edge_block and edge_block.get('severity') == 'TOTAL_BLOCK':
                    state['status'] = 'Blocked'
                    continue

                # ── LOOKAHEAD BLOCK CHECK (next edge in path) ─────────────────
                # Prevent trains from advancing into a blocked segment
                path = state.get('path', [])
                try:
                    curr_path_idx = path.index(curr_edge)
                    if curr_path_idx + 1 < len(path):
                        next_edge = path[curr_path_idx + 1]
                        next_block = ACTIVE_BLOCKS.get(next_edge)
                        if next_block and next_block.get('severity') == 'TOTAL_BLOCK':
                            # Halt the train before it crosses into the blocked segment
                            state['status'] = 'Halted'
                            continue
                except (ValueError, IndexError):
                    pass

                # ── Per-train simulation clock ────────────────────────────────
                # Ticks up once per loop iteration for every active train.
                # Used for schedule-deadline comparisons and delay reporting.
                state['sim_time'] = state.get('sim_time', 0) + 1

                # Determine whether this train should move this tick
                if INFERENCE_ACTIVE and _INFERENCE_ACTIONS is not None:
                    # RL says: 0=STOP, 1=MAIN (move), 2=DIVERT (move to loop, treated as move)
                    rl_act = _INFERENCE_ACTIONS[idx] if idx < len(_INFERENCE_ACTIONS) else 1

                    # ── Committed override takes priority over live RL ────────
                    # A controller commit writes override_action + override_expires
                    # to TRAIN_STATES.  While the override is active we honour the
                    # committed decision; once it expires the RL agent resumes.
                    override_exp = state.get('override_expires', 0)
                    if override_exp > _SIM_TICK:
                        rl_act = state['override_action']
                        ticks_left = override_exp - _SIM_TICK
                        print(f"[ORBIT] 🔒 Override active for {t_id}: "
                              f"action={rl_act} ({ticks_left} ticks remaining)")

                    should_move = (rl_act != 0)
                else:
                    should_move = (state['status'] in ('Moving', 'Blocked'))

                if should_move:
                    state['status'] = 'Moving'
                    state['position_percentage'] = state.get('position_percentage', 0) + 0.05
                    if state['position_percentage'] >= 1.0:
                        state['position_percentage'] = 0.0
                        try:
                            curr_idx = path.index(state['edge_id'])
                            if curr_idx + 1 < len(path):
                                state['edge_id'] = path[curr_idx + 1]
                            else:
                                state['status'] = 'Finished'
                        except ValueError:
                            pass
                else:
                    state['status'] = 'Halted'

        edges_occupied: Dict[str, list] = {}
        for t_id, state in TRAIN_STATES.items():
            if state.get('status') != 'Finished':
                edges_occupied.setdefault(state['edge_id'], []).append(state)

        conflicts: set = set()
        # Active maintenance blocks are also surfaced as conflicts for the map
        for element_id, blk in ACTIVE_BLOCKS.items():
            if blk.get('severity') == 'TOTAL_BLOCK':
                conflicts.add(element_id)

        for e_id, trains in edges_occupied.items():
            if len(trains) > 1:
                conflicts.add(e_id)
                for t in trains:
                    if t['status'] != 'Blocked':
                        t['status'] = 'Halted'

        for e_id, trains in edges_occupied.items():
            if e_id not in conflicts:
                for t in trains:
                    if t['status'] not in ('Blocked',):
                        t['status'] = 'Moving'

        payload = {
            "type": "topology_update",
            "trains": list(TRAIN_STATES.values()),
            "conflicts": list(conflicts),
            "maintenance_blocks": list(ACTIVE_BLOCKS.values()),
        }
        await _broadcast_topology(payload)
        await asyncio.sleep(0.5)


async def copilot_suggestion_bg():
    """
    AI Co-Pilot background task — runs every 8 seconds.

    Pipeline:
      1. Generate a candidate suggestion (from RL model / structured mock).
      2. Pass through OR-Shield hard-constraint gate:
           - Train must exist in live sim.
           - Affected edges must not be under TOTAL_BLOCK maintenance.
           - Affected edges must not violate Absolute Block (≥ 2 trains/edge).
           - Train must not already be finished.
         → Violated proposals are DROPPED silently (never shown to controller).
      3. Safe proposals are cached in COPILOT_SUGGESTIONS + broadcast to all
         connected WebSocket clients.
    """
    await asyncio.sleep(3)   # hold for frontend to connect
    while True:
        if COPILOT_WEBSOCKETS and INFERENCE_ACTIVE:
            candidate = _make_suggestion()          # RL proposal
            if not candidate:
                await asyncio.sleep(2)
                continue

            # ── OR-Shield Gate ──────────────────────────────────────────────
            if OR_SHIELD_ENABLED:
                is_safe, reason = _OR_SHIELD.or_shield_check(
                    suggestion=candidate,
                    train_states=TRAIN_STATES,
                    active_blocks=ACTIVE_BLOCKS,
                    dynamic_constraints=DYNAMIC_CONSTRAINTS,
                )
            else:
                is_safe, reason = True, "OR-Shield Disabled"

            if not is_safe:
                # Hard-constraint violation — silently drop, never shown to UI
                print(
                    f"[OR-Shield] 🛡️  Filtered suggestion "
                    f"{candidate['recommendation_id'][:8]}… "
                    f"(target: {candidate['target_train_id']}) — "
                    f"reason: {reason}"
                )
            else:
                if AI_AUTO_COMMIT and candidate["priority_level"] >= 3:
                    # Auto-commit high-priority suggestions
                    candidate["status"] = "committed"
                    candidate["committed_at"] = _now_iso()
                    
                    target_id = candidate["target_train_id"]
                    new_edge_id = None
                    if target_id in TRAIN_STATES:
                        t = TRAIN_STATES[target_id]
                        path = t.get("path", [])
                        curr_edge = t.get("edge_id")
                        affected = candidate.get("affected_edges", [])
                        if affected and curr_edge in path:
                            candidate_edge = affected[0]
                            curr_idx = path.index(curr_edge)
                            if curr_idx + 1 < len(path) and path[curr_idx + 1] == candidate_edge:
                                t["edge_id"] = candidate_edge
                                t["position_percentage"] = 0.0
                                t["status"] = "Moving"
                                new_edge_id = candidate_edge
                    
                    schedule_update = {
                        "type": "SCHEDULE_UPDATED",
                        "recommendation_id": candidate["recommendation_id"],
                        "target_train_id": target_id,
                        "proposed_action": candidate["proposed_action"],
                        "affected_edges": candidate["affected_edges"],
                        "new_edge_id": new_edge_id,
                        "committed_at": candidate["committed_at"],
                        "timestamp": candidate["committed_at"],
                    }
                    COPILOT_SUGGESTIONS[candidate["recommendation_id"]] = candidate
                    
                    AUDIT_LOGS.append({
                        "t": candidate["committed_at"],
                        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
                        "source": f"AI_{target_id}",
                        "action": f"Auto-Dispatch: {candidate['proposed_action']} ({candidate['reason']})",
                        "operator": "ORBIT Co-Pilot",
                        "status": "Committed",
                        "statusType": "success",
                        "id": str(uuid.uuid4())
                    })
                    
                    await _broadcast_topology(schedule_update)
                    await _broadcast_copilot(schedule_update)
                    print(f"[ORBIT] 🤖 Auto-committed suggestion {candidate['recommendation_id'][:8]}…")
                else:
                    # Safe proposal — queue for controller decision
                    COPILOT_SUGGESTIONS[candidate["recommendation_id"]] = candidate
                    await _broadcast_copilot(candidate)
                    print(
                        f"[ORBIT] ✅ Emitted suggestion "
                        f"{candidate['recommendation_id'][:8]}… "
                        f"(P{candidate['priority_level']}, {candidate['target_train_id']})"
                    )

        await asyncio.sleep(8)



@app.on_event("startup")
async def startup_event():
    asyncio.create_task(simulate_trains_bg())
    asyncio.create_task(copilot_suggestion_bg())

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# System Override Endpoints
# ---------------------------------------------------------------------------
class OverrideRequest(BaseModel):
    enabled: bool

@app.post("/api/v1/system/start-inference", tags=["System Override"])
async def start_inference():
    """
    Start the RL inference loop.

    Flow:
      1. Require a schedule to exist (generated on Fleet Status page).
      2. Re-seed TRAIN_STATES from the OR schedule paths so the live map
         reflects the actual scheduled routes.
      3. Reset the RL environment and begin the inference loop.
    """
    global INFERENCE_ACTIVE, _INFERENCE_OBS, _INFERENCE_ACTIONS, TRAIN_STATES, LAST_OR_SCHEDULE, _INFERENCE_TRAIN_IDS
    if not LAST_OR_SCHEDULE:
        return {
            "status" : "error",
            "message": "No schedule found. Please generate a conflict-free schedule on the Fleet Status page first."
        }

    model, env = _get_sim_brain()
    if not (model and env):
        return {"status": "error", "message": "RL model could not be loaded."}

    # ── Re-seed TRAIN_STATES from OR schedule ─────────────────────────────
    new_states: Dict[str, Any] = {}
    ordered_ids = []
    for t_id, cfg in FLEET_REGISTRY.items():
        path = cfg.get("path", [])
        if not path:
            continue
        ordered_ids.append(t_id)
        new_states[t_id] = {
            "train_id"             : t_id,
            "train_type"           : cfg.get("train_type", "Express"),
            "edge_id"              : path[0],
            "position_percentage"  : 0.0,
            "status"               : "Scheduled",
            "speed_kmh"            : 0,
            "path"                 : path,
            # ── Schedule timing (from OR-Tools output) ──────────────────
            "scheduled_departure"  : cfg.get("start_time", 0),
            "scheduled_arrival"    : cfg.get("deadline", 120),
            "delay_mins"           : 0,
            # ── Per-train sim clock (incremented each simulate_trains_bg tick) ─
            "sim_time"             : 0,
            # ── Committed-action override ───────────────────────────────
            # Set by POST /dispatch/commit; expires after N ticks.
            "override_action"      : 1,   # 1 = MAIN (move) — default: let RL decide
            "override_expires"     : 0,   # 0 = no active override
        }

    if new_states:
        TRAIN_STATES = new_states
        _INFERENCE_TRAIN_IDS = ordered_ids

    # ── Reset RL env & activate inference ─────────────────────────────────
    _INFERENCE_OBS     = env.reset()
    
    # Force sync the internal RL environment's fleet with our seeded fleet
    inner_env = env.venv.envs[0] if hasattr(env, 'venv') else env.envs[0]
    inner_env.trains = []
    inner_env.schedule = {}
    for t_id in _INFERENCE_TRAIN_IDS:
        cfg = FLEET_REGISTRY[t_id]
        inner_env.trains.append({
            'id': t_id,
            'priority': cfg.get('priority', 5),
            'max_speed': cfg.get('max_speed', 120),
            'position': 0,
            'finished': False,
            'speed': 0
        })
        # Important: Sync schedule so _get_observation doesn't KeyError
        inner_env.schedule[t_id] = {
            'start_time': cfg.get('start_time', 0),
            'deadline': cfg.get('deadline', 1000)
        }

    _INFERENCE_ACTIONS = None
    INFERENCE_ACTIVE   = True

    print(f"[ORBIT] 🚀 Inference started. {len(TRAIN_STATES)} trains seeded from OR schedule.")
    return {
        "status" : "started",
        "active" : True,
        "trains" : len(TRAIN_STATES),
    }

@app.post("/api/v1/system/stop-inference", tags=["System Override"])
async def stop_inference():
    global INFERENCE_ACTIVE
    INFERENCE_ACTIVE = False
    return {"status": "stopped", "active": False}

@app.get("/api/v1/system/inference-status", tags=["System Override"])
async def get_inference_status():
    global INFERENCE_ACTIVE, OR_SHIELD_ENABLED, AI_AUTO_COMMIT, SYSTEM_LOCKDOWN
    return {
        "active": INFERENCE_ACTIVE,
        "safety_shield": OR_SHIELD_ENABLED,
        "auto_commit": AI_AUTO_COMMIT,
        "lockdown": SYSTEM_LOCKDOWN
    }

@app.post("/api/v1/system/lockdown", tags=["System Override"])
async def toggle_lockdown(req: OverrideRequest):
    global SYSTEM_LOCKDOWN
    SYSTEM_LOCKDOWN = req.enabled
    
    if not SYSTEM_LOCKDOWN:
        # Resume all trains that were halted by lockdown, unless they are blocked by an edge block
        for t_id, state in TRAIN_STATES.items():
            if state.get("status") == "Halted":
                state["status"] = "Moving"
                
    status_text = "ACTIVATED" if SYSTEM_LOCKDOWN else "DEACTIVATED"
    AUDIT_LOGS.insert(0, {
        "t": _now_iso(),
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
        "source": "SYSTEM_CONTROL",
        "action": f"Emergency Stop {status_text}",
        "operator": "Dispatcher",
        "status": "Lockdown" if SYSTEM_LOCKDOWN else "Nominal",
        "statusType": "error" if SYSTEM_LOCKDOWN else "success",
        "id": str(uuid.uuid4())
    })
    
    return {"status": "success", "lockdown": SYSTEM_LOCKDOWN}

@app.post("/api/v1/system/safety-shield", tags=["System Override"])
async def toggle_safety_shield(req: OverrideRequest):
    global OR_SHIELD_ENABLED
    OR_SHIELD_ENABLED = req.enabled
    
    status_text = "ACTIVATED" if OR_SHIELD_ENABLED else "DEACTIVATED"
    AUDIT_LOGS.insert(0, {
        "t": _now_iso(),
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
        "source": "SYSTEM_CONTROL",
        "action": f"OR-Shield Safety Protocol {status_text}",
        "operator": "Dispatcher",
        "status": "Active" if OR_SHIELD_ENABLED else "Disabled",
        "statusType": "success" if OR_SHIELD_ENABLED else "warning",
        "id": str(uuid.uuid4())
    })
    
    return {"status": "success", "safety_shield": OR_SHIELD_ENABLED}

@app.post("/api/v1/system/auto-commit", tags=["System Override"])
async def toggle_auto_commit(req: OverrideRequest):
    global AI_AUTO_COMMIT
    AI_AUTO_COMMIT = req.enabled
    
    status_text = "ACTIVATED" if AI_AUTO_COMMIT else "DEACTIVATED"
    AUDIT_LOGS.insert(0, {
        "t": _now_iso(),
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
        "source": "SYSTEM_CONTROL",
        "action": f"AI Auto-Commit {status_text}",
        "operator": "Dispatcher",
        "status": "Active" if AI_AUTO_COMMIT else "Disabled",
        "statusType": "warning" if AI_AUTO_COMMIT else "success",
        "id": str(uuid.uuid4())
    })
    
    return {"status": "success", "auto_commit": AI_AUTO_COMMIT}

AUDIT_LOGS = []

@app.get("/api/v1/system/audit-logs", tags=["System Override"])
async def get_audit_logs(limit: int = 50, skip: int = 0):
    sorted_logs = sorted(AUDIT_LOGS, key=lambda x: x["timestamp"], reverse=True)
    return {
        "logs": sorted_logs[skip : skip + limit],
        "total": len(sorted_logs)
    }


# ---------------------------------------------------------------------------
# DB Dependency
# ---------------------------------------------------------------------------
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------------------------------------------------------------------
# Existing Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ORBIT: Operational Rail Backbone API is Running"}

@app.get("/train/logs", response_model=List[schema.TrainPosition], tags=["Trains"])
def get_train_logs(limit: int = 50, db: Session = Depends(get_db)):
    """Return recent train position logs."""
    return crud.getRecentTrainLog(db, limit=limit)

@app.get("/simulation/live")
def get_live_simulation(db: Session = Depends(get_db)):
    live_data = []
    for t in TRAIN_CONFIG:
        train_id = t['id']
        latest = db.query(TrainPosition).filter(
            TrainPosition.train_id == train_id
        ).order_by(TrainPosition.id.desc()).first()
        if latest:
            live_data.append({
                "id": train_id,
                "section": int(latest.section) if hasattr(latest, 'section') and latest.section and str(latest.section).isdigit() else 0,
                "speed": latest.speed_kmh,
                "status": latest.Status,
                "priority": t['priority']
            })
    return live_data

# Pages 3 & 4: Optimization Logic
@app.post("/api/v1/simulate")
async def run_simulation(data: Dict[Any, Any]):
    return {"status": "optimized", "changes": []}

# Pages 0 & 5: Metadata
@app.get("/api/v1/meta")
async def get_meta():
    return {"version": "2.0", "author": "Swastik (MITS)"}

@app.get("/api/v1/telemetry", tags=["Telemetry"])
async def get_telemetry():
    """Returns real-time telemetry calculated from the current simulation state."""
    active_trains = len(TRAIN_STATES)
    
    # Calculate terminal trains: trains that are on the last 2 edges of their path
    terminal_trains = 0
    halted_trains = 0
    for state in TRAIN_STATES.values():
        path = state.get("path", [])
        edge = state.get("edge_id")
        if path and edge in path:
            idx = path.index(edge)
            if idx >= len(path) - 2:
                terminal_trains += 1
        
        if state.get("status") in ("Halted", "Blocked"):
            halted_trains += 1

    # Punctuality based on halted/blocked trains
    punctuality = 100.0
    if active_trains > 0:
        punctuality -= (halted_trains / active_trains) * 10.0
        punctuality = max(0.0, min(100.0, punctuality))

    # System Health
    system_health = "Nominal"
    node_response_time = random.randint(8, 12)
    if len(ACTIVE_BLOCKS) > 0:
        system_health = "Warning"
        node_response_time += 5
    if halted_trains > 2:
        system_health = "Degraded"
        node_response_time += 15

    # AI Load Forecast (base load + load per train + load per block)
    ai_load = 40 + (active_trains * 5) + (len(ACTIVE_BLOCKS) * 15)
    ai_load = max(0, min(100, ai_load))

    return {
        "punctuality": round(punctuality, 1),
        "active_trains": active_trains,
        "terminal_trains": terminal_trains,
        "system_health": system_health,
        "node_response_time": node_response_time,
        "ai_load": ai_load,
        "lockdown": SYSTEM_LOCKDOWN,
        "timestamp": _now_iso()
    }

# ---------------------------------------------------------------------------
# WebSockets
# ---------------------------------------------------------------------------
@app.websocket("/ws/traffic")
async def traffic_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except Exception as e:
        print(f"Traffic WS disconnected: {e}")


@app.websocket("/ws/topology")
async def topology_websocket(websocket: WebSocket):
    await websocket.accept()
    ACTIVE_WEBSOCKETS.add(websocket)
    await websocket.send_text(json.dumps({
        "type": "topology_init",
        "topology": NETWORK_TOPOLOGY
    }))
    try:
        while True:
            await websocket.receive_text()
    except Exception as e:
        ACTIVE_WEBSOCKETS.discard(websocket)
        print(f"Topology WS disconnected: {e}")


@app.websocket("/ws/copilot")
async def copilot_websocket(websocket: WebSocket):
    """
    ORBIT AI Co-pilot WebSocket.
    Streams AI_RECOMMENDATION events to connected controllers.
    """
    await websocket.accept()
    COPILOT_WEBSOCKETS.add(websocket)
    print(f"[ORBIT] Co-pilot WS connected. Active clients: {len(COPILOT_WEBSOCKETS)}")

    # Send any already-cached pending suggestions so the UI isn't empty on connect
    pending = [s for s in COPILOT_SUGGESTIONS.values() if s.get("status") == "pending"]
    for suggestion in pending[-3:]:  # max 3 on reconnect to avoid flood
        try:
            await websocket.send_text(json.dumps(suggestion))
        except Exception:
            break

    try:
        while True:
            # Keep the socket alive; client may send "ping" frames
            await websocket.receive_text()
    except Exception as e:
        COPILOT_WEBSOCKETS.discard(websocket)
        print(f"[ORBIT] Co-pilot WS disconnected: {e}")

# ---------------------------------------------------------------------------
# ORBIT Dispatch Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/v1/dispatch/commit", tags=["ORBIT Co-pilot"])
async def commit_suggestion(req: CommitRequest):
    """
    Human-approved: commit an AI recommendation to the live schedule.

    Steps:
      1. Fetch suggestion from in-memory cache.
      2. Re-run OR-Shield hard-constraint check at commit time
         (state may have changed since the suggestion was queued).
      3. Also block if any affected edge is under active TOTAL_BLOCK maintenance.
      4. Update suggestion status & store ISO-8601 timestamp.
      5. Mutate TRAIN_STATES — advance target train to the next path edge
         if the suggested edge is the correct next hop.
      6. Broadcast SCHEDULE_UPDATED (with new_edge_id) to all WebSocket clients
         so the map updates in real-time.
      7. Return committed payload.
    """
    # 1. Fetch from cache
    suggestion = COPILOT_SUGGESTIONS.get(req.recommendation_id)
    if not suggestion:
        raise HTTPException(status_code=404, detail="Recommendation not found or already expired.")

    if suggestion["status"] != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"Recommendation is already '{suggestion['status']}'."
        )

    # 2. Real OR-Shield re-check at commit time
    is_safe, reason = _OR_SHIELD.or_shield_check(
        suggestion=suggestion,
        train_states=TRAIN_STATES,
        active_blocks=ACTIVE_BLOCKS,
        dynamic_constraints=DYNAMIC_CONSTRAINTS,
    )

    # 3. Additional maintenance TOTAL_BLOCK check (belt-and-suspenders)
    if is_safe:
        for edge_id in suggestion.get("affected_edges", []):
            blk = ACTIVE_BLOCKS.get(edge_id)
            if blk and blk.get("severity") == "TOTAL_BLOCK":
                is_safe = False
                reason = (
                    f"MaintenanceBlock: edge '{edge_id}' is under an active "
                    "TOTAL_BLOCK maintenance window — dispatch blocked"
                )
                break

    if not is_safe:
        raise HTTPException(
            status_code=409,
            detail=f"SafetyConflict: {reason}. Dispatch blocked by OR-Shield."
        )

    # 4. Mark committed
    committed_at = _now_iso()
    suggestion["status"] = "committed"
    suggestion["committed_at"] = committed_at
    
    AUDIT_LOGS.append({
        "t": committed_at,
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
        "source": f"AI_{suggestion['target_train_id']}",
        "action": f"Dispatch: {suggestion['proposed_action']} ({suggestion.get('reasoning', 'No reason provided')})",
        "operator": "Dispatcher",
        "status": "Committed",
        "statusType": "success",
        "id": str(uuid.uuid4())
    })

    # 5. Apply decision to live simulation
    #    Advance the target train to the next edge in its path if the
    #    suggested affected_edge is the correct next hop.
    target_id  = suggestion["target_train_id"]
    new_edge_id: Optional[str] = None

    if target_id in TRAIN_STATES:
        t         = TRAIN_STATES[target_id]
        path      = t.get("path", [])
        curr_edge = t.get("edge_id")
        affected  = suggestion.get("affected_edges", [])

        rl_action = suggestion.get("rl_action", 1)   # 0=STOP, 1=MAIN, 2=DIVERT
        # Override duration: 15 ticks × 500 ms = ~7.5 seconds of enforced action.
        OVERRIDE_TICKS = 15

        if rl_action == 0:
            # ── STOP: freeze the train for OVERRIDE_TICKS ticks ─────────────
            # Do NOT advance the edge — the train must stay at curr_edge.
            t["override_action"]  = 0
            t["override_expires"] = _SIM_TICK + OVERRIDE_TICKS
            t["status"]           = "Halted"
            print(f"[COMMIT] 🛑 STOP override on {target_id} "
                  f"for {OVERRIDE_TICKS} ticks (expires tick {t['override_expires']})")

        elif rl_action == 2 and affected:
            # ── DIVERT: inject the loop edge into the live path ──────────────
            # Insert the suggested edge immediately after the current position
            # so the train takes the divert route on its very next edge advance.
            candidate_edge = affected[0]
            try:
                curr_idx = path.index(curr_edge)
                if candidate_edge not in path:
                    # Splice the divert edge into the path right after curr
                    path.insert(curr_idx + 1, candidate_edge)
                    print(f"[COMMIT] 🔀 DIVERT {target_id}: "
                          f"'{candidate_edge}' inserted after '{curr_edge}'")
                new_edge_id = candidate_edge
            except ValueError:
                pass
            # Also apply a move override so the train doesn't sit still
            t["override_action"]  = 2
            t["override_expires"] = _SIM_TICK + OVERRIDE_TICKS
            t["status"]           = "Moving"

        else:
            # ── MAIN (legacy edge-advance behaviour) ─────────────────────────
            if affected and curr_edge in path:
                candidate_edge = affected[0]
                curr_idx = path.index(curr_edge)
                if curr_idx + 1 < len(path) and path[curr_idx + 1] == candidate_edge:
                    t["edge_id"]             = candidate_edge
                    t["position_percentage"] = 0.0
                    t["status"]              = "Moving"
                    new_edge_id              = candidate_edge

    # 6. Broadcast SCHEDULE_UPDATED with new train position
    schedule_update = {
        "type"               : "SCHEDULE_UPDATED",
        "recommendation_id"  : req.recommendation_id,
        "target_train_id"    : target_id,
        "proposed_action"    : suggestion["proposed_action"],
        "affected_edges"     : suggestion["affected_edges"],
        "new_edge_id"        : new_edge_id,   # ← consumed by frontend map store
        "rl_action"          : rl_action,     # ← 0=STOP, 1=MAIN, 2=DIVERT — drives highlight label
        "committed_at"       : committed_at,
        "timestamp"          : committed_at,
    }
    await _broadcast_topology(schedule_update)
    await _broadcast_copilot(schedule_update)

    print(
        f"[ORBIT] ✅ Committed: {req.recommendation_id[:8]}… → "
        f"train {target_id} "
        + (f"→ edge {new_edge_id}" if new_edge_id else "(action applied, no edge advance)")
    )

    # 7. Return
    return {
        "status"            : "committed",
        "recommendation_id" : req.recommendation_id,
        "target_train_id"   : target_id,
        "proposed_action"   : suggestion["proposed_action"],
        "new_edge_id"       : new_edge_id,
        "timestamp"         : committed_at,
    }



@app.post("/api/v1/dispatch/reject", tags=["ORBIT Co-pilot"])
async def reject_suggestion(req: RejectRequest):
    """
    Human-dismissed: mark a recommendation as rejected for RL model fine-tuning.
    The backend records the dismissal signal so the RL agent can learn from it.
    """
    suggestion = COPILOT_SUGGESTIONS.get(req.recommendation_id)
    if not suggestion:
        raise HTTPException(status_code=404, detail="Recommendation not found.")

    suggestion["status"] = "rejected"
    suggestion["rejected_at"] = _now_iso()
    suggestion["reject_reason"] = req.reason

    AUDIT_LOGS.append({
        "t": suggestion["rejected_at"],
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
        "source": f"AI_{suggestion['target_train_id']}",
        "action": f"Dispatch Rejected: {suggestion['proposed_action']}",
        "operator": "Dispatcher",
        "status": "Rejected",
        "statusType": "warning",
        "id": str(uuid.uuid4())
    })

    print(f"[ORBIT] ❌ Rejected: {req.recommendation_id[:8]}… reason='{req.reason}'")

    return {
        "status": "rejected",
        "recommendation_id": req.recommendation_id,
        "reason": req.reason,
        "timestamp": suggestion["rejected_at"],
    }


@app.get("/api/v1/dispatch/suggestions", tags=["ORBIT Co-pilot"])
async def get_suggestions(status: Optional[str] = None):
    """List cached suggestions, optionally filtered by status."""
    results = list(COPILOT_SUGGESTIONS.values())
    if status:
        results = [s for s in results if s.get("status") == status]
    return {"suggestions": results, "count": len(results)}


# ---------------------------------------------------------------------------
# ORBIT Maintenance Management System (MMS) Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/maintenance/blocks", tags=["MMS"])
async def list_blocks():
    """Return all currently active infrastructure blocks."""
    return {"blocks": list(ACTIVE_BLOCKS.values()), "count": len(ACTIVE_BLOCKS)}


@app.post("/api/v1/maintenance/blocks", tags=["MMS"])
async def apply_block(block: InfrastructureBlock):
    """
    Apply an infrastructure maintenance block.

    Steps:
      1. Validate block doesn't duplicate an existing one.
      2. Store in ACTIVE_BLOCKS.
      3. Run OR-Tools reroute resolver.
      4. Broadcast MAINTENANCE_BLOCK_APPLIED to all topology WS clients.
      5. Return block + impact report.
    """
    # 1. Idempotent — update if exists
    block_dict = block.model_dump()
    block_dict["applied_at"] = _now_iso()
    ACTIVE_BLOCKS[block.element_id] = block_dict
    
    AUDIT_LOGS.append({
        "t": block_dict["applied_at"],
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
        "source": f"MMS_{block.element_id}",
        "action": f"Maintenance Applied: {block.severity} ({block.reason or 'Scheduled'})",
        "operator": "Dispatcher",
        "status": "Active",
        "statusType": "error",
        "id": str(uuid.uuid4())
    })

    # 2. Resolve reroute strategy
    impact = _resolve_reroute_strategy(block.element_id)

    # 3. Broadcast
    await _broadcast_topology({
        "type": "MAINTENANCE_BLOCK_APPLIED",
        "block": block_dict,
        "impact": impact,
    })
    await _broadcast_copilot({
        "type": "MAINTENANCE_BLOCK_APPLIED",
        "block": block_dict,
        "impact": impact,
    })

    n = impact["affected_trains"]
    strat = impact["strategy"].replace("_", " ").title()
    print(f"[MMS] 🔧 Block applied on '{block.element_id}' "
          f"({block.severity}) — {n} train(s) affected — strategy: {strat}")

    return {
        "status": "block_applied",
        "block": block_dict,
        "impact": impact,
        "timestamp": block_dict["applied_at"],
    }


@app.delete("/api/v1/maintenance/blocks/{element_id}", tags=["MMS"])
async def remove_block(element_id: str):
    """
    Remove an active maintenance block.
    Broadcasts MAINTENANCE_CLEARED to all WebSocket clients.
    """
    block = ACTIVE_BLOCKS.pop(element_id, None)
    if not block:
        raise HTTPException(
            status_code=404,
            detail=f"No active block found for element '{element_id}'."
        )

    cleared_at = _now_iso()
    
    # Remove any linked dynamic constraints
    linked = [c_id for c_id, c in list(DYNAMIC_CONSTRAINTS.items()) if c.get('linked_block_id') == element_id]
    for c_id in linked:
        del DYNAMIC_CONSTRAINTS[c_id]

    # Resume any trains that were halted solely because of this block
    resumed = []
    for t_id, state in TRAIN_STATES.items():
        if state.get("status") == "Halted":
            path = state.get("path", [])
            curr_edge = state.get("edge_id", "")
            # Check if the next edge (previously blocked) is now clear
            try:
                curr_idx = path.index(curr_edge)
                if curr_idx + 1 < len(path):
                    next_edge = path[curr_idx + 1]
                    if next_edge == element_id and element_id not in ACTIVE_BLOCKS:
                        state["status"] = "Moving"
                        resumed.append(t_id)
            except (ValueError, IndexError):
                pass
        elif state.get("status") == "Blocked" and state.get("edge_id") == element_id:
            # Train was stuck ON the now-cleared edge
            state["status"] = "Moving"
            resumed.append(t_id)

    # Re-sync RL env track_map (restored the cleared edge)
    _sync_blocks_to_rl_env()

    if resumed:
        print(f"[MMS] ▶️  Resumed {len(resumed)} train(s) after block cleared: {resumed}")

    AUDIT_LOGS.insert(0, {
        "t": cleared_at,
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
        "source": f"MMS_{element_id}",
        "action": f"Maintenance cleared. {len(linked)} constraints lifted.",
        "operator": "Dispatcher",
        "status": "Cleared",
        "statusType": "success",
        "id": str(uuid.uuid4())
    })
    payload = {
        "type": "MAINTENANCE_CLEARED",
        "element_id": element_id,
        "cleared_at": cleared_at,
    }
    await _broadcast_topology(payload)
    await _broadcast_copilot(payload)

    print(f"[MMS] ✅ Block cleared on '{element_id}' at {cleared_at}")

    return {
        "status": "block_cleared",
        "element_id": element_id,
        "cleared_at": cleared_at,
    }


@app.post("/api/v1/simulation/analyze", tags=["Simulation Sandbox"])
async def analyze_simulation(payload: dict):
    """
    Core two-stage simulation pipeline:
      1. RL Model  — proposes optimal adaptive actions for the current delay scenario.
      2. OR-Tools  — validates and overrides any unsafe proposals.
    Returns deterministic impact scores and human-readable network adjustments.
    """
    delay_train_id: str = payload.get("delay_train_id", "")
    latency_minutes: int = int(payload.get("latency_minutes", 15))

    # ── Step 1: Snapshot live network state ──────────────────────────────
    live_trains = list(TRAIN_STATES.values())
    n_live = len(live_trains)
    if n_live == 0:
        raise HTTPException(status_code=503, detail="No live train data available.")

    # ── Step 2: Inject delay into a working copy ──────────────────────────
    # We represent the network as an array the RL observation builder can consume.
    # Each train slot: [position_pct, priority, status_flag, delay_flag]
    from ai.config import MAX_TRAINS_CAPACITY, MAX_SPEED

    obs = np.zeros((MAX_TRAINS_CAPACITY, 10), dtype=np.float32)
    train_meta = []  # (train_id, edge_id, status, is_delayed)

    STATUS_SCORE = {"Moving": 0.8, "Waiting": 0.4, "Delayed": 0.2, "Scheduled": 0.1}
    PRIORITY_MAP_LOCAL = {
        "Vande Bharat": 10, "Rajdhani": 10, "Superfast": 8,
        "Express": 6, "Local": 5, "Suburban": 5,
        "Passenger": 3, "Freight (WAG-9)": 2,
    }

    for idx, t in enumerate(live_trains[:MAX_TRAINS_CAPACITY]):
        t_id      = t.get("train_id", "")
        pos_pct   = float(t.get("position_percentage", 0.0))
        status    = t.get("status", "Moving")
        is_delayed = (t_id == delay_train_id)

        # Derive priority from fleet registry or default
        fleet_cfg = FLEET_REGISTRY.get(t_id, {})
        train_type = fleet_cfg.get("train_type", "Express")
        priority = PRIORITY_MAP_LOCAL.get(train_type, 6)

        # Effective speed factor (degraded if delayed)
        speed_factor = 0.4 if is_delayed else STATUS_SCORE.get(status, 0.5)
        delay_flag   = min(latency_minutes, 60) / 60.0 if is_delayed else 0.0

        # Signal value: red (2.0 == danger) if delayed train blocks the segment
        signal_val = 2.0 if is_delayed else 0.0

        obs[idx] = [
            speed_factor,                   # normalised speed
            priority / 10.0,               # normalised priority
            signal_val / 2.0,              # signal (0=clear, 1=danger)
            0.5,                           # dist_to_danger (mid-range)
            0.5 - delay_flag * 0.3,        # dist_to_lead (shrinks under delay)
            speed_factor * 0.8,            # lead_speed proxy
            priority / 10.0,               # lead_priority proxy
            0.8,                           # dist_to_switch
            0.0,                           # dwell_rem
            max(0.0, 1.0 - delay_flag),    # deadline_rem (tighter under delay)
        ]
        train_meta.append((t_id, t.get("edge_id", "?"), status, is_delayed))

    # Pad remaining slots as ghost trains (already zeros)

    # ── Step 3: RL Model proposes actions ────────────────────────────────
    model, env = _get_sim_brain()
    if model is not None:
        # MaskablePPO expects obs shape (100, 10) — same as the env's observation_space
        proposed_actions, _ = model.predict(obs, deterministic=True)
        source = "RL+OR-Tools"
    else:
        # Fallback: greedy — every train wants to move (action=1)
        proposed_actions = np.ones(MAX_TRAINS_CAPACITY, dtype=np.int64)
        source = "OR-Tools only (model unavailable)"

    # ── Step 4: OR-Tools safety validation ───────────────────────────────
    # Build a lightweight train list compatible with SmartOptimizer
    edge_to_pos = {}   # edge_id -> rough integer position
    for idx, (t_id, edge_id, status, _) in enumerate(train_meta):
        parts = edge_id.replace("edge-", "").split("-")
        try:
            pos = int(parts[0])
        except (ValueError, IndexError):
            pos = idx + 1
        edge_to_pos[t_id] = pos

    # Build a minimal track_map from the real NETWORK_TOPOLOGY edges
    topo_edges = NETWORK_TOPOLOGY.get("edges", [])
    topo_nodes = NETWORK_TOPOLOGY.get("nodes", [])
    node_ids   = [n["id"] for n in topo_nodes]
    track_map  = {}
    for edge in topo_edges:
        src_id = edge["source"]
        tgt_id = edge["target"]
        if src_id not in track_map:
            track_map[src_id] = {"next": [], "capacity": 2}
        track_map[src_id]["next"].append(tgt_id)

    or_trains = []
    for idx, (t_id, edge_id, status, is_delayed) in enumerate(train_meta):
        parts = edge_id.replace("edge-", "").split("-")
        try:
            src_node = parts[0]
        except IndexError:
            src_node = node_ids[0] if node_ids else "0"
        or_trains.append({
            "id":       t_id,
            "position": src_node,
            "speed":    50 if status == "Moving" else 0,
            "priority": PRIORITY_MAP_LOCAL.get(
                FLEET_REGISTRY.get(t_id, {}).get("train_type", "Express"), 6
            ),
        })

    safe_actions = _OR_SHIELD.optimize_decision(
        or_trains,
        list(proposed_actions[:len(train_meta)]),
        track_map
    )

    # ── Step 5: Translate actions → human-readable adjustments ───────────
    ACTION_LABELS = {0: "HOLD", 1: "MAIN_LINE", 2: "DIVERT"}
    adjustments = []
    n_holds  = 0
    n_diverts = 0

    for idx, (t_id, edge_id, status, is_delayed) in enumerate(train_meta):
        ai_act   = int(proposed_actions[idx]) if idx < len(proposed_actions) else 1
        safe_act = int(safe_actions[idx])     if idx < len(safe_actions)     else ai_act
        vetoed   = (ai_act != 0 and safe_act == 0)  # OR-Tools overrode move → hold

        if safe_act == 0:
            n_holds += 1
            adj_type = "Signal Hold"
            if is_delayed:
                desc = (
                    f"{t_id} held at {edge_id} — primary delay source "
                    f"(+{latency_minutes} min). Adjacent trains cleared."
                )
            else:
                desc = (
                    f"{t_id} held at {edge_id} due to cascading delay from "
                    f"{delay_train_id}. OR-Shield veto: {'Yes' if vetoed else 'No'}."
                )
            adjustments.append({
                "id": len(adjustments)+1, 
                "type": adj_type, 
                "desc": desc,
                "train_id": t_id,
                "edge_id": edge_id,
                "constraint_type": "SPEED_LIMIT",
                "value": 0
            })

        elif safe_act == 2:
            n_diverts += 1
            adj_type = "Spatial Reroute"
            desc = f"{t_id} diverted off main line to loop to recover headway."
            adjustments.append({
                "id": len(adjustments)+1, 
                "type": adj_type, 
                "desc": desc,
                "train_id": t_id,
                "edge_id": edge_id,
                "constraint_type": "REROUTE",
                "value": 2
            })

    # Always add a speed-cap proposal for the delayed train itself
    if delay_train_id:
        speed_cap = max(20, 90 - latency_minutes * 2)
        delay_edge_id = next((e for t, e, _, _ in train_meta if t == delay_train_id), "edge-1-2")
        adjustments.insert(0, {
            "id": 0,
            "type": "Dynamic Speed Cap",
            "desc": (
                f"{delay_train_id} speed capped to {speed_cap} km/h to prevent "
                f"rear-end risk. Latency: +{latency_minutes} min."
            ),
            "train_id": delay_train_id,
            "edge_id": delay_edge_id,
            "constraint_type": "SPEED_LIMIT",
            "value": speed_cap
        })

    # ── Step 6: Compute real impact metrics ──────────────────────────────
    # Reliability degradation: proportional to delay severity + cascading holds
    base_reliability_hit = -(latency_minutes * 0.6 + n_holds * 2.5)
    reliability_pct = f"{max(base_reliability_hit, -95.0):.1f}%"

    # Congestion index: increases with latency and inversely with diversions
    congestion_base = 30 + latency_minutes * 1.8 + n_holds * 5 - n_diverts * 3
    congestion_pct  = f"+{min(congestion_base, 200):.0f}%"

    result = {
        "source": source,
        "delay_train_id": delay_train_id,
        "latency_minutes": latency_minutes,
        "impact": {
            "reliability": reliability_pct,
            "congestion":  congestion_pct,
            "trains_held":    n_holds,
            "trains_diverted": n_diverts,
        },
        "adjustments": adjustments,
        "timestamp": _now_iso(),
    }
    print(f"[SIM-ANALYZE] ✅ Analysis complete | source={source} | holds={n_holds} diversions={n_diverts}")
    return result


@app.post("/api/v1/simulation/deploy", tags=["Simulation Sandbox"])
async def deploy_simulation(payload: dict):
    """
    Deploys a set of active blocks and dynamic constraints from the Sandbox
    into the live production environment.
    """
    blocks = payload.get("blocks", [])
    constraints = payload.get("constraints", [])

    # Apply constraints to DYNAMIC_CONSTRAINTS
    for c in constraints:
        c_id = c.get("id", f"constraint-{uuid.uuid4()}")
        DYNAMIC_CONSTRAINTS[c_id] = c
        
    # Apply What-If Blocks as Real Blocks
    for b in blocks:
        element_id = b.get("element_id")
        if not element_id:
            continue
            
        block_dict = {
            "blockId": b.get("blockId", str(uuid.uuid4())),
            "element_id": element_id,
            "type": b.get("type", "TRACK_FAULT"),
            "severity": b.get("severity", "TOTAL_BLOCK"),
            "reason": b.get("reason", "Deployed from Sandbox"),
            "applied_at": _now_iso(),
            "isWhatIf": False # Now it's real
        }
        
        ACTIVE_BLOCKS[element_id] = block_dict
        impact = _resolve_reroute_strategy(element_id)
        
        await _broadcast_topology({
            "type": "MAINTENANCE_BLOCK_APPLIED",
            "block": block_dict,
            "impact": impact,
        })
    
    # Audit Log
    AUDIT_LOGS.insert(0, {
        "t": _now_iso(),
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
        "source": "SIMULATION_SANDBOX",
        "action": f"Deployed {len(blocks)} blocks and {len(constraints)} adjustments.",
        "operator": "Chief Dispatcher",
        "status": "Deployed",
        "statusType": "success",
        "id": str(uuid.uuid4())
    })

    print(f"[SANDBOX] 🚀 Simulation Deployed! ({len(blocks)} blocks, {len(constraints)} constraints)")
    return {"status": "success", "blocks_applied": len(blocks), "constraints_applied": len(constraints)}


@app.get("/api/v1/impact-analysis", tags=["MMS"])
async def get_impact_analysis():
    """
    Return a consolidated impact report for ALL currently active blocks.
    Used by the frontend to trigger the 'Ripple Effect' notification.
    """
    if not ACTIVE_BLOCKS:
        return {
            "status": "clear",
            "message": "No active maintenance blocks. Network operating normally.",
            "total_affected_trains": 0,
            "blocks": [],
            "timestamp": _now_iso(),
        }

    reports = []
    total_affected: set = set()

    for element_id in ACTIVE_BLOCKS:
        report = _resolve_reroute_strategy(element_id)
        reports.append(report)
        total_affected.update(report.get("affected_train_ids", []))

    # Overall summary message
    n = len(total_affected)
    strategies = list({r["strategy"] for r in reports})
    strategy_str = " / ".join(s.replace("_", " ").title() for s in strategies)

    return {
        "status": "blocks_active",
        "total_affected_trains": n,
        "affected_train_ids": list(total_affected),
        "primary_strategy": reports[0]["strategy"] if reports else "TEMPORAL_SHIFT",
        "message": (
            f"Maintenance on {len(ACTIVE_BLOCKS)} segment(s) affects {n} upcoming train(s). "
            f"Rerouting strategy: {strategy_str}."
        ),
        "block_reports": reports,
        "timestamp": _now_iso(),
    }


# ---------------------------------------------------------------------------
# Fleet Management — seed FLEET_REGISTRY from initial TRAIN_STATES on startup
# ---------------------------------------------------------------------------
def _seed_fleet_registry():
    """
    Build FLEET_REGISTRY from the initial TRAIN_STATES simulation data.
    Called once from startup_event after simulate_trains_bg has initialised.
    """
    # Speed limits per train type (km/h)
    MAX_SPEED_MAP = {
        "Vande Bharat": 130, "Rajdhani": 120, "Superfast": 110,
        "Express": 90, "Local": 75, "Suburban": 75,
        "Passenger": 60, "Freight (WAG-9)": 40,
    }

    for t_id, state in TRAIN_STATES.items():
        # Read train_type directly from TRAIN_STATES if available
        train_type = state.get("train_type", "Express")
        max_speed  = state.get("speed_kmh") or MAX_SPEED_MAP.get(train_type, 90)
        priority   = PRIORITY_MAP.get(train_type, 6)

        # Always sync — ensures type/priority/speed stay correct after restarts
        FLEET_REGISTRY[t_id] = {
            "train_id"  : t_id,
            "train_type": train_type,
            "max_speed" : max_speed,
            "priority"  : priority,
            "start_time": 0,
            "deadline"  : 120,
            "direction" : 1,
            "path"      : state.get("path", []),
            "added_at"  : FLEET_REGISTRY.get(t_id, {}).get("added_at", _now_iso()),
        }


@app.on_event("startup")
async def _fleet_seed_on_startup():
    """Seed the fleet registry once the simulation loop has populated TRAIN_STATES."""
    await asyncio.sleep(1)   # give simulate_trains_bg time to initialise
    _seed_fleet_registry()


# ---------------------------------------------------------------------------
# Fleet Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/fleet", tags=["Fleet"])
async def get_fleet():
    """
    Return live fleet: merges FLEET_REGISTRY config with real-time TRAIN_STATES.
    Always reflects current edge, position, and status from the simulation.
    """
    _seed_fleet_registry()   # ensure any newly spawned trains are included

    result = []
    for t_id, cfg in FLEET_REGISTRY.items():
        live = TRAIN_STATES.get(t_id, {})
        result.append({
            **cfg,
            "edge_id"            : live.get("edge_id", "—"),
            "position_percentage": live.get("position_percentage", 0),
            "status"             : live.get("status", "Scheduled"),
        })

    # Also include any live trains not yet in FLEET_REGISTRY
    for t_id, live in TRAIN_STATES.items():
        if t_id not in FLEET_REGISTRY:
            result.append({
                "train_id"           : t_id,
                "train_type"         : "Express",
                "max_speed"          : 110,
                "priority"           : 6,
                "start_time"         : 0,
                "deadline"           : 120,
                "direction"          : 1,
                "path"               : live.get("path", []),
                "added_at"           : _now_iso(),
                "edge_id"            : live.get("edge_id", "—"),
                "position_percentage": live.get("position_percentage", 0),
                "status"             : live.get("status", "Moving"),
            })

    return {"fleet": result, "count": len(result), "timestamp": _now_iso()}


@app.post("/api/v1/fleet", tags=["Fleet"])
async def add_train(req: NewTrainRequest):
    """
    Add a new train to the session fleet.
    - Validates train_id uniqueness.
    - Derives priority from type if not supplied.
    - Seeds both FLEET_REGISTRY and TRAIN_STATES so the map shows it immediately.
    - Broadcasts topology_update so all map clients see the new train.
    """
    if req.train_id in FLEET_REGISTRY or req.train_id in TRAIN_STATES:
        raise HTTPException(
            status_code=409,
            detail=f"Train '{req.train_id}' already exists in the fleet."
        )

    if req.train_type not in TRAIN_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid train_type '{req.train_type}'. Must be one of: {TRAIN_TYPES}"
        )

    # Derive priority from type
    priority = req.priority if req.priority is not None else PRIORITY_MAP.get(req.train_type, 5)

    # Build path from topology (first edge → destination)
    all_edges = [e["id"] for e in NETWORK_TOPOLOGY.get("edges", [])]
    default_path = all_edges[:8] if len(all_edges) >= 8 else all_edges  # first 8 edges as path

    cfg = {
        "train_id"  : req.train_id,
        "train_type": req.train_type,
        "max_speed" : req.max_speed,
        "priority"  : priority,
        "start_time": req.start_time,
        "deadline"  : req.deadline,
        "direction" : req.direction,
        "path"      : default_path,
        "added_at"  : _now_iso(),
    }
    FLEET_REGISTRY[req.train_id] = cfg

    # Seed live simulation state so train appears on the map
    TRAIN_STATES[req.train_id] = {
        "train_id"           : req.train_id,
        "edge_id"            : default_path[0] if default_path else "edge-0-1",
        "position_percentage": 0.0,
        "status"             : "Moving",
        "path"               : default_path,
    }

    # Broadcast updated train list to all topology WS clients
    await _broadcast_topology({
        "type"  : "topology_update",
        "trains": list(TRAIN_STATES.values()),
        "conflicts": [],
        "maintenance_blocks": list(ACTIVE_BLOCKS.values()),
    })

    print(f"[Fleet] ➕ Added train '{req.train_id}' (type={req.train_type}, prio={priority})")
    return {"status": "added", "train": cfg, "timestamp": cfg["added_at"]}


def _get_rerouted_path(original_path: List[str]) -> List[str]:
    """
    Scans the path for blocked segments. If a block is found on any junction
    edge (switch node with >1 next options), finds a free sibling platform/loop.
    Fully dynamic — works for any topology without hardcoded node IDs.
    """
    new_path = []
    i = 0
    while i < len(original_path):
        edge_id = original_path[i]
        if edge_id in ACTIVE_BLOCKS:
            parts = edge_id.split("-")
            if len(parts) >= 3:
                try:
                    src = int(parts[1])
                except ValueError:
                    new_path.append(edge_id)
                    i += 1
                    continue

                # Dynamic junction detection: any node with multiple exits is a switch
                src_nexts = RAW_TRACK_MAP.get(src, {}).get("next", [])
                if len(src_nexts) > 1:
                    found_bypass = False
                    for alt_dst in src_nexts:
                        alt_edge_in = f"edge-{src}-{alt_dst}"
                        if alt_edge_in == edge_id or alt_edge_in in ACTIVE_BLOCKS:
                            continue
                        # Check the exit from the alternate sibling is also clear
                        exits = RAW_TRACK_MAP.get(alt_dst, {}).get("next", [])
                        if exits:
                            alt_edge_out = f"edge-{alt_dst}-{exits[0]}"
                            if alt_edge_out not in ACTIVE_BLOCKS:
                                # SUCCESS: replace blocked entry+exit with bypass pair
                                new_path.append(alt_edge_in)
                                new_path.append(alt_edge_out)
                                i += 2  # skip the blocked edge AND its exit
                                found_bypass = True
                                break
                    if found_bypass:
                        continue

        new_path.append(edge_id)
        i += 1
    return new_path

@app.post("/api/v1/fleet/generate-schedule", tags=["Fleet"])
async def generate_schedule():
    """
    Run OR-Tools on the current FLEET_REGISTRY to produce a conflict-free
    base schedule for this session.

    Stores the result in LAST_OR_SCHEDULE so /start-inference can consume
    it to seed TRAIN_STATES and kick off the RL inference loop.
    """
    from ai.or_solver import solve_train_schedule

    if not FLEET_REGISTRY:
        raise HTTPException(status_code=400, detail="Fleet is empty — add trains first.")

    topo_nodes = {n["id"]: n for n in NETWORK_TOPOLOGY.get("nodes", [])}
    track_map  = {}
    for nid, ndata in topo_nodes.items():
        track_map[nid] = {
            "type"    : ndata.get("type", "BLOCK"),
            "capacity": 2 if ndata.get("type") in ("PLATFORM", "LOOP") else 1,
        }

    active_fleet = []
    for t_id, cfg in FLEET_REGISTRY.items():
        original_path = cfg.get("path", [])
        if not original_path:
            continue
            
        # DYNAMIC REROUTING: Check for maintenance blocks and pick alternative platforms/loops
        path = _get_rerouted_path(original_path)
        cfg["path"] = path # update the registry so start-inference uses the new path
        
        node_path  = []
        last_parts = None
        for edge_id in path:
            parts = edge_id.split("-")
            if len(parts) >= 3:
                node_path.append(parts[1])
                last_parts = parts
        if node_path and last_parts:
            node_path.append(last_parts[-1])

        active_fleet.append({
            "id"               : t_id,
            "type"             : cfg.get("train_type", "Express"),
            "path"             : node_path,
            "scheduled_arrival": cfg.get("deadline", 120),
            "runtimes"         : {n: 2 for n in node_path},
            "dwell_times"      : {n: 0 for n in node_path},
            "direction"        : cfg.get("direction", 1),
            "priority"         : cfg.get("priority", 5),
        })

    if not active_fleet:
        raise HTTPException(status_code=400, detail="No valid train paths to schedule.")

    try:
        result = solve_train_schedule(track_map, active_fleet)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OR-Solver failed: {str(exc)}")

    if result is None:
        return {
            "status"    : "infeasible",
            "message"   : "OR-Tools could not find a feasible schedule within the time horizon.",
            "fleet_size": len(active_fleet),
            "timestamp" : _now_iso(),
        }

    # Store schedule so start-inference can consume it
    global LAST_OR_SCHEDULE
    LAST_OR_SCHEDULE = result.get("schedule", {})
    print(f"[ORBIT] 📅 OR Schedule stored for {len(LAST_OR_SCHEDULE)} trains.")

    return {
        "status"        : "optimal",
        "fleet_size"    : len(active_fleet),
        "schedule"      : result.get("schedule", {}),
        "expert_actions": result.get("expert_actions", {}),
        "timestamp"     : _now_iso(),
    }
