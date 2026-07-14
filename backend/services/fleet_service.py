from typing import Dict, Any, List
from fastapi import HTTPException
from datetime import datetime, timezone
from sqlalchemy.orm import Session
import crud

from state import SimulationState
from config import TRAIN_TYPES, PRIORITY_MAP, DOWN_PATH, UP_PATH
from schema import NewTrainRequest

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")

def get_fleet_live(state: SimulationState) -> Dict[str, Any]:
    _seed_fleet_registry(state)
    result = []
    for t_id, cfg in state.fleet_registry.items():
        live = state.train_states.get(t_id, {})
        status = live.get("status", "Scheduled")
        # None (not a placeholder edge string) means "no live position yet" —
        # i.e. this train hasn't been through start_inference()/a tick since.
        # Let the frontend decide how to display that, rather than fabricating
        # a real-looking edge_id here.
        edge_id = live.get("edge_id") if live.get("edge_id") not in (None, "—") else None
        result.append({
            **cfg,
            "edge_id"            : edge_id,
            "position_percentage": live.get("position_percentage", 0),
            "status"             : status,
            "speed_kmh"          : live.get("speed_kmh", 0),
        })

    for t_id, live in state.train_states.items():
        if t_id not in state.fleet_registry:
            result.append({
                "train_id"           : t_id,
                "train_type"         : "Express",
                "max_speed"          : 110,
                "priority"           : 6,
                "start_time"         : 0,
                "deadline"           : 120,
                "direction"          : live.get("direction", "UP" if "UP" in t_id else "DOWN"),
                "path"               : live.get("path", []),
                "added_at"           : _now_iso(),
                "edge_id"            : live.get("edge_id", "—"),
                "position_percentage": live.get("position_percentage", 0),
                "status"             : live.get("status", "Moving"),
                "speed_kmh"          : live.get("speed_kmh", 0),
            })
    return {"fleet": result, "count": len(result), "timestamp": _now_iso()}

def _seed_fleet_registry(state: SimulationState):
    for t_id, live in state.train_states.items():
        if t_id not in state.fleet_registry:
            direction = live.get("direction", "UP" if "UP" in t_id else "DOWN")
            state.fleet_registry[t_id] = {
                "train_id": t_id,
                "train_type": "Express",
                "max_speed": 110,
                "priority": 6,
                "start_time": 0,
                "deadline": 120,
                "direction": direction,
                "path": live.get("path", UP_PATH if direction == "UP" else DOWN_PATH),
                "added_at": _now_iso()
            }

def remove_train(state: SimulationState, train_id: str, db: Session = None) -> Dict[str, Any]:
    state.fleet_registry.pop(train_id, None)
    removed = state.train_states.pop(train_id, None)
    if removed is None:
        raise HTTPException(status_code=404, detail=f"Train '{train_id}' not found.")
    
    if db:
        crud.remove_fleet_train(db, train_id)
        
    return {"status": "removed", "train_id": train_id}

async def add_train(state: SimulationState, req: NewTrainRequest, broadcast_fn, db: Session = None) -> Dict[str, Any]:
    if req.train_id in state.fleet_registry or req.train_id in state.train_states:
        raise HTTPException(status_code=409, detail=f"Train '{req.train_id}' already exists.")
    if req.train_type not in TRAIN_TYPES:
        raise HTTPException(status_code=422, detail=f"Invalid train_type '{req.train_type}'.")

    priority = req.priority if req.priority is not None else PRIORITY_MAP.get(req.train_type.title(), 5)
    dir_str = "DOWN" if req.direction == 1 else "UP"
    default_path = DOWN_PATH if dir_str == "DOWN" else UP_PATH

    cfg = {
        "train_id"  : req.train_id,
        "train_type": req.train_type,
        "max_speed" : req.max_speed,
        "priority"  : priority,
        "start_time": req.start_time,
        "deadline"  : req.deadline,
        "direction" : dir_str,
        "path"      : default_path,
        "added_at"  : _now_iso(),
    }
    state.fleet_registry[req.train_id] = cfg
    state.train_states[req.train_id] = {
        "train_id"           : req.train_id,
        "edge_id"            : default_path[0] if default_path else "edge-0-1",
        "position_percentage": 0.0,
        "status"             : "Scheduled",
        "path"               : default_path,
        "direction"          : dir_str,
    }

    if broadcast_fn:
        await broadcast_fn({
            "type"  : "topology_update",
            "trains": list(state.train_states.values()),
            "conflicts": [],
            "maintenance_blocks": list(state.active_blocks.values()),
        })

    if db:
        crud.add_fleet_train(db, cfg)

    return {"status": "added", "train": cfg, "timestamp": cfg["added_at"]}

def get_schedule(state: SimulationState) -> Dict[str, Any]:
    if state.last_or_schedule:
        return {"status": "optimal", "fleet_size": len(state.last_or_schedule), "schedule": state.last_or_schedule}
    return {"status": "empty", "fleet_size": 0, "schedule": {}}

def _get_rerouted_path(state: SimulationState, original_path: List[str]) -> List[str]:
    new_path = []
    i = 0
    while i < len(original_path):
        edge_id = original_path[i]
        if edge_id in state.active_blocks:
            parts = edge_id.split("-")
            if len(parts) >= 3:
                try:
                    src = int(parts[1])
                except ValueError:
                    new_path.append(edge_id)
                    i += 1
                    continue
                src_nexts = state.raw_track_map.get(src, {}).get("next", [])
                if len(src_nexts) > 1:
                    found_bypass = False
                    for alt_dst in src_nexts:
                        alt_edge_in = f"edge-{src}-{alt_dst}"
                        if alt_edge_in == edge_id or alt_edge_in in state.active_blocks:
                            continue
                        exits = state.raw_track_map.get(alt_dst, {}).get("next", [])
                        if exits:
                            alt_edge_out = f"edge-{alt_dst}-{exits[0]}"
                            if alt_edge_out not in state.active_blocks:
                                new_path.append(alt_edge_in)
                                new_path.append(alt_edge_out)
                                i += 2
                                found_bypass = True
                                break
                    if found_bypass:
                        continue
        new_path.append(edge_id)
        i += 1
    return new_path

def generate_schedule(state: SimulationState) -> Dict[str, Any]:
    from or_tools.corridor_planner import CorridorPlanner
    from ai.map_generator import STATIONS, generate_realistic_section
    from ai.config import generate_daily_schedule, ARCHETYPE_BY_NAME



    topo_nodes = {n["id"]: n for n in state.network_topology.get("nodes", [])}
    track_map  = {}
    for nid, ndata in topo_nodes.items():
        track_map[nid] = {
            "type"    : ndata.get("type", "BLOCK"),
            "capacity": 2 if ndata.get("type") in ("PLATFORM", "LOOP") else 1,
            "next"    : ndata.get("next", [])
        }

    _, _, _, _, token_blocks = generate_realistic_section()
    planner = CorridorPlanner(track_map, STATIONS, token_blocks)

    active_fleet = []
    schedule_req = {}

    for t_id, cfg in state.fleet_registry.items():
        original_path = cfg.get("path", [])
        if original_path:
            path = _get_rerouted_path(state, original_path)
            cfg["path"] = path

        train_type = cfg.get("train_type", "Express")
        direction_val = cfg.get("direction", 1)
        direction_str = "DOWN" if direction_val in (1, "DOWN") else "UP"

        train_type_upper = train_type.upper()
        if "EXPRESS" in train_type_upper or "MAIL" in train_type_upper:
            train_type_upper = "MAIL_EXPRESS"
        elif "FREIGHT" in train_type_upper or "GOODS" in train_type_upper:
            train_type_upper = "GOODS"
        elif "LOCAL" in train_type_upper or "SUBURBAN" in train_type_upper:
            train_type_upper = "PASSENGER"
        elif "VANDE BHARAT" in train_type_upper:
            train_type_upper = "RAJDHANI"

        archetype = ARCHETYPE_BY_NAME.get(train_type_upper, ARCHETYPE_BY_NAME["MAIL_EXPRESS"])
        stops = archetype.get("stops_down" if direction_str == "DOWN" else "stops_up", [])

        active_fleet.append({
            "id": t_id,
            "direction": direction_str,
            "priority": cfg.get("priority", 5),
            "max_speed": cfg.get("max_speed", 100),
            "banker_required": archetype.get("banker_required", False),
            "finished": False,
            "position": 0 if direction_str == "DOWN" else 998,
        })
        schedule_req[t_id] = {
            "stops": stops,
            "start_time": cfg.get("start_time", 0),
            "deadline": cfg.get("deadline", 120),
        }

    if not active_fleet:
        raise HTTPException(status_code=400, detail="No valid trains to schedule.")

    result = planner.solve(active_fleet, schedule_req, sim_time=0)
    if result is None:
        return {
            "status"    : "infeasible",
            "message"   : "OR-Tools could not find a feasible schedule.",
            "fleet_size": len(active_fleet),
            "timestamp" : _now_iso(),
        }

    state.last_or_schedule = result.get("schedule", {})
    return {
        "status"        : "optimal",
        "fleet_size"    : len(active_fleet),
        "schedule"      : result.get("schedule", {}),
        "expert_actions": result.get("expert_actions", {}),
        "timestamp"     : _now_iso(),
    }
