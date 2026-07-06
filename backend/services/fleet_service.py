from typing import Dict, Any, List
from fastapi import HTTPException
from datetime import datetime, timezone

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
        edge_id = live.get("edge_id")
        if not edge_id or edge_id == "—":
            direction = cfg.get("direction", "DOWN")
            edge_id = "edge-0-1" if direction == "DOWN" else "edge-83-999"
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

def remove_train(state: SimulationState, train_id: str) -> Dict[str, Any]:
    state.fleet_registry.pop(train_id, None)
    removed = state.train_states.pop(train_id, None)
    if removed is None:
        raise HTTPException(status_code=404, detail=f"Train '{train_id}' not found.")
    return {"status": "removed", "train_id": train_id}

async def add_train(state: SimulationState, req: NewTrainRequest, broadcast_fn) -> Dict[str, Any]:
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
        "status"             : "Moving",
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

    if len(state.fleet_registry) < 25:
        needed = 25 - len(state.fleet_registry)
        fleet, schedule_map = generate_daily_schedule(num_trains=needed)
        for t in fleet:
            t_sched = schedule_map.get(t['id'], {})
            base_id = t['id']
            unique_id = base_id
            counter = 1
            while unique_id in state.fleet_registry:
                unique_id = f"{base_id}-{counter}"
                counter += 1
            t['id'] = unique_id
            t['train_id'] = unique_id
            t['path'] = []
            t['train_type'] = t.get('archetype', 'Express')
            t['start_time'] = t_sched.get('start_time', 0)
            t['deadline'] = t_sched.get('deadline', 100)
            dir_str = "UP" if "UP" in unique_id else "DOWN"
            t['direction'] = t.get('direction', dir_str)
            state.fleet_registry[t['id']] = t
    elif not state.fleet_registry:
        fleet, schedule_map = generate_daily_schedule(num_trains=25)
        for t in fleet:
            t_sched = schedule_map.get(t['id'], {})
            t['path'] = []
            t['train_id'] = t['id']
            t['train_type'] = t.get('archetype', 'Express')
            t['start_time'] = t_sched.get('start_time', 0)
            t['deadline'] = t_sched.get('deadline', 100)
            state.fleet_registry[t['id']] = t

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
