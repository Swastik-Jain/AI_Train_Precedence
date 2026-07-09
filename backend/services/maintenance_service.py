from typing import Dict, Any
from datetime import datetime, timezone
import uuid
from fastapi import HTTPException

from state import SimulationState
from schema import InfrastructureBlock
from services import system_service
from config import DOWN_PATH, UP_PATH

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")

def is_block_active(block: Dict[str, Any]) -> bool:
    try:
        now = datetime.now(timezone.utc)
        start = datetime.fromisoformat(block.get("start_time", "").replace("Z", "+00:00"))
        end = datetime.fromisoformat(block.get("end_time", "").replace("Z", "+00:00"))
        return start <= now <= end
    except Exception as e:
        print(f"[WARN] Failed to parse block timestamp for {block.get('element_id', 'unknown')}: {e}")
        return True

def sync_blocks_to_rl_env(state: SimulationState) -> None:
    if state.sim_env is None or not state.inference_active:
        return
    try:
        inner_env = state.sim_env.venv.envs[0] if hasattr(state.sim_env, 'venv') else state.sim_env.envs[0]
        import copy
        patched_map = copy.deepcopy(state.raw_track_map)

        for edge_id, block in state.active_blocks.items():
            if not is_block_active(block):
                continue
            parts = edge_id.split("-")
            if len(parts) < 3:
                continue
            try:
                src = int(parts[1])
                dst = int(parts[2])
            except ValueError:
                continue

            if block.get("severity") == "TOTAL_BLOCK":
                if src in patched_map and dst in patched_map[src].get("next", []):
                    patched_map[src]["next"] = [n for n in patched_map[src]["next"] if n != dst]
                if dst in patched_map and src in patched_map[dst].get("prev", []):
                    patched_map[dst]["prev"] = [n for n in patched_map[dst]["prev"] if n != src]
            elif block.get("severity") == "SPEED_RESTRICTION":
                limit = block.get("speed_limit", 30)
                if dst in patched_map:
                    patched_map[dst]["speed"] = limit

        if hasattr(inner_env, 'track_map'):
            inner_env.track_map = patched_map
    except Exception as e:
        print(f"[ERROR] _sync_blocks_to_rl_env failed: {e}")

def _resolve_reroute_strategy(state: SimulationState, element_id: str) -> Dict[str, Any]:
    affected_trains = []
    strategy = "NONE"
    strategy_details = ""
    
    parts = element_id.split("-")
    if len(parts) >= 3:
        try:
            blocked_src = int(parts[1])
            blocked_dst = int(parts[2])
            
            for t_id, train_state in state.train_states.items():
                if "path" in train_state and element_id in train_state["path"]:
                    affected_trains.append(t_id)
            
            src_nexts = state.raw_track_map.get(blocked_src, {}).get("next", [])
            if len(src_nexts) > 1:
                strategy = "DYNAMIC_REROUTE"
                strategy_details = "Rerouting via sibling platform/loop."
            elif any(t for t in affected_trains if "DOWN" in state.train_states[t].get("direction", "DOWN")):
                strategy = "WAIT_FOR_CLEARANCE"
                strategy_details = "No alternate path available."
            else:
                strategy = "WAIT_FOR_CLEARANCE"
        except Exception:
            pass

    return {
        "strategy": strategy,
        "affected_trains": len(affected_trains),
        "details": strategy_details
    }

async def apply_block(state: SimulationState, block: InfrastructureBlock, broadcast_topology, broadcast_copilot) -> Dict[str, Any]:
    block_dict = block.model_dump()
    block_dict["applied_at"] = _now_iso()
    state.active_blocks[block.element_id] = block_dict
    
    system_service.push_audit_log(state, {
        "t": block_dict["applied_at"],
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
        "source": f"MMS_{block.element_id}",
        "action": f"Maintenance Applied: {block.severity} ({block.reason or 'Scheduled'})",
        "operator": "Dispatcher",
        "status": "Active",
        "statusType": "error",
        "id": str(uuid.uuid4())
    })

    impact = _resolve_reroute_strategy(state, block.element_id)

    sync_blocks_to_rl_env(state)

    if broadcast_topology:
        await broadcast_topology({
            "type": "MAINTENANCE_BLOCK_APPLIED",
            "block": block_dict,
            "impact": impact,
        })
    if broadcast_copilot:
        await broadcast_copilot({
            "type": "MAINTENANCE_BLOCK_APPLIED",
            "block": block_dict,
            "impact": impact,
        })

    return {
        "status": "block_applied",
        "block": block_dict,
        "impact": impact,
        "timestamp": block_dict["applied_at"],
    }

async def remove_block(state: SimulationState, element_id: str, broadcast_topology, broadcast_copilot) -> Dict[str, Any]:
    block = state.active_blocks.pop(element_id, None)
    if not block:
        raise HTTPException(status_code=404, detail=f"No active block found for element '{element_id}'.")

    cleared_at = _now_iso()
    
    linked = [c_id for c_id, c in list(state.dynamic_constraints.items()) if c.get('linked_block_id') == element_id]
    for c_id in linked:
        del state.dynamic_constraints[c_id]

    resumed = []
    for t_id, t_state in state.train_states.items():
        if t_state.get("status") == "Halted":
            path = t_state.get("path", [])
            curr_edge = t_state.get("edge_id", "")
            try:
                curr_idx = path.index(curr_edge)
                if curr_idx + 1 < len(path):
                    next_edge = path[curr_idx + 1]
                    if next_edge == element_id and element_id not in state.active_blocks:
                        t_state["status"] = "Moving"
                        resumed.append(t_id)
            except (ValueError, IndexError):
                pass
        elif t_state.get("status") == "Blocked" and t_state.get("edge_id") == element_id:
            t_state["status"] = "Moving"
            resumed.append(t_id)

    sync_blocks_to_rl_env(state)

    system_service.push_audit_log(state, {
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
    
    if broadcast_topology:
        await broadcast_topology(payload)
    if broadcast_copilot:
        await broadcast_copilot(payload)

    return {
        "status": "block_cleared",
        "element_id": element_id,
        "cleared_at": cleared_at,
    }

def add_sandbox_block(state: SimulationState, block: InfrastructureBlock) -> Dict[str, Any]:
    block_dict = block.model_dump()
    block_dict["applied_at"] = _now_iso()
    block_dict["isWhatIf"] = True
    state.sandbox_blocks[block.element_id] = block_dict
    return {"status": "sandbox_only", "block": block_dict}

def remove_sandbox_block(state: SimulationState, element_id: str) -> Dict[str, Any]:
    removed = state.sandbox_blocks.pop(element_id, None)
    if not removed:
        raise HTTPException(status_code=404, detail=f"No sandbox block found for '{element_id}'")
    return {"status": "removed", "element_id": element_id}
