from typing import Dict, Any, List
from datetime import datetime, timezone
import uuid

import database
import crud
from state import SimulationState

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")

def push_audit_log(state: SimulationState, entry: dict) -> None:
    # AUDIT_LOGS_MAX_SIZE = 500
    state.audit_logs.insert(0, entry)
    if len(state.audit_logs) > 500:
        del state.audit_logs[500:]

def persist_audit_log(entry: dict):
    try:
        db = database.SessionLocal()
        crud.create_audit_log(db, entry)
        db.close()
    except Exception as _e:
        print(f"[ORBIT] Warning: audit log DB write failed: {_e}")

def get_audit_logs(state: SimulationState, limit: int = 50, skip: int = 0) -> Dict[str, Any]:
    try:
        db = database.SessionLocal()
        db_logs = crud.get_recent_audit_logs(db, limit=limit + len(state.audit_logs), skip=0)
        db.close()
        db_entries = [
            {
                "id": row.log_id,
                "t": row.timestamp,
                "timestamp": row.timestamp_ms,
                "source": row.source,
                "action": row.action,
                "operator": row.operator,
                "status": row.status,
                "statusType": row.status_type,
            }
            for row in db_logs
        ]
    except Exception as e:
        print(f"[WARN] Failed to fetch audit logs from database: {e}")
        db_entries = []

    seen_ids = set()
    merged = []
    for entry in state.audit_logs + db_entries:
        eid = entry.get("id", "")
        if eid and eid in seen_ids:
            continue
        seen_ids.add(eid)
        merged.append(entry)

    merged.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return {
        "logs": merged[skip : skip + limit],
        "total": len(merged)
    }

def set_sim_speed(state: SimulationState, factor: float) -> Dict[str, Any]:
    # factor = 0.3x should mean SLOWER, so tick interval gets LONGER.
    # Base tick interval is 1.0s. So interval = 1.0 / factor
    # E.g. factor 0.3 -> 3.33s. factor 1.0 -> 1.0s. factor 2.0 -> 0.5s
    new_interval = 1.0 / max(0.1, factor)
    state.tick_interval_s = max(0.05, min(new_interval, 10.0))
    return {"status": "success", "sim_speed": state.tick_interval_s}

def toggle_lockdown(state: SimulationState, enabled: bool) -> Dict[str, Any]:
    state.system_lockdown = enabled
    if not state.system_lockdown:
        for t_id, t_state in state.train_states.items():
            if t_state.get("status") == "Halted":
                edge_id = t_state.get("edge_id")
                is_blocked = False
                if edge_id in state.active_blocks and state.active_blocks[edge_id].get("severity") == "TOTAL_BLOCK":
                    is_blocked = True
                
                if not is_blocked:
                    t_state["status"] = "Moving"
                
    status_text = "ACTIVATED" if state.system_lockdown else "DEACTIVATED"
    entry = {
        "t"         : _now_iso(),
        "timestamp" : int(datetime.now(timezone.utc).timestamp() * 1000),
        "source"    : "SYSTEM_CONTROL",
        "action"    : f"Emergency Stop {status_text}",
        "operator"  : "Dispatcher",
        "status"    : "Lockdown" if state.system_lockdown else "Nominal",
        "statusType": "error" if state.system_lockdown else "success",
        "id"        : str(uuid.uuid4())
    }
    push_audit_log(state, entry)
    persist_audit_log(entry)
    return {"status": "success", "lockdown": state.system_lockdown}

def toggle_safety_shield(state: SimulationState, enabled: bool) -> Dict[str, Any]:
    state.or_shield_enabled = enabled
    status_text = "ACTIVATED" if state.or_shield_enabled else "DEACTIVATED"
    entry = {
        "t"         : _now_iso(),
        "timestamp" : int(datetime.now(timezone.utc).timestamp() * 1000),
        "source"    : "SYSTEM_CONTROL",
        "action"    : f"OR-Shield Safety Protocol {status_text}",
        "operator"  : "Dispatcher",
        "status"    : "Active" if state.or_shield_enabled else "Disabled",
        "statusType": "success" if state.or_shield_enabled else "warning",
        "id"        : str(uuid.uuid4())
    }
    push_audit_log(state, entry)
    persist_audit_log(entry)
    return {"status": "success", "safety_shield": state.or_shield_enabled}

def toggle_autopilot(state: SimulationState, enabled: bool) -> Dict[str, Any]:
    state.autopilot_mode = enabled
    state.ai_auto_commit = enabled
    status_text = "ACTIVATED" if state.autopilot_mode else "DEACTIVATED"
    entry = {
        "t": _now_iso(),
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
        "source": "SYSTEM_CONTROL",
        "action": f"Autopilot Mode {status_text}",
        "operator": "Dispatcher",
        "status": "Active" if state.autopilot_mode else "Disabled",
        "statusType": "warning" if state.autopilot_mode else "success",
        "id": str(uuid.uuid4())
    }
    push_audit_log(state, entry)
    persist_audit_log(entry)
    return {"status": "success", "autopilot_mode": state.autopilot_mode, "auto_commit": state.autopilot_mode}
