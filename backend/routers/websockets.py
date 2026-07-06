from fastapi import APIRouter, WebSocket, Depends
import json

from state import get_state, SimulationState
from config import SUGGESTION_TTL_TICKS

router = APIRouter(tags=["WebSockets"])

async def broadcast_topology(payload: dict):
    state = get_state()
    dead_sockets = set()
    message = json.dumps(payload)
    for ws in state.active_websockets:
        try:
            await ws.send_text(message)
        except Exception:
            dead_sockets.add(ws)
    state.active_websockets.difference_update(dead_sockets)

async def broadcast_copilot(payload: dict):
    state = get_state()
    dead_sockets = set()
    message = json.dumps(payload)
    for ws in state.copilot_websockets:
        try:
            await ws.send_text(message)
        except Exception:
            dead_sockets.add(ws)
    state.copilot_websockets.difference_update(dead_sockets)

@router.websocket("/ws/traffic")
async def traffic_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except Exception as e:
        print(f"Traffic WS disconnected: {e}")

@router.websocket("/ws/topology")
async def topology_websocket(websocket: WebSocket):
    await websocket.accept()
    state = get_state()
    state.active_websockets.add(websocket)
    await websocket.send_text(json.dumps({
        "type": "topology_init",
        "topology": state.network_topology
    }))
    try:
        while True:
            await websocket.receive_text()
    except Exception as e:
        state.active_websockets.discard(websocket)
        print(f"Topology WS disconnected: {e}")

@router.websocket("/ws/copilot")
async def copilot_websocket(websocket: WebSocket):
    await websocket.accept()
    state = get_state()
    state.copilot_websockets.add(websocket)
    
    pending = [s for s in state.copilot_suggestions.values() if s.get("status") == "pending"]
    pending = [s for s in pending if state.sim_tick - s.get("suggested_at_tick", state.sim_tick) <= SUGGESTION_TTL_TICKS]
    
    for suggestion in pending[-3:]:
        try:
            await websocket.send_text(json.dumps(suggestion))
        except Exception as e:
            print(f"[WARN] Failed to send initial suggestion to copilot websocket: {e}")
            break

    try:
        while True:
            await websocket.receive_text()
    except Exception as e:
        state.copilot_websockets.discard(websocket)
        print(f"[ORBIT] Co-pilot WS disconnected: {e}")
