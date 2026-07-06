from fastapi import APIRouter, Depends
from state import get_state, SimulationState
from schema import SimSpeedRequest, SystemToggleRequest
from services import system_service
from services import simulation_service

router = APIRouter(prefix="/api/v1/system", tags=["System Override", "System Controls"])

@router.post("/start-inference")
async def start_inference(state: SimulationState = Depends(get_state)):
    return simulation_service.start_inference(state)

@router.post("/stop-inference")
async def stop_inference(state: SimulationState = Depends(get_state)):
    return simulation_service.stop_inference(state)

@router.get("/inference-status")
async def get_inference_status(state: SimulationState = Depends(get_state)):
    return {
        "active": state.inference_active,
        "safety_shield": state.or_shield_enabled,
        "auto_commit": state.autopilot_mode,
        "autopilot_mode": state.autopilot_mode,
        "lockdown": state.system_lockdown
    }

@router.post("/sim-speed")
async def set_sim_speed(req: SimSpeedRequest, state: SimulationState = Depends(get_state)):
    return system_service.set_sim_speed(state, req.factor)

@router.post("/lockdown")
async def toggle_lockdown(req: SystemToggleRequest, state: SimulationState = Depends(get_state)):
    return system_service.toggle_lockdown(state, req.enabled)

@router.post("/safety-shield")
async def toggle_safety_shield(req: SystemToggleRequest, state: SimulationState = Depends(get_state)):
    return system_service.toggle_safety_shield(state, req.enabled)

@router.post("/auto-commit")
async def toggle_auto_commit_legacy(req: SystemToggleRequest, state: SimulationState = Depends(get_state)):
    return system_service.toggle_autopilot(state, req.enabled)

@router.post("/autopilot")
async def toggle_autopilot(req: SystemToggleRequest, state: SimulationState = Depends(get_state)):
    return system_service.toggle_autopilot(state, req.enabled)

@router.get("/audit-logs")
async def get_audit_logs(limit: int = 50, skip: int = 0, state: SimulationState = Depends(get_state)):
    return system_service.get_audit_logs(state, limit, skip)
