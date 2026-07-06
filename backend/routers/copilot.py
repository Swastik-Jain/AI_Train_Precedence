from fastapi import APIRouter, Depends
from typing import Optional
from state import get_state, SimulationState
from schema import ForceActionRequest, CopilotOverrideRequest, AcknowledgeRequest
from services import copilot_service
from routers.websockets import broadcast_topology, broadcast_copilot

router = APIRouter(tags=["ORBIT Co-pilot", "System Override"])

@router.post("/api/v1/dispatch/force-action", tags=["System Override"])
async def force_action(req: ForceActionRequest, state: SimulationState = Depends(get_state)):
    return copilot_service.force_action(state, req.train_id, req.action, req.duration_ticks)

@router.post("/api/v1/dispatch/override")
async def override_decision(req: CopilotOverrideRequest, state: SimulationState = Depends(get_state)):
    return await copilot_service.override_decision(state, req, broadcast_topology, broadcast_copilot)

@router.post("/api/v1/dispatch/acknowledge")
async def acknowledge_decision(req: AcknowledgeRequest, state: SimulationState = Depends(get_state)):
    return copilot_service.acknowledge_decision(state, req.recommendation_id)

@router.get("/api/v1/dispatch/suggestions")
async def get_suggestions(status: Optional[str] = None, state: SimulationState = Depends(get_state)):
    return copilot_service.get_suggestions(state, status)

@router.get("/api/v1/copilot/raw-proposal")
async def get_raw_proposal(state: SimulationState = Depends(get_state)):
    return copilot_service.get_raw_proposal(state)
