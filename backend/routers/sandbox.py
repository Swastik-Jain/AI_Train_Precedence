from fastapi import APIRouter, Depends
from state import get_state, SimulationState
from schema import WhatIfScenarioRequest
from services import sandbox_service
from routers.websockets import broadcast_topology, broadcast_copilot

router = APIRouter(prefix="/api/v1", tags=["Simulation Sandbox"])

@router.post("/simulation/analyze")
async def analyze_simulation(req: WhatIfScenarioRequest, state: SimulationState = Depends(get_state)):
    return sandbox_service.analyze_simulation(state, req)

@router.post("/simulation/deploy")
async def deploy_sandbox(payload: dict, state: SimulationState = Depends(get_state)):
    return await sandbox_service.deploy_simulation(payload, state)

@router.get("/impact-analysis")
async def get_impact_analysis(state: SimulationState = Depends(get_state)):
    return sandbox_service.get_impact_analysis(state)
