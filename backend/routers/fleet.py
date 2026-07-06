from fastapi import APIRouter, Depends
from typing import Dict, Any

from state import get_state, SimulationState
from schema import NewTrainRequest
from services import fleet_service
from routers.websockets import broadcast_topology

router = APIRouter(prefix="/api/v1/fleet", tags=["Fleet"])

@router.get("")
async def get_fleet(state: SimulationState = Depends(get_state)):
    return fleet_service.get_fleet_live(state)

@router.post("")
async def add_train(req: NewTrainRequest, state: SimulationState = Depends(get_state)):
    return await fleet_service.add_train(state, req, broadcast_topology)

@router.delete("/{train_id}")
async def remove_train(train_id: str, state: SimulationState = Depends(get_state)):
    return fleet_service.remove_train(state, train_id)

@router.get("/schedule")
async def get_schedule(state: SimulationState = Depends(get_state)):
    return fleet_service.get_schedule(state)

@router.post("/generate-schedule")
async def generate_schedule(state: SimulationState = Depends(get_state)):
    return fleet_service.generate_schedule(state)
