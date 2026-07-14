from fastapi import APIRouter, Depends
from typing import Dict, Any
from sqlalchemy.orm import Session

from state import get_state, SimulationState
from database import get_db
from schema import NewTrainRequest
from services import fleet_service
from routers.websockets import broadcast_topology

router = APIRouter(prefix="/api/v1/fleet", tags=["Fleet"])

@router.get("")
async def get_fleet(state: SimulationState = Depends(get_state)):
    return fleet_service.get_fleet_live(state)

@router.post("")
async def add_train(req: NewTrainRequest, state: SimulationState = Depends(get_state), db: Session = Depends(get_db)):
    return await fleet_service.add_train(state, req, broadcast_topology, db)

@router.delete("/{train_id}")
async def remove_train(train_id: str, state: SimulationState = Depends(get_state), db: Session = Depends(get_db)):
    return fleet_service.remove_train(state, train_id, db)

@router.get("/schedule")
async def get_schedule(state: SimulationState = Depends(get_state)):
    return fleet_service.get_schedule(state)

@router.post("/generate-schedule")
async def generate_schedule(state: SimulationState = Depends(get_state)):
    return fleet_service.generate_schedule(state)
