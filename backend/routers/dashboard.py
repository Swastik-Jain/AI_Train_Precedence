from fastapi import APIRouter, Depends
from state import get_state, SimulationState
from services import dashboard_service

router = APIRouter(tags=["Telemetry", "Dashboard"])

@router.get("/api/v1/telemetry")
async def get_telemetry(state: SimulationState = Depends(get_state)):
    return dashboard_service.get_telemetry(state)

@router.get("/api/v1/meta")
async def get_meta():
    return {"version": "2.0", "author": "Swastik (MITS)"}

@router.post("/api/v1/telemetry")
async def post_telemetry(req: dict):
    print(f"FRONTLOG: {req}")
    return {"status": "ok"}
