from fastapi import APIRouter, Depends
from state import get_state, SimulationState
from schema import InfrastructureBlock
from services import maintenance_service
from routers.websockets import broadcast_topology, broadcast_copilot

router = APIRouter(prefix="/api/v1", tags=["MMS", "Simulation Sandbox"])

@router.get("/maintenance/blocks")
async def list_blocks(state: SimulationState = Depends(get_state)):
    return {"blocks": list(state.active_blocks.values()), "count": len(state.active_blocks)}

@router.post("/maintenance/blocks")
async def apply_block(block: InfrastructureBlock, state: SimulationState = Depends(get_state)):
    return await maintenance_service.apply_block(state, block, broadcast_topology, broadcast_copilot)

@router.delete("/maintenance/blocks/{element_id}")
async def remove_block(element_id: str, state: SimulationState = Depends(get_state)):
    return await maintenance_service.remove_block(state, element_id, broadcast_topology, broadcast_copilot)

@router.get("/sandbox/blocks")
async def list_sandbox_blocks(state: SimulationState = Depends(get_state)):
    return {"blocks": list(state.sandbox_blocks.values()), "count": len(state.sandbox_blocks)}

@router.post("/sandbox/blocks")
async def add_sandbox_block(block: InfrastructureBlock, state: SimulationState = Depends(get_state)):
    return maintenance_service.add_sandbox_block(state, block)

@router.delete("/sandbox/blocks/{element_id}")
async def remove_sandbox_block(element_id: str, state: SimulationState = Depends(get_state)):
    return maintenance_service.remove_sandbox_block(state, element_id)
