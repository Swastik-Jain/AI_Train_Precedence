import asyncio
import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Database initialization
import database

# Global configuration / State
import config
from state import get_state

# Services for background tasks
from services.simulation_service import simulate_trains_bg, copilot_suggestion_bg
from routers.websockets import broadcast_topology, broadcast_copilot
from services.maintenance_service import sync_blocks_to_rl_env
from services.system_service import push_audit_log
from services.copilot_service import _write_feedback, _make_suggestion
from services.simulation_service import _get_sim_brain

# Routers
from routers import fleet, maintenance, system, copilot, dashboard, websockets, sandbox

# Load environment variables
load_dotenv()
CORS_ALLOWED_ORIGINS_ENV = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000")
CORS_ALLOWED_ORIGINS = [origin.strip() for origin in CORS_ALLOWED_ORIGINS_ENV.split(",") if origin.strip()]

# Create database tables
database.Base.metadata.create_all(bind=database.engine)

app = FastAPI(
    title="ORBIT API",
    description="Operational Rail Backbone for AI Traffic simulation and Co-pilot AI features.",
    version="2.0",
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(websockets.router)
app.include_router(dashboard.router)
app.include_router(fleet.router)
app.include_router(system.router)
app.include_router(copilot.router)
app.include_router(maintenance.router)
app.include_router(sandbox.router)


@app.on_event("startup")
async def startup_event():
    # Pass dependencies to background tasks
    state = get_state()
    asyncio.create_task(simulate_trains_bg(state, broadcast_topology, broadcast_copilot, sync_blocks_to_rl_env, push_audit_log))
    asyncio.create_task(copilot_suggestion_bg(state, broadcast_copilot, _write_feedback, _make_suggestion))
