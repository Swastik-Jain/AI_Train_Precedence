from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Literal, Dict


class TrainPositionBase(BaseModel):
    train_id: str
    speed_kmh: int
    status: str           # was "Status" — fixed to match SQLAlchemy model
    section: Optional[str] = None


class TrainPositionCreate(TrainPositionBase):
    pass


class TrainPosition(TrainPositionBase):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True


class AuditLogSchema(BaseModel):
    log_id: str
    timestamp: str
    timestamp_ms: int
    source: str
    action: str
    operator: str
    status: str
    status_type: str

    class Config:
        from_attributes = True


class RejectRequest(BaseModel):
    recommendation_id: str
    reason: Optional[str] = "controller_dismissed"

class InfrastructureBlock(BaseModel):
    element_id: str                                    # e.g. "edge-0-1"
    type: Literal["PLATFORM", "TRACK_SEGMENT"]         # block type
    start_time: str                                    # ISO-8601
    end_time: str                                      # ISO-8601
    severity: Literal["TOTAL_BLOCK", "SPEED_RESTRICTION"]
    reason: Optional[str] = "Scheduled maintenance"
    speed_limit: Optional[int] = 30                   # km/h — only used for SPEED_RESTRICTION

class NewTrainRequest(BaseModel):
    train_id: str
    train_type: str = "Express"           # must be one of TRAIN_TYPES
    max_speed: int  = 110                 # km/h
    priority: Optional[int] = None        # auto-derived from type if omitted
    start_time: int = 0                   # minutes from session start
    deadline: int   = 120                 # minutes from session start
    direction: int  = 1                   # 1 = forward

class SimSpeedRequest(BaseModel):
    factor: float

class SystemToggleRequest(BaseModel):
    enabled: bool

class CopilotOverrideRequest(BaseModel):
    recommendation_id: str
    new_action: Optional[int] = None
    new_edge: Optional[str] = None

class ForceActionRequest(BaseModel):
    train_id: str
    action: int
    duration_ticks: int = 50

class AcknowledgeRequest(BaseModel):
    recommendation_id: str
    reason: Optional[str] = "controller_dismissed"

class WhatIfScenarioRequest(BaseModel):
    label: Optional[str] = "Scenario"
    latencies: Optional[Dict[str, int]] = {}        # train_id -> delay in minutes
    forced_actions: Optional[Dict[str, int]] = {}   # train_id -> 0=HOLD, 1=MAIN, 2=DIVERT
