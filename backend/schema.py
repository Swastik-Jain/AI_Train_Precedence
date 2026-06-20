from pydantic import BaseModel
from datetime import datetime
from typing import Optional


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

