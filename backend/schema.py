from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class TrainPositionBase(BaseModel):
    train_id: str
    speed_kmh: int
    Status: str
    section: Optional[str] = None

class TrainPositionCreate(TrainPositionBase):
    pass

class TrainPosition(TrainPositionBase):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True
