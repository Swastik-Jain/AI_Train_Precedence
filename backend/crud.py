from sqlalchemy.orm import Session
from database import TrainPosition

def getRecentTrainLog(db: Session, limit: int = 50):
    return db.query(TrainPosition).order_by(TrainPosition.timestamp.desc()).limit(limit).all()
