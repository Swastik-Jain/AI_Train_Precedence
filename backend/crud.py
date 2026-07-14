from sqlalchemy.orm import Session
import uuid as _uuid
from database import TrainPosition, AuditLog, FleetTrain
import json


def getRecentTrainLog(db: Session, limit: int = 50):
    """Return most-recent train position records (timestamp column now exists)."""
    return (
        db.query(TrainPosition)
        .order_by(TrainPosition.timestamp.desc())
        .limit(limit)
        .all()
    )


def create_audit_log(db: Session, entry: dict) -> AuditLog:
    """Persist one audit-log entry to SQLite so it survives restarts.

    ``entry`` must contain the same keys written to the in-memory AUDIT_LOGS list:
    id, t (ISO-8601 timestamp), timestamp (epoch ms), source, action, operator,
    status, statusType.
    """
    log = AuditLog(
        log_id      = entry.get("id") or str(_uuid.uuid4()),
        timestamp   = entry.get("t", ""),
        timestamp_ms= int(entry.get("timestamp", 0)),
        source      = entry.get("source", "SYSTEM"),
        action      = entry.get("action", ""),
        operator    = entry.get("operator", "SYSTEM"),
        status      = entry.get("status", ""),
        status_type = entry.get("statusType", "info"),
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def get_recent_audit_logs(db: Session, limit: int = 200, skip: int = 0):
    """Return most-recent audit log entries from the DB."""
    return (
        db.query(AuditLog)
        .order_by(AuditLog.timestamp_ms.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

def load_all_fleet_trains(db: Session):
    return db.query(FleetTrain).all()

def add_fleet_train(db: Session, cfg: dict):
    train = FleetTrain(
        train_id=cfg["train_id"],
        train_type=cfg.get("train_type", ""),
        max_speed=cfg.get("max_speed", 100),
        priority=cfg.get("priority", 5),
        start_time=cfg.get("start_time", 0),
        deadline=cfg.get("deadline", 120),
        direction=cfg.get("direction", "DOWN"),
        path=json.dumps(cfg.get("path", [])),
        added_at=cfg.get("added_at", "")
    )
    db.add(train)
    db.commit()
    db.refresh(train)
    return train

def remove_fleet_train(db: Session, train_id: str):
    train = db.query(FleetTrain).filter(FleetTrain.train_id == train_id).first()
    if train:
        db.delete(train)
        db.commit()
    return train
