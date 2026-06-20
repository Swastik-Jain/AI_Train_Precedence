# backend/database.py
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base

# 1. Database Configuration
# We use a hardcoded SQLite path to avoid "None" port errors.
DATABASE_URL = "sqlite:///./train_simulation.db"

# 2. Create the Engine
# check_same_thread=False is strictly required for SQLite + FastAPI
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# 3. Create a Session Factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. Define the Base Class
Base = declarative_base()

# 5. Define the TrainPosition Table
class TrainPosition(Base):
    __tablename__ = "train_positions"

    id        = Column(Integer, primary_key=True, index=True)
    train_id  = Column(String, index=True)  # e.g., "T100"
    section   = Column(String)              # e.g., "5", "104"
    speed_kmh = Column(Integer)             # e.g., 80
    status    = Column(String)              # e.g., "RUNNING", "LATE"  (was "Status" — fixed)
    timestamp = Column(String, index=True)  # ISO-8601 string — was missing, fixed

# 6. Audit Log Table — persists operator decisions across restarts
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id          = Column(Integer, primary_key=True, index=True)
    log_id      = Column(String, unique=True, index=True)  # uuid from in-memory entry
    timestamp   = Column(String, index=True)               # ISO-8601 string
    timestamp_ms= Column(Integer)                          # epoch ms for fast sort
    source      = Column(String)                           # e.g. "SYSTEM_CONTROL", "OR-SHIELD"
    action      = Column(Text)                             # human-readable action description
    operator    = Column(String)                           # e.g. "Dispatcher", "SYSTEM"
    status      = Column(String)                           # e.g. "Committed", "Rejected"
    status_type = Column(String)                           # e.g. "success", "warning", "error"

# 7. Helper to create tables
def init_db():
    Base.metadata.create_all(bind=engine)