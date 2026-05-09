# backend/database.py
from sqlalchemy import create_engine, Column, Integer, String
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

    id = Column(Integer, primary_key=True, index=True)
    train_id = Column(String, index=True) # e.g., "T100"
    section = Column(String)              # e.g., "5", "104"
    speed_kmh = Column(Integer)           # e.g., 80
    status = Column(String)               # e.g., "RUNNING", "LATE"

# 6. Helper to create tables (Run this once)
def init_db():
    Base.metadata.create_all(bind=engine)