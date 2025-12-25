from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from .session import Base

class AuditLog(Base):
    __tablename__ = "audits"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String, default=lambda: datetime.utcnow().isoformat())
    resume_hash = Column(String, index=True)
    score = Column(Float)
    decision = Column(String)
    model_version = Column(String)
    filename = Column(String)
