from datetime import datetime
from ..core.config import get_settings
from ..core.logging import get_logger
from .session import SessionLocal, engine, Base
from .models import AuditLog

settings = get_settings()
logger = get_logger(__name__)

# Create tables
Base.metadata.create_all(bind=engine)

class AuditLogger:
    def log_decision(self, resume_hash: str, score: float, decision: str, filename: str = "Unknown"):
        db = SessionLocal()
        try:
            audit = AuditLog(
                timestamp=datetime.utcnow().isoformat(),
                resume_hash=resume_hash,
                score=score,
                decision=decision,
                model_version=settings.VERSION,
                filename=filename
            )
            db.add(audit)
            db.commit()
            db.refresh(audit)
            logger.info("Audit record saved", extra_data={"resume_hash": resume_hash, "decision": decision})
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
            db.rollback()
        finally:
            db.close()

    def get_history(self, limit: int = 50):
        db = SessionLocal()
        try:
            return db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"Failed to fetch history: {e}")
            return []
        finally:
            db.close()
