from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from ..core.config import get_settings
import os

settings = get_settings()

# Determine database URL - fallback to SQLite if not set or empty
database_url = settings.DATABASE_URL
if not database_url or database_url.startswith("mysql") and "db:3306" in database_url:
    # Use SQLite for local development
    db_path = os.path.join(settings.BASE_DIR, "audit.db")
    database_url = f"sqlite:///{db_path}"

# Use check_same_thread=False for SQLite if needed
connect_args = {"check_same_thread": False} if "sqlite" in database_url else {}

engine = create_engine(
    database_url, 
    connect_args=connect_args,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

