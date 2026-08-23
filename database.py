"""SQLAlchemy models and short-lived database session helpers."""
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator
import uuid

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, JSON,
    String, Text, UniqueConstraint, create_engine, inspect, text
)
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

from config import get_settings

settings = get_settings()


def normalize_database_url(url: str) -> str:
    """Use Psycopg v3 for both conventional and explicit PostgreSQL URLs."""
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url.removeprefix("postgres://")
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url.removeprefix("postgresql://")
    return url


database_url = normalize_database_url(settings.database_url)
is_sqlite = database_url.startswith("sqlite")
if is_sqlite and database_url.startswith("sqlite:///"):
    Path(database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)

engine_kwargs = {"pool_pre_ping": True, "echo": False}
if is_sqlite:
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    engine_kwargs.update(pool_size=10, max_overflow=20)
engine = create_engine(database_url, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)
Base = declarative_base()


class Lead(Base):
    __tablename__ = "leads"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    reddit_username = Column(String(50), nullable=False, index=True)
    reddit_post_id = Column(String(20), nullable=False, index=True)
    reddit_comment_id = Column(String(20), nullable=True)
    subreddit = Column(String(50), nullable=False, index=True)
    intent_score = Column(Float, default=0.0)
    qualification_score = Column(Float, default=0.0)
    lead_temperature = Column(String(20), default="cold")
    accident_type = Column(String(100), nullable=True)
    injury_description = Column(Text, nullable=True)
    location_hint = Column(String(200), nullable=True)
    has_attorney = Column(Boolean, nullable=True)
    fault_indicated = Column(String(50), nullable=True)
    contact_volunteered = Column(Boolean, default=False)
    contact_method = Column(String(20), nullable=True)
    contact_value = Column(Text, nullable=True)
    status = Column(String(50), default="new")
    assigned_to = Column(String(100), nullable=True)
    notes = Column(Text, nullable=True)
    source_url = Column(String(500), nullable=False)
    discovered_at = Column(DateTime, default=datetime.utcnow)
    last_engaged_at = Column(DateTime, nullable=True)
    converted_at = Column(DateTime, nullable=True)
    engagements = relationship("Engagement", back_populates="lead", cascade="all, delete-orphan")
    __table_args__ = (
        UniqueConstraint("reddit_post_id", name="uq_leads_reddit_post"),
        Index("idx_leads_score", "intent_score", "qualification_score"),
        Index("idx_leads_status", "status", "discovered_at"),
    )


class Engagement(Base):
    __tablename__ = "engagements"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    lead_id = Column(String(36), ForeignKey("leads.id"), nullable=False)
    engagement_type = Column(String(50), nullable=False)
    reddit_post_id = Column(String(20), nullable=False)
    reddit_comment_id = Column(String(20), nullable=True)
    subreddit = Column(String(50), nullable=False)
    original_post_title = Column(Text, nullable=True)
    original_post_body = Column(Text, nullable=True)
    our_response = Column(Text, nullable=False)
    response_sentiment = Column(String(20), nullable=True)
    compliance_check_passed = Column(Boolean, default=False)
    compliance_flags = Column(JSON, default=list)
    safety_score = Column(Float, default=0.0)
    outbound_status = Column(String(20), default="queued", nullable=False)
    idempotency_key = Column(String(120), nullable=False)
    upvotes_received = Column(Integer, default=0)
    replies_received = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    reddit_created_utc = Column(DateTime, nullable=True)
    lead = relationship("Lead", back_populates="engagements")
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_engagement_idempotency"),
        Index("idx_engagements_lead", "lead_id", "created_at"),
        Index("idx_engagements_subreddit", "subreddit", "engagement_type"),
    )


class MonitoredPost(Base):
    __tablename__ = "monitored_posts"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    reddit_post_id = Column(String(20), nullable=False, unique=True, index=True)
    subreddit = Column(String(50), nullable=False, index=True)
    post_title = Column(Text, nullable=False)
    post_author = Column(String(50), nullable=True)
    monitoring_reason = Column(String(200), nullable=True)
    priority = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    last_checked_at = Column(DateTime, nullable=True)
    reply_count_at_start = Column(Integer, default=0)
    current_reply_count = Column(Integer, default=0)
    processed_reply_ids = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)


class RedditBundle(Base):
    __tablename__ = "reddit_bundles"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    reddit_id = Column(String(20), nullable=False, unique=True, index=True)
    subreddit = Column(String(50), nullable=False, index=True)
    content_type = Column(String(20), nullable=False)
    title = Column(Text, nullable=True)
    body = Column(Text, nullable=False)
    author = Column(String(50), nullable=True)
    embedding_model = Column(String(100), nullable=True)
    embedding_json = Column(Text, nullable=True)
    token_count = Column(Integer, nullable=True)
    score = Column(Integer, default=0)
    num_comments = Column(Integer, default=0)
    intent_tags = Column(JSON, default=list)
    location_tags = Column(JSON, default=list)
    accident_type_tags = Column(JSON, default=list)
    reddit_created_utc = Column(DateTime, nullable=True)
    indexed_at = Column(DateTime, default=datetime.utcnow)
    last_retrieved_at = Column(DateTime, nullable=True)
    __table_args__ = (Index("idx_bundles_subreddit", "subreddit", "content_type"),)


class AgentRun(Base):
    __tablename__ = "agent_runs"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_name = Column(String(100), nullable=False, index=True)
    run_type = Column(String(50), nullable=False)
    status = Column(String(50), default="running")
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    items_processed = Column(Integer, default=0)
    items_created = Column(Integer, default=0)
    errors_count = Column(Integer, default=0)
    log_output = Column(Text, nullable=True)
    error_details = Column(Text, nullable=True)
    __table_args__ = (Index("idx_agent_runs_time", "agent_name", "started_at"),)


class SafetyLog(Base):
    __tablename__ = "safety_logs"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    event_type = Column(String(100), nullable=False, index=True)
    severity = Column(String(20), default="info")
    description = Column(Text, nullable=False)
    reddit_post_id = Column(String(20), nullable=True)
    subreddit = Column(String(50), nullable=True)
    triggered_by = Column(String(100), nullable=True)
    resolution = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def get_db() -> Iterator[Session]:
    """FastAPI dependency that always rolls back failed requests and closes."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def db_session() -> Iterator[Session]:
    """Context manager for background work."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    # Older deployments created before these columns existed. Keep startup
    # compatible with them; schema migrations should still be run formally.
    if is_sqlite:
        with engine.begin() as connection:
            columns = {row[1] for row in connection.execute(text("PRAGMA table_info(engagements)"))}
            if "outbound_status" not in columns:
                connection.execute(text("ALTER TABLE engagements ADD COLUMN outbound_status VARCHAR(20) DEFAULT 'queued' NOT NULL"))
            if "idempotency_key" not in columns:
                connection.execute(text("ALTER TABLE engagements ADD COLUMN idempotency_key VARCHAR(120)"))
                connection.execute(text("UPDATE engagements SET idempotency_key = id WHERE idempotency_key IS NULL"))
            columns = {row[1] for row in connection.execute(text("PRAGMA table_info(reddit_bundles)"))}
            if "embedding_json" not in columns:
                connection.execute(text("ALTER TABLE reddit_bundles ADD COLUMN embedding_json TEXT"))
            columns = {row[1] for row in connection.execute(text("PRAGMA table_info(monitored_posts)"))}
            if "processed_reply_ids" not in columns:
                connection.execute(text("ALTER TABLE monitored_posts ADD COLUMN processed_reply_ids JSON"))


def apply_schema_migrations() -> None:
    """Apply the lightweight, versioned compatibility changes before rollout.

    This command is deliberately separate from web/worker startup so a
    production deployment has one schema owner rather than racing two services.
    """
    init_db()
    expected = {
        "engagements": {
            "outbound_status": "VARCHAR(20) NOT NULL DEFAULT 'queued'",
            "idempotency_key": "VARCHAR(120)",
        },
        "reddit_bundles": {"embedding_json": "TEXT"},
        "monitored_posts": {"processed_reply_ids": "JSON"},
    }
    inspector = inspect(engine)
    with engine.begin() as connection:
        for table, columns in expected.items():
            existing = {column["name"] for column in inspector.get_columns(table)}
            for column, definition in columns.items():
                if column not in existing:
                    connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {definition}"))
        if is_sqlite:
            connection.execute(text("UPDATE engagements SET idempotency_key = id WHERE idempotency_key IS NULL"))
        connection.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_engagement_idempotency_index "
            "ON engagements (idempotency_key)"
        ))


def get_db_session() -> Session:
    """Compatibility factory; callers must close or use db_session()."""
    return SessionLocal()