"""
CaseClosedFL Reddit Agent - Database Layer
SQLite/PostgreSQL compatible with SQLAlchemy 2.0
"""
from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, 
    Float, Boolean, ForeignKey, JSON, Index, func
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
import uuid

from config import get_settings

settings = get_settings()

# Detect database type
is_sqlite = settings.database_url.startswith("sqlite")

# Engine with connection pooling
engine = create_engine(
    settings.database_url,
    pool_size=10 if not is_sqlite else 0,
    max_overflow=20 if not is_sqlite else 0,
    pool_pre_ping=True,
    echo=False,
    connect_args={"check_same_thread": False} if is_sqlite else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ─── Models ──────────────────────────────────────────────────────────────

class Lead(Base):
    """Qualified lead extracted from Reddit engagement."""
    __tablename__ = "leads"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    reddit_username = Column(String(50), nullable=False, index=True)
    reddit_post_id = Column(String(20), nullable=False, index=True)
    reddit_comment_id = Column(String(20), nullable=True)
    subreddit = Column(String(50), nullable=False, index=True)

    # Lead scoring
    intent_score = Column(Float, default=0.0)
    qualification_score = Column(Float, default=0.0)
    lead_temperature = Column(String(20), default="cold")

    # Extracted info
    accident_type = Column(String(100), nullable=True)
    injury_description = Column(Text, nullable=True)
    location_hint = Column(String(200), nullable=True)
    has_attorney = Column(Boolean, nullable=True)
    fault_indicated = Column(String(50), nullable=True)

    # Contact (only collected if user volunteers)
    contact_volunteered = Column(Boolean, default=False)
    contact_method = Column(String(20), nullable=True)
    contact_value = Column(String(500), nullable=True)

    # Status workflow
    status = Column(String(50), default="new")
    assigned_to = Column(String(100), nullable=True)
    notes = Column(Text, nullable=True)

    # Metadata
    source_url = Column(String(500), nullable=False)
    discovered_at = Column(DateTime, default=datetime.utcnow)
    last_engaged_at = Column(DateTime, nullable=True)
    converted_at = Column(DateTime, nullable=True)

    engagements = relationship("Engagement", back_populates="lead", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_leads_score", "intent_score", "qualification_score"),
        Index("idx_leads_status", "status", "discovered_at"),
    )


class Engagement(Base):
    """Every interaction with a Reddit user."""
    __tablename__ = "engagements"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    lead_id = Column(String(36), ForeignKey("leads.id"), nullable=False)

    engagement_type = Column(String(50), nullable=False)
    reddit_post_id = Column(String(20), nullable=False)
    reddit_comment_id = Column(String(20), nullable=True)
    subreddit = Column(String(50), nullable=False)

    # Content
    original_post_title = Column(Text, nullable=True)
    original_post_body = Column(Text, nullable=True)
    our_response = Column(Text, nullable=False)
    response_sentiment = Column(String(20), nullable=True)

    # Safety / Compliance
    compliance_check_passed = Column(Boolean, default=False)
    compliance_flags = Column(JSON, default=list)
    safety_score = Column(Float, default=0.0)

    # Metrics
    upvotes_received = Column(Integer, default=0)
    replies_received = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    reddit_created_utc = Column(DateTime, nullable=True)

    lead = relationship("Lead", back_populates="engagements")

    __table_args__ = (
        Index("idx_engagements_lead", "lead_id", "created_at"),
        Index("idx_engagements_subreddit", "subreddit", "engagement_type"),
    )


class MonitoredPost(Base):
    """Posts we are actively watching for replies."""
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

    created_at = Column(DateTime, default=datetime.utcnow)


class RedditBundle(Base):
    """RAG storage for Reddit posts/comments indexed for retrieval."""
    __tablename__ = "reddit_bundles"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    reddit_id = Column(String(20), nullable=False, unique=True, index=True)
    subreddit = Column(String(50), nullable=False, index=True)
    content_type = Column(String(20), nullable=False)
    title = Column(Text, nullable=True)
    body = Column(Text, nullable=False)
    author = Column(String(50), nullable=True)

    embedding_model = Column(String(100), nullable=True)
    token_count = Column(Integer, nullable=True)

    score = Column(Integer, default=0)
    num_comments = Column(Integer, default=0)

    intent_tags = Column(JSON, default=list)
    location_tags = Column(JSON, default=list)
    accident_type_tags = Column(JSON, default=list)

    reddit_created_utc = Column(DateTime, nullable=True)
    indexed_at = Column(DateTime, default=datetime.utcnow)
    last_retrieved_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_bundles_subreddit", "subreddit", "content_type"),
    )


class AgentRun(Base):
    """Audit log for every agent execution cycle."""
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

    __table_args__ = (
        Index("idx_agent_runs_time", "agent_name", "started_at"),
    )


class SafetyLog(Base):
    """Compliance and safety event logging."""
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


# ─── Database Utilities ─────────────────────────────────────────────────

def get_db():
    """Dependency for FastAPI to get DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables. Call once at startup."""
    Base.metadata.create_all(bind=engine)


def get_db_session() -> Session:
    """Get a database session for background tasks."""
    return SessionLocal()
