"""
CaseClosedFL Reddit Agent - FastAPI Web UI
Control dashboard for monitoring and managing the agent.
"""
import logging
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, Request, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from config import get_settings
from database import get_db, init_db, Lead, Engagement, AgentRun, RedditBundle
from scheduler import AgentScheduler
from agent_orchestrator import get_orchestrator
from reddit_client import get_reddit_client
from rag_engine import get_rag_engine
from safety_guardrails import get_guardrails

logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize FastAPI
app = FastAPI(title="CaseClosedFL Agent Control Center")

# Templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global scheduler instance
scheduler: Optional[AgentScheduler] = None


@app.on_event("startup")
async def startup():
    """Initialize database and services on startup."""
    init_db()
    global scheduler
    scheduler = AgentScheduler()
    scheduler.start()
    logger.info("Agent API started")


@app.on_event("shutdown")
async def shutdown():
    """Graceful shutdown."""
    if scheduler:
        scheduler.shutdown()


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    """Main control dashboard."""
    # Stats
    total_leads = db.query(Lead).count()
    qualified_leads = db.query(Lead).filter(Lead.status == "qualified").count()
    new_leads = db.query(Lead).filter(Lead.status == "new").count()
    total_engagements = db.query(Engagement).count()
    recent_runs = db.query(AgentRun).order_by(AgentRun.started_at.desc()).limit(5).all()

    # Reddit stats
    reddit_stats = get_reddit_client().get_stats()

    # Safety stats
    safety_stats = get_guardrails().get_stats()

    # RAG stats
    rag_stats = get_rag_engine().get_stats()

    # Scheduler status
    sched_status = scheduler.get_status() if scheduler else {}

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "stats": {
            "total_leads": total_leads,
            "qualified_leads": qualified_leads,
            "new_leads": new_leads,
            "total_engagements": total_engagements,
            "reddit": reddit_stats,
            "safety": safety_stats,
            "rag": rag_stats,
            "scheduler": sched_status
        },
        "recent_runs": recent_runs,
        "settings": {
            "auto_reply": settings.enable_auto_reply,
            "dm_outreach": settings.enable_dm_outreach,
            "max_daily": settings.max_daily_engagements,
            "model": settings.nvidia_model
        }
    })


@app.get("/api/leads")
async def get_leads(status: Optional[str] = None, db: Session = Depends(get_db)):
    """API endpoint for leads."""
    query = db.query(Lead)
    if status:
        query = query.filter(Lead.status == status)
    leads = query.order_by(Lead.discovered_at.desc()).limit(100).all()

    return {
        "leads": [
            {
                "id": str(l.id),
                "reddit_username": l.reddit_username,
                "subreddit": l.subreddit,
                "intent_score": l.intent_score,
                "status": l.status,
                "temperature": l.lead_temperature,
                "discovered_at": l.discovered_at.isoformat(),
                "source_url": l.source_url
            }
            for l in leads
        ]
    }


@app.get("/api/engagements")
async def get_engagements(db: Session = Depends(get_db)):
    """API endpoint for engagements."""
    engagements = db.query(Engagement).order_by(Engagement.created_at.desc()).limit(100).all()
    return {
        "engagements": [
            {
                "id": str(e.id),
                "subreddit": e.subreddit,
                "type": e.engagement_type,
                "safety_score": e.safety_score,
                "compliance_passed": e.compliance_check_passed,
                "created_at": e.created_at.isoformat()
            }
            for e in engagements
        ]
    }


@app.post("/api/trigger-discovery")
async def trigger_discovery(background_tasks: BackgroundTasks):
    """Manually trigger a discovery cycle."""
    if scheduler:
        background_tasks.add_task(scheduler._run_discovery_cycle)
        return {"status": "triggered", "message": "Discovery cycle started"}
    return {"status": "error", "message": "Scheduler not running"}


@app.post("/api/toggle-auto-reply")
async def toggle_auto_reply():
    """Toggle auto-reply setting."""
    # In production, update database config; here we just return status
    return {
        "status": "success",
        "auto_reply": settings.enable_auto_reply,
        "message": "Auto-reply is " + ("enabled" if settings.enable_auto_reply else "disabled")
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint for DigitalOcean/load balancers."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }
