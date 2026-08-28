"""
CaseClosedFL Reddit Agent - FastAPI Web UI
Operator-only control dashboard for monitoring and managing the agent.
The scheduler runs in the separate worker process (python run.py).
"""
import logging
import secrets
from contextlib import asynccontextmanager
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, Request, Depends, BackgroundTasks, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from config import get_settings
from database import get_db, init_db, Lead, Engagement, AgentRun
from reddit_client import get_reddit_client
from rag_engine import get_rag_engine
from safety_guardrails import get_guardrails

logger = logging.getLogger(__name__)
settings = get_settings()


def require_operator(request: Request) -> None:
    """Constant-time check of the operator API key on protected routes."""
    expected = settings.operator_api_key
    provided = request.headers.get("X-Operator-Key", "")
    if not expected or not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Operator authentication required",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and services on startup."""
    init_db()
    await get_rag_engine().initialize_kb()
    logger.info("Agent API started; scheduler is owned by the worker process")
    yield


# Initialize FastAPI
app = FastAPI(title="CaseClosedFL Agent Control Center", lifespan=lifespan)

# Templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Scheduler runs in the worker process; the web surface only reports status.
scheduler = None


@app.get("/", response_class=HTMLResponse, dependencies=[Depends(require_operator)])
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

    # Scheduler runs in the worker process; report ownership only.
    sched_status = {"running": "worker process", "jobs": []}

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


@app.get("/api/leads", dependencies=[Depends(require_operator)])
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


@app.get("/api/engagements", dependencies=[Depends(require_operator)])
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


@app.post("/api/trigger-discovery", dependencies=[Depends(require_operator)])
async def trigger_discovery(background_tasks: BackgroundTasks):
    """Manually trigger a discovery cycle in the background."""
    from discovery_agent import DiscoveryAgent

    async def _run_safe():
        try:
            await DiscoveryAgent().run()
        except Exception:
            logger.exception("Manual discovery cycle failed")

    background_tasks.add_task(_run_safe)
    return {"status": "triggered", "message": "Discovery cycle started"}


@app.post("/api/toggle-auto-reply", dependencies=[Depends(require_operator)])
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
