"""Operator-only FastAPI dashboard and readiness endpoints."""
import asyncio
import logging
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import text
from sqlalchemy.orm import Session

from agent_orchestrator import get_orchestrator
from config import get_settings
from database import AgentRun, Engagement, Lead, engine, get_db, init_db
from rag_engine import get_rag_engine
from reddit_client import get_reddit_client
from safety_guardrails import get_guardrails

logger = logging.getLogger(__name__)
settings = get_settings()
BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="CaseClosedFL Agent Control Center", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def require_operator(request: Request) -> None:
    expected = settings.operator_api_key
    provided = request.headers.get("X-Operator-Key", "")
    if not expected or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Operator authentication required")


@app.on_event("startup")
async def startup():
    init_db()
    await get_rag_engine().initialize_kb()
    logger.info("API started in %s mode; scheduler is owned by the worker process.", settings.app_env)


@app.get("/", response_class=HTMLResponse, dependencies=[Depends(require_operator)])
async def dashboard(request: Request, db: Session = Depends(get_db)):
    recent_runs = db.query(AgentRun).order_by(AgentRun.started_at.desc()).limit(5).all()
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "stats": {
            "total_leads": db.query(Lead).count(),
            "qualified_leads": db.query(Lead).filter(Lead.status == "qualified").count(),
            "new_leads": db.query(Lead).filter(Lead.status == "new").count(),
            "total_engagements": db.query(Engagement).count(),
            "reddit": get_reddit_client().get_stats(),
            "safety": get_guardrails().get_stats(),
            "rag": get_rag_engine().get_stats(),
            "scheduler": {"running": False, "owner": "worker"},
        },
        "recent_runs": recent_runs,
        "settings": {"auto_reply": settings.enable_auto_reply, "dm_outreach": settings.enable_dm_outreach,
                     "max_daily": settings.max_daily_engagements, "model": settings.nvidia_model},
    })


@app.get("/api/leads", dependencies=[Depends(require_operator)])
async def get_leads(status: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(Lead)
    if status:
        query = query.filter(Lead.status == status)
    return {"leads": [{
        "id": lead.id, "reddit_username": lead.reddit_username, "subreddit": lead.subreddit,
        "intent_score": lead.intent_score, "status": lead.status, "temperature": lead.lead_temperature,
        "discovered_at": lead.discovered_at.isoformat(), "source_url": lead.source_url,
    } for lead in query.order_by(Lead.discovered_at.desc()).limit(100).all()]}


@app.get("/api/engagements", dependencies=[Depends(require_operator)])
async def get_engagements(db: Session = Depends(get_db)):
    return {"engagements": [{
        "id": item.id, "subreddit": item.subreddit, "type": item.engagement_type,
        "safety_score": item.safety_score, "compliance_passed": item.compliance_check_passed,
        "outbound_status": item.outbound_status, "created_at": item.created_at.isoformat(),
    } for item in db.query(Engagement).order_by(Engagement.created_at.desc()).limit(100).all()]}


@app.post("/api/trigger-discovery", dependencies=[Depends(require_operator)])
async def trigger_discovery(background_tasks: BackgroundTasks):
    background_tasks.add_task(get_orchestrator().run_full_cycle)
    return {"status": "accepted", "message": "Discovery and drafting cycle queued; outbound replies remain opt-in."}


@app.post("/api/toggle-auto-reply", dependencies=[Depends(require_operator)])
async def toggle_auto_reply():
    raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                        detail="Automation cannot be toggled at runtime. Update reviewed deployment configuration and restart.")


@app.get("/api/health")
async def health_check():
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), "version": "1.1.0", "scheduler_owner": "worker"}
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {**payload, "status": "healthy"}
    except Exception as exc:
        logger.error("Readiness check failed: %s", exc)
        return JSONResponse(status_code=503, content={**payload, "status": "unhealthy", "reason": "database_unavailable"})