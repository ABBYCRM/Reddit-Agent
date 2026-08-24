"""Operator-only FastAPI dashboard and readiness endpoints."""
import asyncio
import logging
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

from agent_orchestrator import get_orchestrator
from config import get_settings
from database import AgentRun, Engagement, Lead, engine, get_db, init_db
from outreach_service import OutreachUnavailableError, get_outreach_service
from rag_engine import get_rag_engine
from reddit_client import get_reddit_client
from safety_guardrails import get_guardrails

logger = logging.getLogger(__name__)
settings = get_settings()
BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="CaseClosedFL Agent Control Center", docs_url=None, redoc_url=None)
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.secret_key,
    https_only=settings.app_env.lower() == "production",
    same_site="strict",
)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def require_operator(request: Request) -> None:
    expected = settings.operator_api_key
    provided = request.headers.get("X-Operator-Key", "")
    if expected and secrets.compare_digest(provided, expected):
        request.state.operator_via_header = True
        request.session["operator_authenticated"] = True
        request.session.setdefault("csrf_token", secrets.token_urlsafe(32))
        return
    if request.session.get("operator_authenticated") is True:
        request.state.operator_via_header = False
        return
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Operator authentication required")


def require_csrf(request: Request) -> None:
    """Session-authenticated browser actions require a same-session CSRF token."""
    if getattr(request.state, "operator_via_header", False):
        return
    expected = request.session.get("csrf_token", "")
    provided = request.headers.get("X-CSRF-Token", "")
    if not expected or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF validation failed")


class OutreachUpdate(BaseModel):
    enabled: bool


@app.on_event("startup")
async def startup():
    init_db()
    await get_rag_engine().initialize_kb()
    logger.info("API started in %s mode; scheduler is owned by the worker process.", settings.app_env)


@app.get("/access", response_class=HTMLResponse)
async def operator_access(request: Request):
    if request.session.get("operator_authenticated"):
        return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse("operator_access.html", {"request": request})


@app.post("/api/operator-session")
async def create_operator_session(request: Request):
    expected = settings.operator_api_key
    provided = request.headers.get("X-Operator-Key", "")
    if not expected or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Operator authentication required")
    request.session["operator_authenticated"] = True
    request.session["csrf_token"] = secrets.token_urlsafe(32)
    return {"status": "authenticated"}


@app.post("/api/operator-session/logout", dependencies=[Depends(require_operator), Depends(require_csrf)])
async def destroy_operator_session(request: Request):
    request.session.clear()
    return {"status": "signed_out"}


@app.get("/favicon.ico", include_in_schema=False, status_code=status.HTTP_204_NO_CONTENT)
async def favicon():
    return None


@app.get("/", response_class=HTMLResponse, dependencies=[Depends(require_operator)])
async def dashboard(request: Request, db: Session = Depends(get_db)):
    recent_runs = db.query(AgentRun).order_by(AgentRun.started_at.desc()).limit(5).all()
    outreach = await get_outreach_service().get_status()
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
        "settings": {
            "max_daily": outreach["daily_limit"],
            "model": settings.nvidia_model,
        },
        "outreach": outreach,
        "csrf_token": request.session.get("csrf_token", ""),
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


@app.post("/api/trigger-discovery", dependencies=[Depends(require_operator), Depends(require_csrf)])
async def trigger_discovery(background_tasks: BackgroundTasks):
    background_tasks.add_task(get_orchestrator().run_full_cycle)
    return {"status": "accepted", "message": "Discovery and drafting cycle queued."}


@app.get("/api/outreach", dependencies=[Depends(require_operator)])
async def outreach_status():
    return await get_outreach_service().get_status()


@app.put("/api/outreach", dependencies=[Depends(require_operator), Depends(require_csrf)])
async def update_outreach(payload: OutreachUpdate, request: Request):
    try:
        result = await get_outreach_service().set_enabled(payload.enabled, operator="operator")
    except OutreachUnavailableError as exc:
        readiness = await get_outreach_service().readiness()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Autonomous outreach cannot be enabled: {readiness['message']}",
        ) from exc
    state = "enabled" if result["enabled"] else "disabled"
    return {**result, "message": f"Autonomous outreach {state}."}


@app.post("/api/toggle-auto-reply", dependencies=[Depends(require_operator), Depends(require_csrf)])
async def toggle_auto_reply():
    """Compatibility endpoint for existing operator clients."""
    current = await get_outreach_service().get_status()
    try:
        result = await get_outreach_service().set_enabled(not current["enabled"], operator="operator")
    except OutreachUnavailableError as exc:
        readiness = await get_outreach_service().readiness()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=readiness["message"]) from exc
    return {**result, "message": f"Autonomous outreach {'enabled' if result['enabled'] else 'disabled'}."}


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