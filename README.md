# CaseClosedFL Reddit Agent

An operator-only, review-first Reddit discovery and intake assistant. It uses
Composio as the default Reddit transport and keeps all automated public replies
and direct messages disabled by default.

## Local run

1. Copy `.env.example` to `.env` and set **new** local values for
   `SECRET_KEY` and `OPERATOR_API_KEY`.
2. Install dependencies: `pip install -r requirements.txt`
3. Start the protected web control surface: `uvicorn api:app --host 0.0.0.0 --port 8000`
4. Start the scheduler separately: `python run.py`

Pass `X-Operator-Key` to access the dashboard or operational APIs. The health
endpoint does not expose lead data and is intentionally unauthenticated.

## Verification

Run `pytest -q` from this directory. Tests use temporary SQLite databases and
mocked provider boundaries; they never perform live Reddit, Composio, or LLM
actions.

## Production

See `deploy/digitalocean.md`. Use encrypted environment variables, a managed
PostgreSQL database, a separate web and worker process, and leave
`ENABLE_AUTO_REPLY=false` plus `ENABLE_DM_OUTREACH=false`.