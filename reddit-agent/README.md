# CaseClosedFL Reddit Agent

An operator-only Reddit discovery and intake assistant. It uses Composio as
the default Reddit transport and keeps autonomous outreach off until a protected
operator switch verifies that both public-comment and direct-message actions are
available.

## Local run

1. Copy `.env.example` to `.env` and set **new** local values for
   `SECRET_KEY` and `OPERATOR_API_KEY`.
2. Install dependencies: `pip install -r requirements.txt`
3. Start the protected web control surface: `uvicorn api:app --host 0.0.0.0 --port 8000`
4. Start the scheduler separately: `python run.py`

Open `/access` and enter the operator key to create an HTTP-only operator
session, or pass `X-Operator-Key` to access operational APIs. The health
endpoint does not expose lead data and is intentionally unauthenticated.

## Autonomous outreach

The dashboard's **Turn Autonomous Outreach ON** control is a shared,
database-backed kill switch. It refuses to enable unless the current Reddit
transport can verify both comments and direct messages. Every provider call is
reserved in PostgreSQL before it is made, with an absolute combined limit of
10 comments/DMs per UTC day. Provider timeouts are marked `unknown` and are
never automatically retried, preventing duplicate messages.

The currently configured Composio Reddit toolkit must expose a direct-message
action before this mode can be enabled. If it does not, the dashboard remains
OFF with a clear readiness reason; comments are not enabled on their own.

## Verification

Run `pytest -q` from this directory. Tests use temporary SQLite databases and
mocked provider boundaries; they never perform live Reddit, Composio, or LLM
actions.

## Production

See `deploy/digitalocean.md`. Use encrypted environment variables, a managed
PostgreSQL database, a separate web and worker process, and leave
`ENABLE_AUTO_REPLY=false` plus `ENABLE_DM_OUTREACH=false`; the runtime
operator control is the only mechanism that can enable outreach.