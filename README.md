# Reddit Agent

FastAPI control center and scheduled Reddit discovery pipeline for CaseClosedFL.

## Safety defaults

- Automated comments and direct messages are disabled by default.
- Protected communities, existing-attorney language, legal advice, spam patterns,
  and unapproved links are checked before engagement.
- Production auto-reply can only be changed through the deployment environment.

## Local setup

Python 3.11 is the container target.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open:

- Dashboard: http://localhost:8000/
- Health check: http://localhost:8000/api/health

The web service and scheduler run in the same process. Do not start a second
`run.py` worker beside the API or scheduled cycles will be duplicated.

## Required configuration

The service can boot without external credentials so health checks and the
dashboard remain available. Reddit discovery requires a Reddit application:

- `REDDIT_CLIENT_ID`
- `REDDIT_CLIENT_SECRET`
- `REDDIT_USER_AGENT`
- `REDDIT_USERNAME` and `REDDIT_PASSWORD` for authenticated writes

`NVIDIA_API_KEY` enables NVIDIA NIM analysis and embeddings. Without it, RAG
uses a deterministic local hashing fallback; LLM analysis remains unavailable.

`COMPOSIO_API_KEY` is accepted for the legacy adapter, but the active agent
pipeline currently uses PRAW and therefore still requires Reddit credentials.

## Data

Local defaults use SQLite at `./data/caseclosed.db` and Chroma at
`./data/chroma_db`. Docker Compose persists both in the `app_data` volume.

For production persistence, set `DATABASE_URL` to a managed PostgreSQL URL.
The included Psycopg driver accepts URLs such as
`postgresql+psycopg://USER:PASSWORD@HOST:PORT/DATABASE?sslmode=require`.
DigitalOcean App Platform's local filesystem is not durable across deploys, so
SQLite is suitable only when data loss on replacement is acceptable.

## Tests

```bash
python -m pytest -q
```

The suite isolates external APIs and covers settings parsing, rate limiting,
safety rules, contact extraction, database relationships, NVIDIA response
parsing, idempotent RAG initialization, discovery, and the full orchestrator
state flow.
