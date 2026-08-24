# DigitalOcean App Platform deployment

Deploy two components from the same revision:

- **web**: `uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}` with health check `/api/health`.
- **worker**: `python run.py`. Only this component starts the scheduler.

Use a managed PostgreSQL database and provide its private SQLAlchemy URL as
`DATABASE_URL` (for example `postgresql+psycopg://...`). Do not expose a
database port or use local container storage as production persistence.

Add all secrets in App Platform's encrypted environment settings: `SECRET_KEY`,
`OPERATOR_API_KEY`, `DATABASE_URL`, `COMPOSIO_API_KEY`,
`COMPOSIO_REDDIT_CONNECTED_ACCOUNT_ID`, `COMPOSIO_REDDIT_USER_ID`, and optional
`NVIDIA_API_KEY`.
Set `APP_ENV=production`, `REDDIT_TRANSPORT=composio`,
`ENABLE_AUTO_REPLY=false`, `ENABLE_DM_OUTREACH=false`, and
`MAX_DAILY_ENGAGEMENTS=10`.

Deployment order:

1. Create or select the managed database and configure its private URL.
2. Run `python migrate.py` **once** against that database before updating the
   web and worker components. Do not run concurrent schema upgrades in both.
3. Configure the web component and confirm `/api/health` returns `healthy`.
4. Add the worker component and verify it starts exactly one scheduler.
5. Sign in at `/access` with the operator key, then confirm the dashboard
   reports autonomous outreach as OFF.
6. The dashboard will enable autonomy only after the live Composio account
   verifies both the public-comment and direct-message actions. Keep it OFF if
   either action is unavailable; comments are intentionally not enabled alone.