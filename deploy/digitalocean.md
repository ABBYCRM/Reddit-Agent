# DigitalOcean App Platform deployment

## Component

Deploy this repository as one Web Service using the root `Dockerfile`.

- Source directory: repository root
- Build command: leave blank when using the Dockerfile
- Run command: leave blank to use the Dockerfile command
- HTTP port: the platform-provided `PORT`
- Health-check path: `/api/health`
- Instance count: 1 unless the scheduler is moved to a separate component

The Docker command runs Uvicorn on `0.0.0.0:${PORT:-8000}`. FastAPI startup
creates the database tables, initializes the RAG knowledge base idempotently,
and starts the scheduler.

## Environment variables

Store credentials as encrypted runtime secrets, never build-time variables.

Required for Reddit discovery:

- `REDDIT_CLIENT_ID`
- `REDDIT_CLIENT_SECRET`
- `REDDIT_USER_AGENT`

Required for authenticated Reddit writes:

- `REDDIT_USERNAME`
- `REDDIT_PASSWORD`

Recommended:

- `NVIDIA_API_KEY`
- `DATABASE_URL`
- `APP_ENV=production`
- `ENABLE_AUTO_REPLY=false`
- `ENABLE_DM_OUTREACH=false`

Optional tuning values are documented in `.env.example`.

## Database

Use a managed PostgreSQL database for durable production data. Set
`DATABASE_URL` to the provider URL using the Psycopg dialect, for example:

```text
postgresql+psycopg://USER:PASSWORD@HOST:PORT/DATABASE?sslmode=require
```

The SQLite default lets the service boot without a database component, but App
Platform's local filesystem is ephemeral. A replacement or redeploy can remove
the database and Chroma files.

## Verification

After deployment:

1. Confirm the build finishes with dependency installation exit code 0.
2. Confirm runtime logs reach `Application startup complete`.
3. Confirm DigitalOcean marks the Web Service Active/Healthy.
4. Request `/api/health` and verify HTTP 200.
5. Open `/` and verify the dashboard renders.
6. Trigger one discovery cycle with auto-reply disabled and inspect the run log.

If startup exits before Uvicorn listens, use the first fatal runtime exception
as the failure boundary. Do not change the health check to mask a process,
database, or credential failure.
