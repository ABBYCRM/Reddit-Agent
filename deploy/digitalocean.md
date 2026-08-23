# Deploy CaseClosedFL Agent to DigitalOcean

## Option 1: DigitalOcean App Platform (Easiest)

1. Push code to GitHub
2. In DigitalOcean Console: Apps -> Create App -> GitHub
3. Select repository and branch
4. Configure environment variables from `.env`
5. Add PostgreSQL database (managed)
6. Add Redis (managed)
7. Deploy

## Option 2: DigitalOcean Droplet + Docker

```bash
# 1. Create Droplet (Ubuntu 22.04, 2GB RAM minimum)
# 2. SSH in
sudo apt update && sudo apt install -y docker.io docker-compose
git clone <your-repo>
cd caseclosed-reddit-agent
cp .env.example .env
# Edit .env with real credentials
sudo docker-compose up -d
```

## Option 3: DigitalOcean Gradient AI ADK (Future)

When ADK supports custom agent runtimes:
```bash
pip install gradient-adk
gradient agent deploy
```

## Environment Variables Required

- `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET`
- `REDDIT_USERNAME` / `REDDIT_PASSWORD`
- `NVIDIA_API_KEY` (get free at build.nvidia.com)
- `DATABASE_URL`
- `REDIS_URL`

## Monitoring

- Dashboard: `http://your-droplet-ip:8000`
- Health: `http://your-droplet-ip:8000/api/health`
- Logs: `docker logs caseclosed-worker`

## Reddit API Commercial Note

Per Reddit's 2026 terms: Free tier = 100 QPM. Commercial use requires contract at $0.24/1K calls.
CaseClosedFL should request commercial approval if scaling beyond research/monitoring.
