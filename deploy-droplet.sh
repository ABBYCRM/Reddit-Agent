#!/bin/bash
# CaseClosedFL Reddit Agent - Droplet Deploy Script
# Run this on a fresh Ubuntu 22.04/24.04 Droplet

set -e

echo "=========================================="
echo "CaseClosedFL Agent - Droplet Setup"
echo "=========================================="

# Update system
echo "[1/6] Updating system..."
apt-get update && apt-get upgrade -y

# Install dependencies
echo "[2/6] Installing dependencies..."
apt-get install -y python3 python3-pip python3-venv git gcc g++ libpq-dev

# Create app directory
echo "[3/6] Setting up app directory..."
mkdir -p /opt/caseclosed-agent
cd /opt/caseclosed-agent

# Clone repo
echo "[4/6] Cloning repository..."
git clone https://github.com/ABBYCRM/Reddit-Agent.git .

# Create virtual environment
echo "[5/6] Creating Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install requirements
echo "[6/6] Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file
echo "Creating .env file..."
cat > .env << 'EOF'
APP_ENV=production
LOG_LEVEL=INFO
NVIDIA_API_KEY=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
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=meta/llama-3.3-70b-instruct
COMPOSIO_API_KEY=ak_fCyyEyny9xQrq9slmGhL
DATABASE_URL=sqlite:///app/agent.db
CHROMA_PERSIST_DIR=./chroma_db
REDDIT_USER_AGENT=caseclosedfl-agent/1.0
ENABLE_AUTO_REPLY=false
ENABLE_DM_OUTREACH=false
MAX_DAILY_ENGAGEMENTS=50
FLORIDA_BAR_COMPLIANT=true
LEAD_SCORE_THRESHOLD=75
AUTO_QUALIFY_THRESHOLD=90
DISCOVERY_INTERVAL_MINUTES=30
MONITOR_INTERVAL_MINUTES=15
HEARTBEAT_INTERVAL_SECONDS=60
SECRET_KEY=caseclosed-production-secret-2026
EOF

echo ""
echo "=========================================="
echo "SETUP COMPLETE"
echo "=========================================="
echo ""
echo "NEXT STEPS:"
echo "1. Add your Reddit credentials to .env:"
echo "   REDDIT_CLIENT_ID=your_id"
echo "   REDDIT_CLIENT_SECRET=your_secret"
echo "   REDDIT_USERNAME=your_username"
echo "   REDDIT_PASSWORD=your_password"
echo ""
echo "2. Start the agent:"
echo "   cd /opt/caseclosed-agent"
echo "   source venv/bin/activate"
echo "   python run.py"
echo ""
echo "3. Access dashboard at:"
echo "   http://YOUR_DROPLET_IP:8000"
echo ""
echo "4. To run in background:"
echo "   nohup python run.py > agent.log 2>&1 &"
echo "=========================================="
