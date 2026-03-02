#!/bin/bash
set -e

PROJECT_DIR="/home/ec2-user/Horisation"
FRONTEND_DIR="$PROJECT_DIR/frontend"
SERVICE="horisation"

echo "=============================="
echo " Horisation Deploy Script"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================="

# 1. Pull latest code
echo "[1/4] Pulling latest code..."
cd "$PROJECT_DIR"
git fetch origin
git reset --hard origin/main
echo "      Done."

# 2. Install frontend dependencies (only if package.json changed)
echo "[2/4] Checking frontend dependencies..."
cd "$FRONTEND_DIR"
npm install --silent
echo "      Done."

# 3. Build React app
echo "[3/4] Building React app..."
npm run build
echo "      Done."

# 4. Restart service
echo "[4/4] Restarting service..."
sudo systemctl restart "$SERVICE"
sleep 2
sudo systemctl status "$SERVICE" --no-pager -l
echo "=============================="
echo " Deploy complete!"
echo "=============================="
