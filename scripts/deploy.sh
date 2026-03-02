#!/bin/bash
set -e

PROJECT_DIR="/home/ec2-user/Horisation"
FRONTEND_DIR="$PROJECT_DIR/frontend"
VENV="/home/ec2-user/venv311"
SERVICE="horisation"

echo "=============================="
echo " Horisation Deploy Script"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================="

echo "[1/5] Pulling latest code..."
cd "$PROJECT_DIR"
git fetch origin
git reset --hard origin/main
echo "      Done."

echo "[2/5] Installing Python dependencies..."
$VENV/bin/pip install -r requirements.txt --quiet
echo "      Done."

echo "[3/5] Checking frontend dependencies..."
cd "$FRONTEND_DIR"
npm install --silent
echo "      Done."

echo "[4/5] Building React app..."
npm run build
echo "      Done."

echo "[5/5] Restarting service..."
sudo systemctl restart "$SERVICE"
sleep 2
sudo systemctl status "$SERVICE" --no-pager -l
echo "=============================="
echo " Deploy complete!"
echo "=============================="
