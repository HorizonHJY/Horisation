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

# Preserve runtime data that must survive git reset
USERS_FILE="$PROJECT_DIR/_data/users.json"
SESSIONS_FILE="$PROJECT_DIR/_data/sessions.json"
[ -f "$USERS_FILE" ]   && cp "$USERS_FILE"   /tmp/_users_backup.json
[ -f "$SESSIONS_FILE" ] && cp "$SESSIONS_FILE" /tmp/_sessions_backup.json

git fetch origin
git reset --hard origin/main

# Restore preserved data
[ -f /tmp/_users_backup.json ]    && cp /tmp/_users_backup.json   "$USERS_FILE"
[ -f /tmp/_sessions_backup.json ] && cp /tmp/_sessions_backup.json "$SESSIONS_FILE"

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
