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

# _data/ is gitignored — git reset --hard never touches users or market.db
git fetch origin
git reset --hard origin/main

echo "      Done."

echo "[2/5] Installing Python dependencies..."
$VENV/bin/pip install -r requirements.txt --quiet
echo "      Done."

# The frontend is built in CI and uploaded to frontend/dist.new (SKIP_BUILD=1).
# The box is too small to build it: since plotly landed (2026-10-19) `vite build`
# thrashed memory for 10 minutes, and because vite empties dist/ first, the site
# answered 404 on / for ~33 hours while /api stayed up. Building on the box is
# kept only as a manual fallback, and it too builds aside and swaps at the end.
pkill -f "vite build" 2>/dev/null || true   # a timed-out build can outlive its ssh session and eat the RAM
if [ "$SKIP_BUILD" = "1" ]; then
  echo "[3/5] Using the frontend built in CI..."
  [ -f "$FRONTEND_DIR/dist.new/index.html" ] || { echo "ERROR: frontend/dist.new/index.html missing"; exit 1; }
  echo "[4/5] Swapping in the new frontend..."
else
  echo "[3/5] Checking frontend dependencies..."
  cd "$FRONTEND_DIR"
  npm install --silent
  echo "[4/5] Building React app (aside, then swap)..."
  rm -rf dist.new
  NODE_OPTIONS=--max-old-space-size=768 timeout 600 npx vite build --outDir dist.new --emptyOutDir
fi
cd "$FRONTEND_DIR"
rm -rf dist.old
[ -d dist ] && mv dist dist.old
mv dist.new dist
rm -rf dist.old
echo "      Done."

echo "[5/5] Restarting service..."
sudo systemctl restart "$SERVICE"
sleep 2
sudo systemctl status "$SERVICE" --no-pager -l
echo "=============================="
echo " Deploy complete!"
echo "=============================="
