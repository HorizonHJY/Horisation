#!/bin/bash
# Production-like local test: build React then serve via Flask on :5000
# Open http://localhost:5000 in browser

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=============================="
echo " Horisation — Build & Run"
echo "=============================="

# Build React
echo "[1/2] Building React app..."
cd "$ROOT/frontend"
npm run build
echo "      Done."

# Start Flask (serves dist/)
echo "[2/2] Starting Flask..."
echo "      Open http://localhost:5000"
echo "      Press Ctrl+C to stop"
echo "=============================="
cd "$ROOT"
python app.py
