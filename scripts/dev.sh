#!/bin/bash
# Dev mode: Vite (hot reload) on :5173 + Flask API on :5000
# Open http://localhost:5173 in browser

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=============================="
echo " Horisation — Dev Mode"
echo " Frontend : http://localhost:5173"
echo " API      : http://localhost:5000"
echo " Press Ctrl+C to stop both"
echo "=============================="

# Kill both servers on exit
cleanup() {
  echo ""
  echo "Stopping servers..."
  kill "$FLASK_PID" 2>/dev/null
  kill "$VITE_PID" 2>/dev/null
  exit 0
}
trap cleanup INT TERM

# Start Flask
cd "$ROOT"
python app.py &
FLASK_PID=$!

# Install frontend deps if needed
cd "$ROOT/frontend"
npm install --silent

# Start Vite
npm run dev &
VITE_PID=$!

# Wait for either to exit
wait "$FLASK_PID" "$VITE_PID"
