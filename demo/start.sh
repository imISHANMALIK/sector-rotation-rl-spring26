#!/usr/bin/env bash
set -e

REPO="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Sector Rotation RL — Dashboard ==="
echo ""

# clear stale processes
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
sleep 0.5

# ── Backend ────────────────────────────────────────────
echo "▶ Starting FastAPI backend on :8000"
cd "$REPO"

if ! python3 -c "import fastapi, uvicorn" 2>/dev/null; then
  echo "  Installing backend dependencies..."
  pip install -q fastapi uvicorn[standard]
fi

python3 -m uvicorn demo.backend.main:app --reload --port 8000 &
BACKEND_PID=$!
echo "  Backend PID: $BACKEND_PID"

# Wait for backend to be ready
sleep 2
echo "  Backend ready at http://localhost:8000"
echo ""

# ── Frontend ───────────────────────────────────────────
echo "▶ Starting Next.js frontend on :3000"
cd "$REPO/demo/web"

if [ ! -d node_modules ]; then
  echo "  Installing npm packages..."
  npm install
fi

npm run dev &
FRONTEND_PID=$!
echo "  Frontend PID: $FRONTEND_PID"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Dashboard: http://localhost:3000"
echo "  API:       http://localhost:8000"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Press Ctrl+C to stop both servers"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo ''; echo 'Stopped.'" INT TERM
wait
