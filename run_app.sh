# ── Load .env file ────────────────────────────────────────
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi
#!/bin/bash

# ── Cleanup on Exit ───────────────────────────────────────
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit
}
trap cleanup SIGINT

# ── Kill Process on Port ──────────────────────────────────
kill_port() {
    local port=$1
    local pids=$(lsof -t -i:$port)
    if [ ! -z "$pids" ]; then
        echo "Cleaning up port $port..."
        echo "$pids" | xargs kill -9 2>/dev/null
    fi
}

# ── Redis Connection Check ────────────────────────────────
echo "━━━ Checking Redis Connection ━━━"

python3 << 'EOF'
import os
import asyncio

try:
    import redis.asyncio as redis
except ImportError:
    print("❌ redis package not installed")
    raise SystemExit(0)

redis_url = os.getenv("REDIS_URL")

if not redis_url:
    print("⚠️  REDIS_URL not set (cache disabled)")
    raise SystemExit(0)

async def test():
    global redis_url
    if redis_url and redis_url.startswith("redis://") and ".upstash.io" in redis_url:
        redis_url = redis_url.replace("redis://", "rediss://", 1)

    try:
        r = redis.from_url(
            redis_url,
            decode_responses=True
        )
        await r.ping()
        print("✅ Redis connected successfully")
        await r.aclose()
    except Exception as e:
        print(f"❌ Redis connection FAILED: {e}")

asyncio.run(test())
EOF

echo ""

# ── Run Backend Tests ─────────────────────────────────────
echo "━━━ Running Backend Tests ━━━"
cd backend
python3 -m pytest tests/ -v --tb=short
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Tests failed! Fix the failing tests before starting the app."
    exit 1
fi
echo "✅ All tests passed!"
echo ""
cd ..

# ── Start Backend ─────────────────────────────────────────
echo "Starting Backend..."
kill_port 8000
cd backend
python3 -m uvicorn app.main:app --reload --port 8000 &
BACKEND_PID=$!
cd ..

# ── Start Frontend ────────────────────────────────────────
echo "Starting Frontend..."
kill_port 5173
cd frontend
npm run dev -- --host &
FRONTEND_PID=$!
cd ..

echo ""
echo "🚀 Application running!"
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo ""

# ── Wait for Processes ────────────────────────────────────
wait $BACKEND_PID $FRONTEND_PID