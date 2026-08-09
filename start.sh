#!/bin/bash
# Start script for GPT-2 project
# Starts both FastAPI backend and Vite frontend

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== GPT-2 Project ==="
echo ""

# Check if backend venv exists
if [ ! -d "$SCRIPT_DIR/backend/venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$SCRIPT_DIR/backend/venv"
fi

# Activate venv
source "$SCRIPT_DIR/backend/venv/bin/activate"

# Install dependencies
echo "Installing Python dependencies..."
pip install -q -r "$SCRIPT_DIR/backend/requirements.txt"

# Start FastAPI backend in background
echo "Starting FastAPI backend on port 8011..."
cd "$SCRIPT_DIR/backend"
python app.py &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to be ready
echo "Waiting for backend..."
for i in {1..30}; do
    if curl -s http://localhost:8011/api/health > /dev/null 2>&1; then
        echo "Backend ready!"
        break
    fi
    sleep 1
done

# Start Vite frontend
echo "Starting Vite frontend on port 5173..."
cd "$SCRIPT_DIR"
npm run dev &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

echo ""
echo "=== Services Running ==="
echo "Frontend: http://localhost:5173"
echo "Backend:  http://localhost:8011"
echo "API Docs: http://localhost:8011/docs"
echo ""
echo "Press Ctrl+C to stop both services"

# Trap to cleanup
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait