#!/bin/bash
set -e

echo "Starting VISTA AI Natively..."

if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Run 'make setup' first."
    exit 1
fi

source .venv/bin/activate

# Default to native mode unless overridden
export MODE=${MODE:-native}
export TELEMETRY_EXPORTER=${TELEMETRY_EXPORTER:-console}

echo "Mode: $MODE"
echo "Telemetry: $TELEMETRY_EXPORTER"

# Start FastAPI server
uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
