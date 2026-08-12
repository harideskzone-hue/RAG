#!/bin/bash
set -e

echo "Running Test Suite..."
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Run 'make setup' first."
    exit 1
fi

source .venv/bin/activate
pytest tests/unit tests/integration tests/e2e
