#!/bin/bash
set -e

echo "Running Benchmarks..."
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Run 'make setup' first."
    exit 1
fi

source .venv/bin/activate
pytest tests/benchmark
