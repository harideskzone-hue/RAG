#!/bin/bash
set -e

echo "Running Linting..."
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Run 'make setup' first."
    exit 1
fi

source .venv/bin/activate
echo "Running Ruff..."
ruff check .
echo "Running Black..."
black --check .
echo "Running Bandit..."
bandit -r app/
echo "Linting complete!"
