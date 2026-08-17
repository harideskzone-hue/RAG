#!/usr/bin/env bash
# ==============================================================================
# VISTA AI — Single-Command Full Automated System Pipeline
# ==============================================================================
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Prioritize Python 3.10 framework with installed ML/DB dependencies
if [ -x "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3" ]; then
    PYTHON_BIN="/Library/Frameworks/Python.framework/Versions/3.10/bin/python3"
elif [ -d "venv" ]; then
    source venv/bin/activate
    PYTHON_BIN="python3"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
    PYTHON_BIN="python3"
else
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi

export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export DOCKER_HOST="unix://${HOME}/.colima/default/docker.sock"

# Launch the unified All-in-One orchestrator
exec "$PYTHON_BIN" scripts/start_all.py "$@"
