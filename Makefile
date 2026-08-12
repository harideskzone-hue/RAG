.PHONY: setup run test lint format benchmark docs clean start stop reset smoke logs backup help

help:
	@echo "VISTA AI Makefile"
	@echo "-----------------"
	@echo "Native Developer Commands:"
	@echo "  setup     - Setup Python virtual environment and dependencies"
	@echo "  run       - Run the FastAPI server locally (Native mode)"
	@echo "  test      - Run the Pytest suite"
	@echo "  lint      - Lint the codebase (Ruff/MyPy/Bandit)"
	@echo "  format    - Format the codebase (Black)"
	@echo "  benchmark - Run performance benchmarks"
	@echo "  docs      - Build/Preview documentation (Placeholder)"
	@echo "  clean     - Remove .venv and caches"
	@echo ""
	@echo "Docker Deployment Commands:"
	@echo "  start     - Start the local Docker Compose stack"
	@echo "  stop      - Stop the stack"
	@echo "  reset     - Destroy the stack and purge all volumes"
	@echo "  smoke     - Run the smoke tests against the active stack"
	@echo "  logs      - Tail the logs for the API service"
	@echo "  backup    - Backup local infrastructure data"

# Native Commands
setup:
	@python3 -m venv .venv
	@. .venv/bin/activate && pip install -U pip && pip install -r requirements-dev.txt
	@echo "Setup complete. Run 'source .venv/bin/activate' and 'make run'."

run:
	@./scripts/run_local.sh

test:
	@./scripts/test_local.sh

lint:
	@./scripts/lint_local.sh

format:
	@. .venv/bin/activate && black app/ tests/

benchmark:
	@./scripts/benchmark_local.sh

docs:
	@echo "Docs generation coming soon."

clean:
	@rm -rf .venv __pycache__ .pytest_cache .ruff_cache reports/
	@echo "Cleaned workspace."

# Docker Commands
start:
	@./scripts/start.sh

stop:
	@./scripts/stop.sh

reset:
	@./scripts/reset.sh

smoke:
	@./scripts/smoke_test.sh

logs:
	@cd deployment/docker && docker-compose logs -f vista-api

backup:
	@./scripts/backup.sh
