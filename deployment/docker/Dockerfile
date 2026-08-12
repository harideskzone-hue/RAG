# Stage 1: Build dependencies
FROM python:3.10-slim as builder

WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# Stage 2: Production image
FROM python:3.10-slim

# Create a non-root user
RUN groupadd -r vista && useradd -r -g vista vista

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder and install
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache /wheels/* && rm -rf /wheels

# Copy application code
COPY ./app /app/app
# COPY ./scripts /app/scripts (if needed for entrypoint)

# Set proper ownership
RUN chown -R vista:vista /app

# Switch to non-root user
USER vista

# Environment settings
ENV PYTHONPATH=/app
ENV PORT=8000

# Expose port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
