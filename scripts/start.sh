#!/usr/bin/env bash
set -e

# Load environment variables from .env.local if present, else .env.dev, else .env.example
if [ -f ".env.local" ]; then
    ENV_FILE=".env.local"
elif [ -f ".env.dev" ]; then
    ENV_FILE=".env.dev"
else
    ENV_FILE=".env.example"
fi

echo "Starting VISTA AI Local Stack using $ENV_FILE..."

cd deployment/docker
docker-compose --env-file ../../$ENV_FILE up -d --build

echo ""
echo "VISTA AI stack is starting!"
echo "API will be available at http://localhost:8000"
echo "Grafana at http://localhost:3000"
echo "Jaeger at http://localhost:16686"
echo "MinIO Console at http://localhost:9001"
echo ""
echo "Run 'docker-compose logs -f vista-api' to see application logs."
