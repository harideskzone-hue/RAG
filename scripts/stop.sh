#!/usr/bin/env bash
set -e

echo "Stopping VISTA AI Local Stack..."

cd deployment/docker
docker-compose down

echo "Stack stopped."
