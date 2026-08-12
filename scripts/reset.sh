#!/usr/bin/env bash
set -e

echo "WARNING: This will destroy the VISTA AI Local Stack and ALL associated data."
read -p "Are you sure? (y/N) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Destroying stack and volumes..."
    cd deployment/docker
    docker-compose down -v
    echo "Reset complete."
else
    echo "Aborted."
fi
