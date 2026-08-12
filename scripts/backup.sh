#!/usr/bin/env bash
set -e

BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "Starting VISTA AI Local Infrastructure Backup..."

echo "1. Backing up PostgreSQL..."
docker exec vista-postgres pg_dump -U vista_user vista_db > "$BACKUP_DIR/postgres_dump.sql"

echo "2. Backing up MongoDB..."
docker exec vista-mongo mongodump --archive > "$BACKUP_DIR/mongo_dump.archive"

echo "3. Backing up MinIO Data..."
# We can copy the mapped volume data, or use MinIO client. Here we just copy the volume contents roughly.
# For local dev, a simple docker cp works well enough.
docker cp vista-minio:/data "$BACKUP_DIR/minio_data"

echo "Backup completed successfully! Stored in $BACKUP_DIR"
