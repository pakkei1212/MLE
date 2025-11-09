#!/bin/bash
set -e

# This script is run when the PostgreSQL container starts for the first time.
# It creates a dedicated database for MLflow if it doesn't already exist.
# This version uses `echo` to pipe the command into psql, which is a robust
# way to avoid issues with both line endings and command interpretation.

echo "SELECT 'CREATE DATABASE mlflow' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec" | psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" 