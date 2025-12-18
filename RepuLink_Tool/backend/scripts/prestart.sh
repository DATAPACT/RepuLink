#! /usr/bin/env bash

set -e
set -x

# Let the DB start
python app/backend_pre_start.py

# Run migrations
# Use 'heads' to apply all heads when multiple head revisions exist
alembic upgrade heads

# Create initial data in DB
python app/initial_data.py
