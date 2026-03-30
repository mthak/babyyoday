#!/bin/sh
set -e

# Seed EFS with sample data index on first boot
if [ ! -f /app/data/faiss.index ]; then
  echo "No FAISS index found — seeding from sample_data..."
  mkdir -p /app/data/docs /app/data/incoming

  # Copy sample docs to docs dir
  cp /app/sample_data/* /app/data/docs/

  # Build index
  python3 -c "
import sys
sys.path.insert(0, '/app')
from data_pipeline.reindex import reindex
reindex(docs_dir='/app/data/docs', output_dir='/app/data')
print('Index seeded successfully')
"
  echo "Seeding complete."
fi

# Start inference server and admin panel
uvicorn inference.server:app --host 0.0.0.0 --port 8000 &
uvicorn admin.app:admin_app --host 0.0.0.0 --port 8001 &
wait
