# Shared base for both the dashboard and the AIS collector — one image,
# two entrypoints (see docker-compose.yml) so the heavy Python deps
# (rasterio, pandas, streamlit) aren't duplicated across images.
FROM python:3.11-slim AS base

# rasterio's PyPI wheel bundles its own GDAL, but still dynamically links a
# few system libs the slim base doesn't ship (libexpat); curl is used by
# the dashboard's healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/
COPY app/ app/

# Data lives on a mounted volume — not baked into the image, so config,
# the AIS store, and cached snapshots survive image rebuilds/redeploys.
RUN mkdir -p /app/data

EXPOSE 8501

# No CMD here — docker-compose.yml sets the command per service
# (streamlit for the dashboard, `python -m backend.ingestion.ais_collector`
# for the collector) so one image serves both resource-isolated processes.
