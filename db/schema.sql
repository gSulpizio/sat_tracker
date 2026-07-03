-- sat_tracker PostGIS schema
-- Requires: CREATE EXTENSION postgis;

CREATE EXTENSION IF NOT EXISTS postgis;

-- Satellite scenes ingested from the Copernicus STAC catalog
CREATE TABLE scenes (
    scene_id        TEXT PRIMARY KEY,              -- STAC item id
    platform        TEXT NOT NULL,                 -- sentinel-1a / sentinel-2b ...
    product_type    TEXT NOT NULL,                 -- GRD / L2A
    acquired_at     TIMESTAMPTZ NOT NULL,          -- T_img
    footprint       GEOMETRY(POLYGON, 4326) NOT NULL,
    asset_href      TEXT NOT NULL,                 -- COG URL
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX scenes_footprint_gix ON scenes USING GIST (footprint);
CREATE INDEX scenes_acquired_ix   ON scenes (acquired_at);

-- Raw AIS pings (append-only)
CREATE TABLE ais_pings (
    ping_id     BIGSERIAL PRIMARY KEY,
    mmsi        BIGINT NOT NULL,
    ts          TIMESTAMPTZ NOT NULL,
    geom        GEOMETRY(POINT, 4326) NOT NULL,
    sog_knots   DOUBLE PRECISION,                  -- speed over ground
    cog_deg     DOUBLE PRECISION,                  -- course over ground
    heading_deg DOUBLE PRECISION,
    source      TEXT NOT NULL DEFAULT 'stream',
    UNIQUE (mmsi, ts)
);
CREATE INDEX ais_pings_geom_gix ON ais_pings USING GIST (geom);
CREATE INDEX ais_pings_mmsi_ts_ix ON ais_pings (mmsi, ts);

-- ML detections extracted from a scene
CREATE TABLE detections (
    detection_id  BIGSERIAL PRIMARY KEY,
    scene_id      TEXT NOT NULL REFERENCES scenes(scene_id),
    centroid      GEOMETRY(POINT, 4326) NOT NULL,
    obb           GEOMETRY(POLYGON, 4326),         -- oriented bounding box
    confidence    DOUBLE PRECISION NOT NULL,
    length_m      DOUBLE PRECISION,                -- estimated from OBB
    model_name    TEXT NOT NULL,                   -- e.g. yolov8m-obb@sha
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX detections_centroid_gix ON detections USING GIST (centroid);
CREATE INDEX detections_scene_ix ON detections (scene_id);

-- Output of the spatial-temporal fusion for one scene
CREATE TABLE fusion_results (
    result_id     BIGSERIAL PRIMARY KEY,
    scene_id      TEXT NOT NULL REFERENCES scenes(scene_id),
    detection_id  BIGINT REFERENCES detections(detection_id),  -- NULL for AIS_ONLY
    mmsi          BIGINT,                                      -- NULL for DARK
    status        TEXT NOT NULL CHECK (status IN ('VERIFIED', 'AIS_ONLY', 'DARK')),
    match_dist_m  DOUBLE PRECISION,                            -- Hungarian match distance
    geom          GEOMETRY(POINT, 4326) NOT NULL,              -- display position
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (detection_id IS NOT NULL OR mmsi IS NOT NULL)
);
CREATE INDEX fusion_results_scene_ix ON fusion_results (scene_id, status);
CREATE INDEX fusion_results_geom_gix ON fusion_results USING GIST (geom);

-- Human-in-the-loop corrections (audit trail; doubles as a label source
-- for model retraining)
CREATE TABLE corrections (
    correction_id BIGSERIAL PRIMARY KEY,
    scene_id      TEXT NOT NULL REFERENCES scenes(scene_id),
    result_id     BIGINT REFERENCES fusion_results(result_id),
    action        TEXT NOT NULL CHECK (action IN ('ADD', 'DELETE', 'RELINK')),
    geom          GEOMETRY(POINT, 4326),           -- for ADD
    linked_mmsi   BIGINT,                          -- for RELINK
    analyst       TEXT NOT NULL,
    note          TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX corrections_scene_ix ON corrections (scene_id);
