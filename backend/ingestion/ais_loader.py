"""AIS ingestion: normalize real CSV/JSON feeds or generate mock traffic.

Canonical ping schema (pandas DataFrame columns):
    mmsi (int64) · ts (datetime64[ns, UTC]) · lat · lon · sog_knots · cog_deg
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["mmsi", "ts", "lat", "lon", "sog_knots", "cog_deg"]

# Common header variants seen in AIS exports (Marine Cadastre, Spire, etc.)
COLUMN_ALIASES = {
    "MMSI": "mmsi",
    "BaseDateTime": "ts",
    "timestamp": "ts",
    "LAT": "lat",
    "LON": "lon",
    "latitude": "lat",
    "longitude": "lon",
    "SOG": "sog_knots",
    "speed": "sog_knots",
    "COG": "cog_deg",
    "course": "cog_deg",
}


def load_ais(path: str | Path) -> pd.DataFrame:
    """Read a CSV or JSON(-lines) AIS export into the canonical schema."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    df = df.rename(columns=COLUMN_ALIASES)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"AIS file {path} missing columns: {sorted(missing)}")

    # Tolerant timestamp parsing: trims nanosecond precision and Go-style
    # "+0000 UTC" suffixes (as emitted by aisstream.io) before conversion.
    ts_clean = (
        df["ts"].astype(str)
        .str.replace(r"(\.\d{6})\d+", r"\1", regex=True)
        .str.replace(r"\s*\+0000 UTC$", "+00:00", regex=True)
        .str.replace(r"\s*UTC$", "", regex=True)
    )
    df["ts"] = pd.to_datetime(ts_clean, utc=True, format="mixed")
    df = df.dropna(subset=["mmsi", "ts", "lat", "lon"])
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]
    df["mmsi"] = df["mmsi"].astype("int64")
    df = df.drop_duplicates(subset=["mmsi", "ts"]).sort_values(["mmsi", "ts"])
    return df[REQUIRED_COLUMNS].reset_index(drop=True)


def generate_mock_ais(
    vessels: list[dict],
    t_img: datetime,
    pings_per_vessel: int = 5,
    interval_s: int = 90,
    seed: int = 7,
) -> pd.DataFrame:
    """Fabricate a short AIS track history for each vessel dict
    ({mmsi, lat, lon, sog_knots, cog_deg} — position given AT t_img).

    Pings are laid out *backwards in time* from t_img along the reciprocal
    course, so dead-reckoning them forward to t_img recovers the true position
    (plus GPS-grade noise). This exercises the projection math for real.
    """
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    rng = np.random.default_rng(seed)
    knots_to_ms = 0.514444
    rows = []

    for v in vessels:
        for i in range(1, pings_per_vessel + 1):
            dt = i * interval_s  # seconds before t_img
            back_dist = v["sog_knots"] * knots_to_ms * dt
            lon_p, lat_p, _ = geod.fwd(
                v["lon"], v["lat"], (v["cog_deg"] + 180.0) % 360.0, back_dist
            )
            rows.append(
                {
                    "mmsi": v["mmsi"],
                    "ts": t_img - timedelta(seconds=dt),
                    "lat": lat_p + rng.normal(0, 1e-5),   # ~1 m GPS jitter
                    "lon": lon_p + rng.normal(0, 1e-5),
                    "sog_knots": v["sog_knots"] + rng.normal(0, 0.1),
                    "cog_deg": (v["cog_deg"] + rng.normal(0, 1.0)) % 360.0,
                }
            )

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["mmsi"] = df["mmsi"].astype("int64")
    return df.sort_values(["mmsi", "ts"]).reset_index(drop=True)
