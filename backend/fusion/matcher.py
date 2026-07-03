"""Spatial-temporal fusion of SAR detections with AIS tracks.

Algorithm
---------
1. Window AIS pings to |t_ping − T_img| ≤ Δt (default ±5 min); keep the ping
   closest to T_img per MMSI.
2. Dead-reckon each vessel from its ping time to T_img along its COG at its
   SOG (geodesic forward solve on the WGS84 ellipsoid; negative Δt projects
   backwards for pings received after the acquisition).
3. Build a detections × AIS geodesic distance matrix and solve the optimal
   assignment (Hungarian / linear_sum_assignment).
4. Gate assignments at `match_gate_m` (1 km). Ungated pairs become VERIFIED;
   unmatched detections become DARK; unmatched AIS targets become AIS_ONLY
   (broadcasting but not imaged — possible spoofing or sub-resolution hull).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.optimize import linear_sum_assignment

from backend.detection.base import Detection

GEOD = Geod(ellps="WGS84")
KNOTS_TO_MS = 0.514444


@dataclass
class FusedVessel:
    status: str                     # VERIFIED | AIS_ONLY | DARK
    lat: float
    lon: float
    mmsi: int | None = None
    detection: Detection | None = None
    match_dist_m: float | None = None
    projected_from_s: float | None = None  # AIS dead-reckoning Δt used

    def to_dict(self) -> dict:
        d = {
            "status": self.status,
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "mmsi": self.mmsi,
            "match_dist_m": round(self.match_dist_m, 1) if self.match_dist_m is not None else None,
        }
        if self.detection is not None:
            d["detection"] = self.detection.to_dict()
        return d


def project_ais_to_image_time(ais: pd.DataFrame, t_img: datetime, window_s: int) -> pd.DataFrame:
    """Steps 1–2: temporal window + per-MMSI dead reckoning to T_img.

    Returns one row per MMSI with columns:
    mmsi, lat_proj, lon_proj, sog_knots, cog_deg, dt_s.
    """
    t_img = pd.Timestamp(t_img).tz_convert("UTC")
    dt = (t_img - ais["ts"]).dt.total_seconds()
    windowed = ais.loc[dt.abs() <= window_s].copy()
    if windowed.empty:
        return pd.DataFrame(
            columns=["mmsi", "lat_proj", "lon_proj", "sog_knots", "cog_deg", "dt_s"]
        )
    windowed["dt_s"] = (t_img - windowed["ts"]).dt.total_seconds()

    # Latest usable ping per vessel (smallest |Δt|)
    idx = windowed["dt_s"].abs().groupby(windowed["mmsi"]).idxmin()
    latest = windowed.loc[idx].reset_index(drop=True)

    # Vectorized geodesic forward solve. Negative distance (ping after T_img)
    # walks backwards along the course — pyproj handles the sign natively.
    dist_m = latest["sog_knots"].to_numpy() * KNOTS_TO_MS * latest["dt_s"].to_numpy()
    lon_p, lat_p, _ = GEOD.fwd(
        latest["lon"].to_numpy(),
        latest["lat"].to_numpy(),
        latest["cog_deg"].to_numpy(),
        dist_m,
    )
    latest["lat_proj"] = lat_p
    latest["lon_proj"] = lon_p
    return latest[["mmsi", "lat_proj", "lon_proj", "sog_knots", "cog_deg", "dt_s"]]


def _distance_matrix(detections: list[Detection], ais_proj: pd.DataFrame) -> np.ndarray:
    """Geodesic metres, shape (n_detections, n_ais)."""
    n_det, n_ais = len(detections), len(ais_proj)
    det_lon = np.array([d.lon for d in detections])
    det_lat = np.array([d.lat for d in detections])
    ais_lon = ais_proj["lon_proj"].to_numpy()
    ais_lat = ais_proj["lat_proj"].to_numpy()

    # Broadcast to flat pairs for one vectorized GEOD.inv call
    dl = np.repeat(det_lon, n_ais)
    dp = np.repeat(det_lat, n_ais)
    al = np.tile(ais_lon, n_det)
    ap = np.tile(ais_lat, n_det)
    _, _, dist = GEOD.inv(dl, dp, al, ap)
    return dist.reshape(n_det, n_ais)


def fuse(
    detections: list[Detection],
    ais: pd.DataFrame,
    t_img: datetime,
    window_s: int = 300,
    gate_m: float = 1000.0,
) -> list[FusedVessel]:
    ais_proj = project_ais_to_image_time(ais, t_img, window_s)

    matched_det: set[int] = set()
    matched_ais: set[int] = set()
    fused: list[FusedVessel] = []

    if detections and not ais_proj.empty:
        cost = _distance_matrix(detections, ais_proj)
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if cost[r, c] > gate_m:
                continue  # optimal pairing but outside the 1 km gate
            det = detections[r]
            track = ais_proj.iloc[c]
            matched_det.add(r)
            matched_ais.add(c)
            fused.append(
                FusedVessel(
                    status="VERIFIED",
                    lat=det.lat,               # radar position is the more precise one
                    lon=det.lon,
                    mmsi=int(track["mmsi"]),
                    detection=det,
                    match_dist_m=float(cost[r, c]),
                    projected_from_s=float(track["dt_s"]),
                )
            )

    for i, det in enumerate(detections):
        if i not in matched_det:
            fused.append(
                FusedVessel(status="DARK", lat=det.lat, lon=det.lon, detection=det)
            )

    for j in range(len(ais_proj)):
        if j not in matched_ais:
            track = ais_proj.iloc[j]
            fused.append(
                FusedVessel(
                    status="AIS_ONLY",
                    lat=float(track["lat_proj"]),
                    lon=float(track["lon_proj"]),
                    mmsi=int(track["mmsi"]),
                    projected_from_s=float(track["dt_s"]),
                )
            )

    return fused


def summarize(fused: list[FusedVessel]) -> dict:
    n_verified = sum(1 for f in fused if f.status == "VERIFIED")
    n_dark = sum(1 for f in fused if f.status == "DARK")
    n_ais_only = sum(1 for f in fused if f.status == "AIS_ONLY")
    total_detected = n_verified + n_dark
    return {
        "total_detected": total_detected,
        "total_ais_active": n_verified + n_ais_only,
        "verified": n_verified,
        "ais_only": n_ais_only,
        "dark_vessels": n_dark,
        # Fraction of radar-imaged traffic not accounted for by AIS
        "dark_fleet_ratio": round(n_dark / total_detected, 3) if total_detected else 0.0,
        # Multiplier: how much bigger the real fleet is vs. what AIS matching shows
        "dark_fleet_multiplier": round(total_detected / n_verified, 2) if n_verified else None,
    }
