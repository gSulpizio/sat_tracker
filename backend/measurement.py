"""Physical vessel size estimation from native-resolution SAR chips.

The detection pass reads the whole AOI decimated to ≤2048 px (often 30–60
m/px), where a 150 m ship is 3–5 px — fine for finding ships, useless for
measuring them. This module goes back to the source COG and reads a small
NATIVE-resolution window (~10 m/px for Sentinel-1 IW GRD) around each
detection, segments the bright target from sea clutter, and measures its
principal-axis extent:

    length_m   — extent along the major axis
    width_m    — extent along the minor axis (beam)
    heading_deg — major-axis orientation (0–180°, N=0, E=90; a SAR blob
                  can't distinguish bow from stern, hence the 180° ambiguity)

No ML involved: at 10 m/px an isolated bright ship on dark water is a
textbook threshold-and-PCA problem, and this works identically no matter
which detector backend produced the centroid.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from backend.config import Settings, settings

log = logging.getLogger(__name__)

CHIP_PX = 192          # native-res window ≈ 1.9 km — plenty around one ship
# Empirically tuned on real Gulf-of-Suez anchorage chips: 6 dB absorbs the
# cross-shaped sidelobe arms bright ships throw (446 m "ships"), 14 dB
# fragments the hull; 10 dB yields physically plausible hull geometries.
# Expect ±1–2 px (≈10–25 m) accuracy; faint hull extremities are clipped,
# so treat results as a lower-bound estimate, not registry-grade lengths.
THRESHOLD_DB = 10.0    # target must exceed local sea background by this much
SEARCH_RADIUS_PX = 12  # blob must start within this range of the detection
MAX_SHIP_M = 500.0     # sanity cap — longer "ships" are land/clutter leakage


@dataclass
class SizeEstimate:
    length_m: float
    width_m: float
    heading_deg: float
    reliable: bool      # False when the blob touches the chip edge etc.
    # Oriented footprint rectangle [(lon, lat) × 4] of the measured radar
    # blob — lets the UI show exactly WHAT was measured, not just numbers.
    corners_lonlat: list[tuple[float, float]] | None = None


def measure_detections(asset_hrefs: list[str], detections: list,
                       cfg: Settings | None = None) -> int:
    """Fill in length/width/heading on Detection objects in place.

    `asset_hrefs` are the pass's frame COGs (a mosaicked pass has several);
    each detection is measured from the first frame whose extent contains
    it. Returns how many detections got a measurement. Never raises — size
    is a bonus annotation, not worth failing a pass over.
    """
    cfg = cfg or settings
    import rasterio
    from rasterio.vrt import WarpedVRT

    from backend.ingestion.stac_client import _gdal_env

    measured = 0
    for href in asset_hrefs:
        # heading is the "already measured" marker: detections may arrive
        # with a model-estimated length (kept authoritative) but still get
        # their footprint geometry measured here for visualization.
        todo = [d for d in detections if d.heading_deg is None]
        if not todo:
            break
        try:
            with rasterio.Env(**_gdal_env(href, cfg)):
                with rasterio.open(href) as src:
                    gcps, gcp_crs = src.gcps
                    if src.crs is None and gcps:
                        vrt = WarpedVRT(src, src_crs=gcp_crs, crs="EPSG:4326")
                    elif src.crs is not None and src.crs.to_epsg() != 4326:
                        vrt = WarpedVRT(src, crs="EPSG:4326")
                    else:
                        vrt = src
                    try:
                        for det in todo:
                            est = _measure_one(vrt, det.lon, det.lat)
                            if est is not None and est.reliable:
                                if det.length_m is None:
                                    det.length_m = round(est.length_m, 1)
                                det.width_m = round(est.width_m, 1)
                                det.heading_deg = round(est.heading_deg, 1)
                                if est.corners_lonlat and not det.obb_lonlat:
                                    det.obb_lonlat = est.corners_lonlat
                                measured += 1
                    finally:
                        if vrt is not src:
                            vrt.close()
        except Exception as exc:
            log.warning("Size measurement failed for frame %s: %s",
                        href.rsplit('/', 1)[-1][:40], exc)
    return measured


def _measure_one(vrt, lon: float, lat: float) -> SizeEstimate | None:
    from rasterio.windows import Window

    fcol, frow = (~vrt.transform) * (lon, lat)
    row, col = int(round(frow)), int(round(fcol))
    half = CHIP_PX // 2
    r0, c0 = row - half, col - half
    if r0 < 0 or c0 < 0 or r0 + CHIP_PX > vrt.height or c0 + CHIP_PX > vrt.width:
        return None  # detection sits at the frame edge — skip, next frame may cover it
    amp = vrt.read(1, window=Window(c0, r0, CHIP_PX, CHIP_PX),
                   out_dtype="float32")
    valid = amp > 0
    if valid.mean() < 0.5:
        return None
    db = 10.0 * np.log10(np.maximum(amp, 1.0) ** 2)

    # Sea background from the median of valid pixels (ships are a tiny
    # fraction of the chip so the median is robustly "sea").
    background = float(np.median(db[valid]))
    mask = (db > background + THRESHOLD_DB) & valid

    from scipy import ndimage

    labels, n = ndimage.label(mask)
    if n == 0:
        return None
    # Component nearest the chip centre (the detection point), within a
    # small search radius — otherwise we'd happily measure a neighbour.
    cy = cx = CHIP_PX // 2
    yy, xx = np.mgrid[max(0, cy - SEARCH_RADIUS_PX):cy + SEARCH_RADIUS_PX + 1,
                      max(0, cx - SEARCH_RADIUS_PX):cx + SEARCH_RADIUS_PX + 1]
    near = labels[yy, xx]
    near = near[near > 0]
    if near.size == 0:
        return None
    target = np.bincount(near).argmax()
    ys, xs = np.nonzero(labels == target)

    # Ground sampling distance, metres per pixel, at this latitude
    mpp_x = abs(vrt.transform.a) * 111_320.0 * math.cos(math.radians(lat))
    mpp_y = abs(vrt.transform.e) * 110_540.0

    # PCA in metric space (lon/lat pixels are anisotropic in metres)
    pts = np.column_stack([xs * mpp_x, ys * mpp_y]).astype("float64")
    pts -= pts.mean(axis=0)
    if len(pts) < 3:
        return None
    cov = np.cov(pts.T)
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, np.argmax(evals)]
    proj_major = pts @ major
    minor = evecs[:, np.argmin(evals)]
    proj_minor = pts @ minor
    # Full extent, padded half a pixel per end (blob edges are pixel centres)
    pad = (mpp_x + mpp_y) / 2
    length = float(proj_major.max() - proj_major.min()) + pad
    width = float(proj_minor.max() - proj_minor.min()) + pad

    # Heading: angle of major axis, geographic convention (N=0°, E=90°),
    # folded to [0, 180) — SAR can't tell bow from stern.
    heading = math.degrees(math.atan2(major[0], -major[1])) % 180.0

    touches_edge = (ys.min() == 0 or xs.min() == 0
                    or ys.max() == CHIP_PX - 1 or xs.max() == CHIP_PX - 1)
    reliable = not touches_edge and length <= MAX_SHIP_M

    # Oriented footprint rectangle in geographic coords: centre of the blob
    # ± half-extents along the principal axes, converted metre→pixel→lonlat.
    cy_blob = float(ys.mean())
    cx_blob = float(xs.mean())
    mid_major = (proj_major.max() + proj_major.min()) / 2
    mid_minor = (proj_minor.max() + proj_minor.min()) / 2
    corners = []
    for sa, sb in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
        off_m = ((mid_major + sa * length / 2) * major
                 + (mid_minor + sb * width / 2) * minor)  # metres (x, y)
        px = cx_blob + off_m[0] / mpp_x
        py = cy_blob + off_m[1] / mpp_y
        glon, glat = vrt.transform * (c0 + px, r0 + py)
        corners.append((round(glon, 6), round(glat, 6)))

    return SizeEstimate(length, width, heading, reliable, corners)
