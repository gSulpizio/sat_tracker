"""Sentinel-1 scene access via STAC + windowed Cloud-Optimized GeoTIFF reads.

Three-step API so the pipeline can process a *series* of passes:
    refs   = search_scenes(bbox, start, end)      # cheap metadata search
    passes = group_into_passes(refs)              # group same-overpass frames
    scene  = load_scene(passes[i], bbox)          # mosaic + windowed COG read

Only the AOI pixels are transferred (rasterio range-reads over /vsicurl/ or
/vsis3/), and large windows are decimated to a display/detection-friendly
resolution.

Georeferencing: assets already in EPSG:4326 are read directly; anything else
— including CDSE's GRD-COG products, which carry ground control points
instead of a map grid — is warped on the fly via a WarpedVRT. Over ocean
(no terrain) GCP warping is accurate to tens of metres, far inside the 1 km
match gate; for land-adjacent precision work prefer a terrain-corrected
(RTC) collection.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from backend.config import Settings, settings

log = logging.getLogger(__name__)

BBox = tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
MAX_READ_PX = 2048  # decimate reads so a large AOI stays tractable


@dataclass
class SceneRef:
    """Lightweight result of a STAC search — enough to list, sort, and load."""

    scene_id: str
    platform: str
    product_type: str
    acquired_at: datetime
    asset_href: str
    simulated: bool = False


@dataclass
class Scene:
    """A loaded acquisition with everything the pipeline needs downstream."""

    scene_id: str
    platform: str
    product_type: str
    acquired_at: datetime           # T_img — drives the AIS temporal window
    bbox: BBox
    crs: str                        # CRS of `image` pixel grid
    image: np.ndarray               # 2-D float array, SAR backscatter in dB
    asset_href: str

    @property
    def transform_params(self) -> tuple[float, float, float, float]:
        """(lon0, lat0, lon_per_px, lat_per_px) for a north-up EPSG:4326 grid."""
        h, w = self.image.shape
        min_lon, min_lat, max_lon, max_lat = self.bbox
        return min_lon, max_lat, (max_lon - min_lon) / w, -(max_lat - min_lat) / h

    def pixel_to_lonlat(self, row: float, col: float) -> tuple[float, float]:
        lon0, lat0, dlon, dlat = self.transform_params
        return lon0 + col * dlon, lat0 + row * dlat

    def lonlat_to_pixel(self, lon: float, lat: float) -> tuple[int, int]:
        lon0, lat0, dlon, dlat = self.transform_params
        return int((lat - lat0) / dlat), int((lon - lon0) / dlon)


def search_scenes(
    bbox: BBox,
    start: datetime,
    end: datetime,
    cfg: Settings | None = None,
) -> list[SceneRef]:
    """All scenes intersecting bbox in [start, end], oldest → newest.

    Simulate mode fabricates a series of 3 passes spread over the search
    window (capped at the last 40 min so the synthetic fleet stays inside the
    AOI while sailing between passes) so multi-pass navigation is testable
    offline. Real S1 revisit is 1–3 days; the short spacing is purely for
    demo convenience.
    """
    cfg = cfg or settings
    if cfg.mode == "simulate":
        span_start = max(start, end - timedelta(minutes=40))
        times = [span_start + (end - span_start) * k / 2 for k in range(3)]
        return [
            SceneRef(
                scene_id=f"SIMULATED_S1_GRD_{t:%Y%m%dT%H%M%S}",
                platform="SIMULATED sentinel-1",
                product_type="GRD (simulated)",
                acquired_at=t.astimezone(timezone.utc),
                asset_href="simulated://sentinel-1/grd",
                simulated=True,
            )
            for t in times
        ]

    from pystac_client import Client

    catalog = Client.open(cfg.stac_url)
    search = catalog.search(
        collections=[cfg.stac_collection],
        bbox=bbox,
        datetime=f"{start.isoformat()}/{end.isoformat()}",
        max_items=50,
    )
    refs = []
    for item in search.items():
        asset = next(
            (a for k, a in item.assets.items() if "vv" in k.lower()),
            next(iter(item.assets.values())),
        )
        refs.append(
            SceneRef(
                scene_id=item.id,
                platform=item.properties.get("platform", "sentinel-1"),
                product_type=item.properties.get("sar:product_type", "GRD"),
                acquired_at=datetime.fromisoformat(
                    item.properties["datetime"].replace("Z", "+00:00")
                ),
                asset_href=asset.href,
            )
        )
    refs.sort(key=lambda r: r.acquired_at)
    log.info("STAC: %d scenes in %s → %s", len(refs), start.isoformat(), end.isoformat())
    return refs


def group_into_passes(refs: list[SceneRef], max_gap_s: float = 120) -> list[list[SceneRef]]:
    """Group scene files belonging to the same overpass.

    CDSE splits one Sentinel-1 overpass into consecutive IW frame files a
    few tens of seconds apart along the orbit track; the next overpass over
    the same area is ~99 minutes later (one orbit). A 120s gap threshold
    cleanly separates frames-of-one-pass from the next pass. `refs` must
    already be sorted by acquired_at (as returned by search_scenes).
    """
    if not refs:
        return []
    groups: list[list[SceneRef]] = [[refs[0]]]
    for ref in refs[1:]:
        if (ref.acquired_at - groups[-1][-1].acquired_at).total_seconds() <= max_gap_s:
            groups[-1].append(ref)
        else:
            groups.append([ref])
    return groups


def load_scene(refs: list[SceneRef], bbox: BBox, cfg: Settings | None = None) -> Scene:
    """Read and mosaic every frame of one overpass onto a single AOI canvas.

    A Sentinel-1 IW frame's footprint is a rotated parallelogram (the orbit
    track isn't north-south), so a single frame typically covers only a
    diagonal slice of a north-up AOI box — for an AOI wider than one frame,
    or straddling a frame boundary, that leaves most of the box empty.
    Mosaicking every frame of the pass (grouped by group_into_passes) fills
    the box fully, mirroring the approach used for the Sentinel-2 overlay.
    """
    cfg = cfg or settings
    if refs[0].simulated:
        return _simulate_scene(refs[0], bbox)

    import rasterio
    from rasterio.errors import WindowError
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window, from_bounds
    from rasterio.windows import bounds as window_bounds

    min_lon, min_lat, max_lon, max_lat = bbox
    canvas_w = MAX_READ_PX
    canvas_h = max(1, int(canvas_w * (max_lat - min_lat) / (max_lon - min_lon)))
    canvas = np.zeros((canvas_h, canvas_w), dtype="float32")  # 0.0 dB == nodata

    for ref in refs:
        log.info("Reading COG window from %s", ref.asset_href)
        with rasterio.Env(**_gdal_env(ref.asset_href, cfg)):
            with rasterio.open(ref.asset_href) as src:
                gcps, gcp_crs = src.gcps
                if src.crs is None and gcps:
                    # CDSE GRD-COGs are georeferenced via ground control
                    # points, not a north-up grid. Warp on the fly to
                    # EPSG:4326 — over ocean (no terrain) GCP warping is
                    # accurate to tens of metres, far inside the 1 km gate.
                    vrt_src = WarpedVRT(src, src_crs=gcp_crs, crs="EPSG:4326")
                elif src.crs is not None and src.crs.to_epsg() != 4326:
                    vrt_src = WarpedVRT(src, crs="EPSG:4326")
                else:
                    vrt_src = src
                try:
                    window = from_bounds(*bbox, transform=vrt_src.transform)
                    try:
                        # WarpedVRT forbids boundless reads — clip to the
                        # frame extent; frames not overlapping this AOI at
                        # all are simply skipped.
                        window = window.intersection(
                            Window(0, 0, vrt_src.width, vrt_src.height)
                        ).round_offsets().round_lengths()
                    except WindowError:
                        continue
                    if window.height < 1 or window.width < 1:
                        continue
                    left, bottom, right, top = window_bounds(window, vrt_src.transform)
                    c0 = max(0, int((left - min_lon) / (max_lon - min_lon) * canvas_w))
                    c1 = min(canvas_w, int((right - min_lon) / (max_lon - min_lon) * canvas_w))
                    r0 = max(0, int((max_lat - top) / (max_lat - min_lat) * canvas_h))
                    r1 = min(canvas_h, int((max_lat - bottom) / (max_lat - min_lat) * canvas_h))
                    if r1 <= r0 or c1 <= c0:
                        continue
                    amplitude = vrt_src.read(
                        1, window=window, out_shape=(r1 - r0, c1 - c0),
                        out_dtype="float32",
                    )
                finally:
                    if vrt_src is not src:
                        vrt_src.close()

        # Amplitude → dB; only paste pixels the frame actually recorded
        # (amplitude<=0 at swath/frame edges is nodata, kept transparent).
        db = 10.0 * np.log10(np.maximum(amplitude, 1.0) ** 2)
        valid = amplitude > 0
        canvas_slice = canvas[r0:r1, c0:c1]
        canvas_slice[valid] = db[valid]

    if not canvas.any():
        raise ValueError(
            f"AOI {bbox} does not overlap any frame of this pass "
            f"({', '.join(r.scene_id for r in refs)})."
        )

    first = refs[0]
    combined_id = first.scene_id if len(refs) == 1 else f"{first.scene_id}+{len(refs) - 1}"
    return Scene(
        scene_id=combined_id,
        platform=first.platform,
        product_type=first.product_type,
        acquired_at=first.acquired_at,
        bbox=bbox,
        crs="EPSG:4326",
        image=canvas,
        asset_href=first.asset_href,
    )


# --------------------------------------------------------------------------
# Sentinel-2 optical context layer
# --------------------------------------------------------------------------

@dataclass
class S2Scene:
    scene_id: str
    acquired_at: datetime
    cloud_cover: float
    bbox: BBox                     # clipped to actual data
    rgba: np.ndarray               # (H, W, 4) uint8 true-color + alpha


def fetch_s2_rgb(
    bbox: BBox,
    around: datetime,
    cfg: Settings | None = None,
    max_cloud: float = 40.0,
    window_days: int = 5,
) -> S2Scene | None:
    """Nearest-in-time, sufficiently cloud-free Sentinel-2 L2A true-color
    image for the AOI, within ±window_days of `around` (the S1 pass time).

    Context layer only: S2 revisits every ~5 days and never at the same
    instant as S1, so ships visible here are NOT the ships in the radar
    scene — use it for coastline, ports, and confirming static targets.
    Returns None when nothing acceptable exists.
    """
    from pystac_client import Client
    import rasterio
    from rasterio.errors import WindowError
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window, from_bounds
    from rasterio.windows import bounds as window_bounds

    cfg = cfg or settings
    catalog = Client.open(cfg.stac_url)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=(f"{(around - timedelta(days=window_days)).isoformat()}/"
                  f"{(around + timedelta(days=window_days)).isoformat()}"),
        max_items=50,
    )
    items = list(search.items())

    def cloud(it) -> float:
        return float(it.properties.get("eo:cloud_cover",
                                       it.properties.get("cloudCover", 100.0)))

    usable = [it for it in items if cloud(it) <= max_cloud]
    if not usable:
        log.info("No S2 scene ≤%.0f%% cloud within ±%dd of %s (%d candidates)",
                 max_cloud, window_days, around.isoformat(), len(items))
        return None
    def when(it) -> datetime:
        return datetime.fromisoformat(it.properties["datetime"].replace("Z", "+00:00"))

    def tci_asset(it):
        return (it.assets.get("TCI_10m") or it.assets.get("TCI_20m")
                or next((a for k, a in it.assets.items()
                         if "tci" in k.lower() or "visual" in k.lower()), None))

    best = min(usable, key=lambda it: abs((when(it) - around).total_seconds()))
    best_when = when(best)
    # An S2 pass is delivered as ~110 km tiles; an AOI can straddle several.
    # Mosaic every usable tile of the same overflight onto one AOI canvas.
    group = [it for it in usable
             if abs((when(it) - best_when).total_seconds()) < 900]

    min_lon, min_lat, max_lon, max_lat = bbox
    canvas_w = MAX_READ_PX
    canvas_h = max(1, int(canvas_w * (max_lat - min_lat) / (max_lon - min_lon)))
    canvas = np.zeros((canvas_h, canvas_w, 4), np.uint8)

    for item in group:
        asset = tci_asset(item)
        if asset is None:
            continue
        try:
            with rasterio.Env(**_gdal_env(asset.href, cfg)):
                with rasterio.open(asset.href) as src:
                    vrt = (WarpedVRT(src, crs="EPSG:4326")
                           if src.crs is not None and src.crs.to_epsg() != 4326
                           else src)
                    try:
                        window = from_bounds(*bbox, transform=vrt.transform)
                        try:
                            window = window.intersection(
                                Window(0, 0, vrt.width, vrt.height)
                            ).round_offsets().round_lengths()
                        except WindowError:
                            continue
                        if window.height < 1 or window.width < 1:
                            continue
                        left, bottom, right, top = window_bounds(window, vrt.transform)
                        c0 = max(0, int((left - min_lon) / (max_lon - min_lon) * canvas_w))
                        c1 = min(canvas_w, int((right - min_lon) / (max_lon - min_lon) * canvas_w))
                        r0 = max(0, int((max_lat - top) / (max_lat - min_lat) * canvas_h))
                        r1 = min(canvas_h, int((max_lat - bottom) / (max_lat - min_lat) * canvas_h))
                        if r1 <= r0 or c1 <= c0:
                            continue
                        data = vrt.read([1, 2, 3], window=window,
                                        out_shape=(3, r1 - r0, c1 - c0))
                    finally:
                        if vrt is not src:
                            vrt.close()
            rgb = np.transpose(data, (1, 2, 0)).astype(np.uint8)
            has_data = rgb.max(axis=2) > 0
            canvas[r0:r1, c0:c1, :3][has_data] = rgb[has_data]
            canvas[r0:r1, c0:c1, 3][has_data] = 255
        except Exception as exc:
            log.warning("S2 tile %s read failed: %s", item.id, exc)

    if canvas[..., 3].max() == 0:
        return None
    log.info("S2 context: %s (+%d sibling tiles, %.1f%% cloud) from %s",
             best.id, len(group) - 1, cloud(best), best_when.isoformat())
    return S2Scene(
        scene_id=best.id,
        acquired_at=best_when,
        cloud_cover=cloud(best),
        bbox=bbox,
        rgba=canvas,
    )


# --------------------------------------------------------------------------
# Simulation path (testing only) — fabricated but geospatially consistent
# scene, unmistakably labeled SIMULATED in scene_id and platform.
# --------------------------------------------------------------------------

def _simulate_scene(ref: SceneRef, bbox: BBox, size: int = 900) -> Scene:
    """Synthetic SAR GRD chip: multiplicative gamma speckle over a low ocean
    backscatter floor, seeded per acquisition time so each pass differs.
    Vessel signatures are injected by the pipeline so imagery and ground
    truth stay in sync.
    """
    rng = np.random.default_rng(int(ref.acquired_at.timestamp()) % 2**32)
    ocean_db = -22.0  # typical calm-sea sigma0 for S1 VV
    speckle = rng.gamma(shape=4.0, scale=1.0 / 4.0, size=(size, size))
    image = ocean_db + 10.0 * np.log10(speckle)

    return Scene(
        scene_id=ref.scene_id,
        platform=ref.platform,
        product_type=ref.product_type,
        acquired_at=ref.acquired_at,
        bbox=bbox,
        crs="EPSG:4326",
        image=image.astype(np.float32),
        asset_href=ref.asset_href,
    )


def inject_vessel_signatures(scene: Scene, lonlats: list[tuple[float, float]]) -> None:
    """Paint bright point targets (with a small cross-shaped sidelobe pattern,
    as real ships appear in SAR) into a simulated scene at the given positions.
    """
    h, w = scene.image.shape
    for lon, lat in lonlats:
        r, c = scene.lonlat_to_pixel(lon, lat)
        if not (2 <= r < h - 2 and 2 <= c < w - 2):
            continue
        scene.image[r - 1 : r + 2, c - 1 : c + 2] = 8.0     # bright core
        scene.image[r, max(0, c - 4) : c + 5] += 12.0        # range sidelobes
        scene.image[max(0, r - 4) : r + 5, c] += 12.0        # azimuth sidelobes


def _gdal_env(href: str, cfg: Settings) -> dict:
    """GDAL/rasterio session options for the asset's access scheme.

    * s3://eodata/… (Copernicus Data Space) needs CDSE **S3 keys** — a
      separate credential from the OAuth client, generated at
      https://eodata-s3keysmanager.dataspace.copernicus.eu
    * https:// assets on CDSE use an OAuth bearer token.
    """
    if href.startswith("s3://"):
        if not (cfg.cdse_s3_key and cfg.cdse_s3_secret):
            raise ValueError(
                "This asset is streamed from CDSE S3 (s3://eodata/…), which "
                "needs S3 keys — a DIFFERENT credential from the OAuth "
                "client id/secret. Generate a key pair at "
                "https://eodata-s3keysmanager.dataspace.copernicus.eu and "
                "enter it as 'CDSE S3 key/secret' in the app (or set "
                "CDSE_S3_KEY / CDSE_S3_SECRET)."
            )
        # rasterio disallows AWS_* as Env options (reserved for boto3
        # sessions), but GDAL reads them fine from the process environment —
        # and this avoids a boto3 dependency for a non-AWS S3 endpoint.
        import os

        os.environ.update({
            "AWS_ACCESS_KEY_ID": cfg.cdse_s3_key,
            "AWS_SECRET_ACCESS_KEY": cfg.cdse_s3_secret,
            "AWS_S3_ENDPOINT": "eodata.dataspace.copernicus.eu",
            "AWS_VIRTUAL_HOSTING": "FALSE",
            "AWS_HTTPS": "YES",
        })
        return {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}
    env = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}
    if cfg.cdse_client_id:
        env["GDAL_HTTP_HEADERS"] = _auth_header(cfg)
    return env


def _auth_header(cfg: Settings) -> str:
    """OAuth2 client-credentials token for CDSE asset downloads."""
    import requests

    resp = requests.post(
        cfg.cdse_token_url,
        data={
            "grant_type": "client_credentials",
            "client_id": cfg.cdse_client_id,
            "client_secret": cfg.cdse_client_secret,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return f"Authorization: Bearer {resp.json()['access_token']}"
