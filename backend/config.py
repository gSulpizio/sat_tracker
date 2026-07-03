"""Environment-driven configuration for the sat_tracker pipeline.

Precedence: saved user config (data/user_config.json, written by the app on
every run) > environment variables > built-in defaults. The saved file lives
under data/, which is git-ignored, and is written with 0600 permissions
because it may contain credentials.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SNAPSHOTS_ROOT = DATA_DIR / "snapshots"
AIS_STORE_DIR = DATA_DIR / "ais_store"
USER_CONFIG_PATH = DATA_DIR / "user_config.json"
LOCATIONS_PATH = DATA_DIR / "locations.json"


def aoi_slug(bbox: tuple[float, float, float, float]) -> str:
    """Stable, filesystem-safe key for an AOI — used to give each area its
    own snapshot folder so switching locations never deletes another area's
    cached imagery/detections. Independent of whether the AOI has been
    saved as a named location: any distinct box gets its own folder.
    """
    import hashlib

    rounded = tuple(round(float(x), 4) for x in bbox)
    digest = hashlib.sha1(repr(rounded).encode()).hexdigest()[:8]
    return f"aoi_{digest}"


@dataclass(frozen=True)
class Settings:
    # "live" (default): real Sentinel-1 scenes + real AIS data required.
    # "simulate": synthetic scene/AIS for testing — every output is flagged
    # `simulated: true` and the UI shows a large SIMULATED banner.
    mode: str = os.getenv("SAT_TRACKER_MODE", "live")

    # STAC catalog. Default: Copernicus Data Space. The reader needs a
    # geocoded COG collection (e.g. a Sentinel-1 RTC collection) — raw GRD
    # SAFE archives must be terrain-corrected first.
    stac_url: str = os.getenv(
        "STAC_URL", "https://catalogue.dataspace.copernicus.eu/stac"
    )
    stac_collection: str = os.getenv("S1_STAC_COLLECTION", "sentinel-1-grd")

    cdse_token_url: str = os.getenv(
        "CDSE_TOKEN_URL",
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
        "/protocol/openid-connect/token",
    )
    cdse_client_id: str = os.getenv("CDSE_CLIENT_ID", "")
    cdse_client_secret: str = os.getenv("CDSE_CLIENT_SECRET", "")
    # CDSE S3 keys (separate from OAuth!) for streaming COG windows from
    # s3://eodata — generate at https://eodata-s3keysmanager.dataspace.copernicus.eu
    cdse_s3_key: str = os.getenv("CDSE_S3_KEY", "")
    cdse_s3_secret: str = os.getenv("CDSE_S3_SECRET", "")

    # AIS source: "store" (local store fed by the aisstream collector),
    # "rest" (historical AIS REST API), "file" (manual upload fallback).
    ais_provider: str = os.getenv("AIS_PROVIDER", "store")
    ais_api_url: str = os.getenv("AIS_API_URL", "")     # REST URL template
    ais_api_key: str = os.getenv("AIS_API_KEY", "")     # REST bearer / aisstream key
    ais_store_dir: Path = AIS_STORE_DIR

    # Detection backend: "yolo" | "roboflow" | "vertex" ("mock": simulate only)
    detector_backend: str = os.getenv("DETECTOR_BACKEND", "yolo")
    yolo_weights: str = os.getenv("YOLO_WEIGHTS", "models/yolov8m-obb-sar.pt")
    roboflow_api_key: str = os.getenv("ROBOFLOW_API_KEY", "")
    # 'project/version' — e.g. the public Universe model below, or your own
    roboflow_model_id: str = os.getenv("ROBOFLOW_MODEL_ID", "sar-ship-dataset-ogvbz/1")

    # Sentinel-2 optical context layer (nearest cloud-free scene per pass)
    include_s2: bool = os.getenv("INCLUDE_S2", "1") == "1"
    s2_max_cloud: float = float(os.getenv("S2_MAX_CLOUD", "40"))

    # Measure each detection's physical size (length/beam/heading) from a
    # native-resolution chip after detection — a handful of small extra S3
    # window reads per pass, no API cost.
    measure_lengths: bool = os.getenv("MEASURE_LENGTHS", "1") == "1"

    # Fusion parameters
    temporal_window_s: int = int(os.getenv("FUSION_TEMPORAL_WINDOW_S", "300"))  # ±5 min
    match_gate_m: float = float(os.getenv("FUSION_MATCH_GATE_M", "1000"))       # 1 km

    # Scene search: how far back to look and how many passes to process.
    # S1 revisit is 1–3 days at mid-latitudes; 7 days gives safe margin.
    search_days: int = int(os.getenv("SEARCH_DAYS", "7"))
    max_scenes: int = int(os.getenv("MAX_SCENES", "5"))

    # Default area of interest (Strait of Gibraltar approaches) —
    # (min_lon, min_lat, max_lon, max_lat); edit freely in the app.
    # Overridable via AOI_BBOX="min_lon,min_lat,max_lon,max_lat" — mainly
    # for the standalone AIS collector process/container, which has no
    # user_config.json of its own to read the app's active location from.
    aoi_bbox: tuple[float, float, float, float] = field(
        default_factory=lambda: (
            tuple(float(x) for x in os.environ["AOI_BBOX"].split(","))
            if os.getenv("AOI_BBOX") else (-6.20, 35.75, -5.30, 36.20)
        )
    )

    @property
    def snapshots_dir(self) -> Path:
        """Per-AOI snapshot folder — isolates each location's imagery,
        detections, and overlays so switching locations is instant and
        never deletes another area's cached data (see aoi_slug)."""
        return SNAPSHOTS_ROOT / aoi_slug(self.aoi_bbox)


settings = Settings()

# User-editable fields persisted between sessions (everything the app's
# config panel exposes — deliberately not the derived Path fields).
_PERSISTED_FIELDS = (
    "mode", "stac_url", "stac_collection",
    "cdse_client_id", "cdse_client_secret",
    "cdse_s3_key", "cdse_s3_secret",
    "ais_provider", "ais_api_url", "ais_api_key",
    "detector_backend", "yolo_weights",
    "roboflow_api_key", "roboflow_model_id",
    "temporal_window_s", "match_gate_m",
    "search_days", "max_scenes", "aoi_bbox",
    "include_s2", "s2_max_cloud", "measure_lengths",
)


def save_user_settings(cfg: Settings) -> Path:
    """Persist the run configuration so the next session starts from it.

    Written under data/ (git-ignored) with 0600 permissions — the file may
    contain API credentials and must never end up in version control.
    """
    DATA_DIR.mkdir(exist_ok=True)
    payload = {f: getattr(cfg, f) for f in _PERSISTED_FIELDS}
    payload["aoi_bbox"] = list(cfg.aoi_bbox)
    USER_CONFIG_PATH.write_text(json.dumps(payload, indent=2))
    USER_CONFIG_PATH.chmod(0o600)
    return USER_CONFIG_PATH


def load_user_settings() -> Settings:
    """Env-based settings overlaid with the saved user config, if present.
    A corrupt or partial file degrades gracefully to env defaults."""
    if not USER_CONFIG_PATH.exists():
        return settings
    try:
        data = json.loads(USER_CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return settings
    known = {k: v for k, v in data.items() if k in _PERSISTED_FIELDS}
    if "aoi_bbox" in known:
        known["aoi_bbox"] = tuple(float(x) for x in known["aoi_bbox"])
    # Migrate values saved by older versions whose defaults were wrong —
    # CDSE's actual collection id is lowercase 'sentinel-1-grd'.
    if known.get("stac_collection") == "SENTINEL-1":
        known["stac_collection"] = "sentinel-1-grd"
    try:
        return replace(settings, **known)
    except (TypeError, ValueError):
        return settings


def has_user_settings() -> bool:
    return USER_CONFIG_PATH.exists()


def clear_user_settings() -> None:
    USER_CONFIG_PATH.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Named location presets — bookmarked AOIs an analyst can switch between
# (e.g. "Gibraltar", "Hormuz", "Long Beach") without redrawing the box.
# ---------------------------------------------------------------------------

def load_locations() -> dict[str, tuple[float, float, float, float]]:
    if not LOCATIONS_PATH.exists():
        return {}
    try:
        data = json.loads(LOCATIONS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    out = {}
    for name, bbox in data.items():
        try:
            out[name] = tuple(float(x) for x in bbox)
        except (TypeError, ValueError):
            continue
    return out


def save_location(name: str, bbox: tuple[float, float, float, float]) -> None:
    name = name.strip()
    if not name:
        raise ValueError("Location name cannot be empty.")
    DATA_DIR.mkdir(exist_ok=True)
    locations = load_locations()
    locations[name] = tuple(bbox)
    LOCATIONS_PATH.write_text(
        json.dumps({k: list(v) for k, v in locations.items()}, indent=2)
    )


def delete_location(name: str) -> None:
    locations = load_locations()
    if locations.pop(name, None) is not None:
        LOCATIONS_PATH.write_text(
            json.dumps({k: list(v) for k, v in locations.items()}, indent=2)
        )
