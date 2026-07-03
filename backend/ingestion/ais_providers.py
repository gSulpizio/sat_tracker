"""AIS data providers — the pipeline asks for pings by (bbox, start, end)
and never cares where they come from.

Providers:
  * StoreAISProvider — reads the local store continuously fed by the
    aisstream.io collector daemon (`python -m backend.ingestion.ais_collector`).
    The right pattern for free data: live AIS cannot be queried retroactively,
    so you record it and query the recording at fusion time.
  * RestAISProvider — one HTTP call to a historical AIS API (Datalastic,
    Spire, an in-house endpoint …) via a configurable URL template.
  * FileAISProvider — manual CSV/JSON upload, kept as a fallback.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from backend.config import Settings
from backend.ingestion.ais_loader import COLUMN_ALIASES, REQUIRED_COLUMNS, load_ais

log = logging.getLogger(__name__)

BBox = tuple[float, float, float, float]


def empty_ais() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mmsi": pd.Series(dtype="int64"),
            "ts": pd.Series(dtype="datetime64[ns, UTC]"),
            "lat": pd.Series(dtype="float64"),
            "lon": pd.Series(dtype="float64"),
            "sog_knots": pd.Series(dtype="float64"),
            "cog_deg": pd.Series(dtype="float64"),
        }
    )


def _clip(df: pd.DataFrame, bbox: BBox, start: datetime, end: datetime) -> pd.DataFrame:
    if df.empty:
        return df
    min_lon, min_lat, max_lon, max_lat = bbox
    mask = (
        df["ts"].between(pd.Timestamp(start), pd.Timestamp(end))
        & df["lon"].between(min_lon, max_lon)
        & df["lat"].between(min_lat, max_lat)
    )
    return df.loc[mask].reset_index(drop=True)


class AISProvider(ABC):
    """Contract: canonical ping DataFrame for a space-time box."""

    @abstractmethod
    def fetch(self, bbox: BBox, start: datetime, end: datetime) -> pd.DataFrame:
        ...


class FileAISProvider(AISProvider):
    """Wraps a manually supplied CSV/JSON export."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.df = load_ais(self.path)

    def coverage(self) -> tuple[datetime, datetime]:
        return self.df["ts"].min().to_pydatetime(), self.df["ts"].max().to_pydatetime()

    def fetch(self, bbox: BBox, start: datetime, end: datetime) -> pd.DataFrame:
        return _clip(self.df, bbox, start, end)


class StoreAISProvider(AISProvider):
    """Reads the local ping store written by the aisstream collector daemon
    (one canonical-schema CSV per UTC day: data/ais_store/YYYY-MM-DD.csv).
    """

    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)

    def fetch(self, bbox: BBox, start: datetime, end: datetime) -> pd.DataFrame:
        if not self.store_dir.exists():
            raise FileNotFoundError(
                f"AIS store {self.store_dir} does not exist. Start the "
                "collector first: python -m backend.ingestion.ais_collector"
            )
        frames = []
        day = start.date()
        while day <= end.date():
            f = self.store_dir / f"{day:%Y-%m-%d}.csv"
            if f.exists():
                frames.append(load_ais(f))
            day += timedelta(days=1)
        if not frames:
            log.warning("AIS store has no files covering %s → %s", start, end)
            return empty_ais()
        return _clip(pd.concat(frames, ignore_index=True), bbox, start, end)


class RestAISProvider(AISProvider):
    """Generic historical AIS REST API.

    `url_template` is formatted with: {min_lon} {min_lat} {max_lon} {max_lat}
    {start_iso} {end_iso}. The JSON response (a list of pings, or an object
    whose 'data'/'results' key holds one) is normalized through the same
    column aliases as file imports — covering Datalastic-style and in-house
    endpoints without provider-specific code.
    """

    def __init__(self, url_template: str, api_key: str = ""):
        if not url_template:
            raise ValueError(
                "AIS provider 'rest' needs an API URL template "
                "(AIS_API_URL or the app field), e.g. "
                "https://api.example.com/ais?bbox={min_lon},{min_lat},{max_lon},"
                "{max_lat}&from={start_iso}&to={end_iso}"
            )
        if "example.com" in url_template:
            raise ValueError(
                "The AIS REST URL is still the documentation placeholder "
                "(api.example.com). Enter your actual provider's endpoint — "
                "historical AIS REST APIs are commercial (e.g. Datalastic, "
                "Spire, MarineTraffic). Free alternatives: switch the AIS "
                "provider to 'store' and run the aisstream.io collector "
                "(records live traffic for future passes), or upload an AIS "
                "file export covering the scene times."
            )
        self.url_template = url_template
        self.api_key = api_key

    def fetch(self, bbox: BBox, start: datetime, end: datetime) -> pd.DataFrame:
        import requests

        url = self.url_template.format(
            min_lon=bbox[0], min_lat=bbox[1], max_lon=bbox[2], max_lat=bbox[3],
            start_iso=start.isoformat(), end_iso=end.isoformat(),
            api_key=self.api_key,
        )
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        body = resp.json()
        records = body if isinstance(body, list) else (
            body.get("data") or body.get("results") or []
        )
        if not records:
            log.warning("AIS API returned no pings for %s → %s", start, end)
            return empty_ais()

        df = pd.DataFrame(records).rename(columns=COLUMN_ALIASES)
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"AIS API response missing fields {sorted(missing)} after "
                f"alias mapping — got columns {list(df.columns)}"
            )
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df["mmsi"] = df["mmsi"].astype("int64")
        df = df.dropna(subset=["lat", "lon"]).drop_duplicates(subset=["mmsi", "ts"])
        return _clip(df[REQUIRED_COLUMNS], bbox, start, end)


def build_ais_provider(cfg: Settings, ais_file: str | None = None) -> AISProvider:
    """An uploaded file always wins; otherwise the configured API provider."""
    if ais_file:
        return FileAISProvider(ais_file)
    if cfg.ais_provider == "store":
        return StoreAISProvider(cfg.ais_store_dir)
    if cfg.ais_provider == "rest":
        return RestAISProvider(cfg.ais_api_url, cfg.ais_api_key)
    if cfg.ais_provider == "file":
        raise ValueError("AIS provider 'file' selected but no file was supplied.")
    raise ValueError(f"Unknown AIS provider: {cfg.ais_provider!r}")
