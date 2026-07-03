"""Health checks for every external data source the pipeline depends on.

Used by the dashboard's "📡 Data source status" panel so an analyst can
tell at a glance whether a run of all-dark detections means "dark fleet"
or "my AIS feed is down".
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend.config import Settings

AISSTREAM_URL = "wss://stream.aisstream.io/v0/stream"


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    latency_s: float | None = None


def store_coverage(cfg: Settings) -> dict | None:
    """Fast summary of the local AIS store: first/last ping, count.
    Reads only the head/tail lines of each day file — no pandas."""
    store = Path(cfg.ais_store_dir)
    files = sorted(store.glob("*.csv")) if store.exists() else []
    if not files:
        return None
    pings = 0
    first_ts = last_ts = None
    for f in files:
        lines = f.read_text().splitlines()
        rows = [ln for ln in lines[1:] if ln.strip()]
        pings += len(rows)
        if rows:
            if first_ts is None:
                first_ts = _norm_ts(rows[0].split(",")[1])
            last_ts = _norm_ts(rows[-1].split(",")[1])
    if pings == 0:
        return None
    return {"start": first_ts, "end": last_ts, "pings": pings, "files": len(files)}


def _norm_ts(ts: str) -> str:
    """Normalize store timestamps (incl. legacy Go-format rows) to ISO 8601
    so string comparison against scene times is order-correct."""
    import re

    s = re.sub(r"(\.\d{6})\d+", r"\1", ts.strip()).replace(" +0000 UTC", "+00:00")
    if len(s) > 10 and s[10] == " ":
        s = s[:10] + "T" + s[11:]
    return s


def check_stac(cfg: Settings) -> CheckResult:
    t0 = time.monotonic()
    try:
        from pystac_client import Client

        catalog = Client.open(cfg.stac_url)
        catalog.get_collection(cfg.stac_collection)
        return CheckResult(
            "STAC catalog", True,
            f"collection '{cfg.stac_collection}' reachable at {cfg.stac_url}",
            time.monotonic() - t0,
        )
    except Exception as exc:
        return CheckResult("STAC catalog", False,
                           f"{type(exc).__name__}: {str(exc)[:160]}",
                           time.monotonic() - t0)


def check_imagery(cfg: Settings) -> CheckResult:
    """Opens the newest scene's COG header — proves S3 keys actually work."""
    t0 = time.monotonic()
    if not (cfg.cdse_s3_key and cfg.cdse_s3_secret):
        return CheckResult(
            "Imagery (CDSE S3)", False,
            "S3 keys not set — generate at eodata-s3keysmanager.dataspace.copernicus.eu",
        )
    try:
        from backend.ingestion.stac_client import _gdal_env, search_scenes
        import rasterio

        end = datetime.now(tz=timezone.utc)
        refs = search_scenes(cfg.aoi_bbox, end - timedelta(days=cfg.search_days),
                             end, cfg=cfg)
        if not refs:
            return CheckResult(
                "Imagery (CDSE S3)", True,
                f"keys set; no scene in the last {cfg.search_days}d to probe",
                time.monotonic() - t0,
            )
        ref = refs[-1]
        with rasterio.Env(**_gdal_env(ref.asset_href, cfg)):
            with rasterio.open(ref.asset_href) as src:
                dims = f"{src.width}×{src.height}"
        return CheckResult(
            "Imagery (CDSE S3)", True,
            f"opened {ref.scene_id[:32]}… ({dims}) — keys valid",
            time.monotonic() - t0,
        )
    except Exception as exc:
        return CheckResult("Imagery (CDSE S3)", False,
                           f"{type(exc).__name__}: {str(exc)[:160]}",
                           time.monotonic() - t0)


def check_ais(cfg: Settings, stream_probe_s: int = 15) -> list[CheckResult]:
    results = []
    if cfg.ais_provider == "store":
        cov = store_coverage(cfg)
        if cov is None:
            results.append(CheckResult(
                "AIS store", False,
                f"no pings recorded in {cfg.ais_store_dir} — is the collector "
                "running? (python -m backend.ingestion.ais_collector)",
            ))
        else:
            age_min = _age_minutes(cov["end"])
            fresh = age_min is not None and age_min < 15
            results.append(CheckResult(
                "AIS store", fresh,
                f"{cov['pings']} pings, {cov['start'][:16]} → {cov['end'][:16]} UTC "
                + (f"(last ping {age_min:.0f} min ago"
                   + ("" if fresh else " — collector down or regional receiver gap")
                   + ")" if age_min is not None else ""),
            ))
        results.append(_check_aisstream(cfg, stream_probe_s))
    elif cfg.ais_provider == "rest":
        results.append(_check_rest(cfg))
    else:
        results.append(CheckResult("AIS (file)", True,
                                   "manual file uploads — nothing to probe"))
    return results


def _check_aisstream(cfg: Settings, seconds: int) -> CheckResult:
    """Subscribe briefly: proves the key is valid and shows live throughput."""
    t0 = time.monotonic()
    if not cfg.ais_api_key:
        return CheckResult("aisstream.io feed", False, "AIS API key not set")

    async def probe() -> tuple[bool, str]:
        import asyncio
        import websockets

        sub = {
            "APIKey": cfg.ais_api_key,
            "BoundingBoxes": [[[cfg.aoi_bbox[1], cfg.aoi_bbox[0]],
                               [cfg.aoi_bbox[3], cfg.aoi_bbox[2]]]],
            "FilterMessageTypes": ["PositionReport"],
        }
        n = 0
        async with websockets.connect(AISSTREAM_URL, open_timeout=10) as ws:
            await ws.send(json.dumps(sub))
            loop = asyncio.get_event_loop()
            deadline = loop.time() + seconds
            while loop.time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(),
                                                 timeout=deadline - loop.time())
                except asyncio.TimeoutError:
                    break
                msg = json.loads(raw)
                if msg.get("error"):
                    return False, f"stream rejected key: {msg['error']}"
                n += 1
        if n:
            return True, f"key valid — {n} live pings in {seconds}s for this AOI"
        return True, (f"key valid, connected — 0 pings in {seconds}s (no vessel "
                      "movement or a receiver coverage gap in the AOI right now)")

    try:
        import asyncio

        ok, detail = asyncio.run(probe())
        return CheckResult("aisstream.io feed", ok, detail, time.monotonic() - t0)
    except Exception as exc:
        return CheckResult("aisstream.io feed", False,
                           f"{type(exc).__name__}: {str(exc)[:160]}",
                           time.monotonic() - t0)


def _check_rest(cfg: Settings) -> CheckResult:
    t0 = time.monotonic()
    try:
        from backend.ingestion.ais_providers import RestAISProvider

        provider = RestAISProvider(cfg.ais_api_url, cfg.ais_api_key)
        end = datetime.now(tz=timezone.utc)
        df = provider.fetch(cfg.aoi_bbox, end - timedelta(minutes=15), end)
        return CheckResult("AIS REST API", True,
                           f"endpoint OK — {len(df)} pings in the last 15 min",
                           time.monotonic() - t0)
    except Exception as exc:
        return CheckResult("AIS REST API", False,
                           f"{type(exc).__name__}: {str(exc)[:160]}",
                           time.monotonic() - t0)


def check_detector(cfg: Settings) -> CheckResult:
    t0 = time.monotonic()
    if cfg.detector_backend == "replicate":
        try:
            import requests

            if not cfg.replicate_api_token:
                return CheckResult("Detector (Replicate)", False,
                                   "API token not set — replicate.com/account/api-tokens")
            if not cfg.replicate_model or "/" not in cfg.replicate_model:
                return CheckResult("Detector (Replicate)", False,
                                   "model not set — use 'owner/name'")
            r = requests.get(
                f"https://api.replicate.com/v1/models/{cfg.replicate_model}",
                headers={"Authorization": f"Bearer {cfg.replicate_api_token}"},
                timeout=30,
            )
            if r.status_code == 401:
                return CheckResult("Detector (Replicate)", False,
                                   "token rejected (401) — regenerate at "
                                   "replicate.com/account/api-tokens",
                                   time.monotonic() - t0)
            if r.status_code == 404:
                return CheckResult("Detector (Replicate)", False,
                                   f"model '{cfg.replicate_model}' not found — "
                                   "check owner/name, and that the token "
                                   "belongs to the owning account",
                                   time.monotonic() - t0)
            r.raise_for_status()
            version = (r.json().get("latest_version") or {}).get("id")
            if not version:
                return CheckResult("Detector (Replicate)", False,
                                   f"model '{cfg.replicate_model}' exists but "
                                   "has no pushed version yet — wait for/rerun "
                                   "`cog push` (see replicate_xview3/cog.yaml)",
                                   time.monotonic() - t0)
            return CheckResult("Detector (Replicate)", True,
                               f"model '{cfg.replicate_model}' reachable, "
                               f"version {version[:8]}… (metadata check only — "
                               "no billable inference run)",
                               time.monotonic() - t0)
        except Exception as exc:
            return CheckResult("Detector (Replicate)", False,
                               f"{type(exc).__name__}: {str(exc)[:160]}",
                               time.monotonic() - t0)
    if cfg.detector_backend == "roboflow":
        try:
            import base64
            import io
            import numpy as np
            import requests
            from PIL import Image

            from backend.detection.hosted_api import (
                ROBOFLOW_INFER_HOST,
                normalize_roboflow_model_id,
                validate_roboflow_model_id,
            )

            model_id = normalize_roboflow_model_id(cfg.roboflow_model_id)
            problem = validate_roboflow_model_id(model_id)
            if problem:
                return CheckResult("Detector (Roboflow)", False, problem,
                                   time.monotonic() - t0)
            img = (np.random.default_rng(0).gamma(4, 0.25, (64, 64)) * 40
                   ).clip(0, 255).astype("uint8")
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="PNG")
            resp = requests.post(
                f"https://{ROBOFLOW_INFER_HOST}/{model_id}",
                params={"api_key": cfg.roboflow_api_key, "confidence": 50},
                data=base64.b64encode(buf.getvalue()),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30,
            )
            if resp.status_code == 404:
                return CheckResult("Detector (Roboflow)", False,
                                   f"model '{model_id}' not found (404) — this "
                                   "API key's workspace doesn't have a project/"
                                   "version by that exact id. Roboflow's 'Model "
                                   "URL' copy button can show a descriptive "
                                   "name that isn't the real id — check the "
                                   "project's Versions page or its Python/cURL "
                                   "snippet for the exact slug/version.",
                                   time.monotonic() - t0)
            if resp.status_code == 405:
                return CheckResult("Detector (Roboflow)", False,
                                   f"model '{model_id}' returned 405 — Roboflow "
                                   "didn't accept this id/version shape. Copy "
                                   "it straight from the project's Deploy tab "
                                   "rather than typing it by hand.",
                                   time.monotonic() - t0)
            if resp.status_code in (401, 403):
                return CheckResult("Detector (Roboflow)", False,
                                   f"model '{model_id}' rejected the API key "
                                   f"({resp.status_code}) — check it belongs to "
                                   "the workspace that owns this model",
                                   time.monotonic() - t0)
            resp.raise_for_status()
            return CheckResult("Detector (Roboflow)", True,
                               f"model '{model_id}' responding "
                               "(1 test call used)",
                               time.monotonic() - t0)
        except Exception as exc:
            return CheckResult("Detector (Roboflow)", False,
                               f"{type(exc).__name__}: {str(exc)[:160]}",
                               time.monotonic() - t0)
    if cfg.detector_backend == "yolo":
        weights_ok = Path(cfg.yolo_weights).exists()
        try:
            import ultralytics  # noqa: F401
            lib_ok = True
        except ImportError:
            lib_ok = False
        ok = weights_ok and lib_ok
        return CheckResult(
            "Detector (YOLO)", ok,
            f"weights {'found' if weights_ok else 'MISSING'} at "
            f"{cfg.yolo_weights}; ultralytics "
            f"{'installed' if lib_ok else 'NOT installed'}",
        )
    return CheckResult(f"Detector ({cfg.detector_backend})", True,
                       "no local probe implemented for this backend")


def run_all(cfg: Settings) -> list[CheckResult]:
    results = [check_stac(cfg), check_imagery(cfg)]
    results.extend(check_ais(cfg))
    results.append(check_detector(cfg))
    return results


def _age_minutes(ts: str) -> float | None:
    try:
        t = datetime.fromisoformat(ts)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return (datetime.now(tz=timezone.utc) - t).total_seconds() / 60
    except ValueError:
        return None
