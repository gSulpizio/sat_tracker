"""Continuous AIS collector — feeds the local store from aisstream.io.

Live AIS cannot be queried retroactively on the free tier, so the pattern is:
record continuously, query the recording at fusion time. Run this as a
daemon (systemd unit, tmux, docker) alongside the platform:

    pip install websockets
    export AIS_API_KEY=<your aisstream.io key>      # free at aisstream.io
    python -m backend.ingestion.ais_collector

By default this watches every area saved under 📍 Saved locations in the
app (data/locations.json) — aisstream accepts multiple bounding boxes in
one subscription, so one lightweight process/websocket covers all of them
at once rather than needing one container per location. The list is
re-read periodically (see RELOAD_INTERVAL_S) so a location added or
removed in the app takes effect without restarting the collector. Pass
--bbox to watch a single explicit area instead (ignores locations.json).

Pings are appended to data/ais_store/YYYY-MM-DD.csv in the canonical schema
(mmsi, ts, lat, lon, sog_knots, cog_deg) that StoreAISProvider reads.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import signal
from datetime import datetime, timezone
from pathlib import Path

from backend.config import load_locations, settings

log = logging.getLogger("ais_collector")

AISSTREAM_URL = "wss://stream.aisstream.io/v0/stream"
FIELDNAMES = ["mmsi", "ts", "lat", "lon", "sog_knots", "cog_deg"]
RELOAD_INTERVAL_S = 900  # cap on how stale the watched-locations list can get

BBox = tuple[float, float, float, float]


def _normalize_ts(raw: str) -> str:
    """aisstream emits Go time.Time strings ('… .584591715 +0000 UTC');
    store clean microsecond ISO 8601 so any parser downstream copes."""
    import re

    s = raw.replace(" UTC", "").strip()
    s = re.sub(r"(\.\d{6})\d+", r"\1", s)      # nanoseconds → microseconds
    s = re.sub(r"\s+([+-]\d{4})$", r"\1", s)   # ' +0000' → '+0000'
    try:
        return datetime.fromisoformat(s).isoformat()
    except ValueError:
        return datetime.now(tz=timezone.utc).isoformat()


def _append_ping(store_dir: Path, row: dict) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)
    day_file = store_dir / f"{datetime.now(tz=timezone.utc):%Y-%m-%d}.csv"
    new_file = not day_file.exists()
    with day_file.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _to_aisstream_boxes(bboxes: list[BBox]) -> list:
    # aisstream expects a list of [[lat, lon], [lat, lon]] corner pairs —
    # one entry per watched area, all delivered over the single connection.
    return [[[b[1], b[0]], [b[3], b[2]]] for b in bboxes]


async def collect(
    api_key: str,
    store_dir: Path,
    static_bboxes: list[BBox] | None = None,
) -> None:
    """Subscribe to aisstream and append every position report to the
    local store. If `static_bboxes` is None, the watched area list comes
    from data/locations.json and is refreshed every RELOAD_INTERVAL_S by
    forcing a reconnect — otherwise it's fixed for the process lifetime.
    """
    import websockets

    def current_bboxes() -> list[BBox]:
        if static_bboxes is not None:
            return static_bboxes
        locs = list(load_locations().values())
        return locs or [settings.aoi_bbox]  # nothing saved yet: fall back

    backoff = 1
    idle_reconnect_s = 600  # silent-connection watchdog
    while True:
        bboxes = current_bboxes()
        subscribe = {
            "APIKey": api_key,
            "BoundingBoxes": _to_aisstream_boxes(bboxes),
            "FilterMessageTypes": ["PositionReport"],
        }
        try:
            async with websockets.connect(AISSTREAM_URL) as ws:
                await ws.send(json.dumps(subscribe))
                log.info("Subscribed to aisstream for %d area(s): %s",
                         len(bboxes), bboxes)
                backoff = 1
                session_deadline = (
                    asyncio.get_event_loop().time() + RELOAD_INTERVAL_S
                    if static_bboxes is None else None
                )
                while True:
                    timeout = idle_reconnect_s
                    if session_deadline is not None:
                        remaining = session_deadline - asyncio.get_event_loop().time()
                        if remaining <= 0:
                            log.info("Refreshing watched-locations list "
                                     "(periodic %ds reload)", RELOAD_INTERVAL_S)
                            break
                        timeout = min(timeout, remaining)
                    # A websocket can stay "connected" while silently
                    # delivering nothing (dead subscription upstream).
                    # Force a resubscribe after a long silence — real
                    # regional gaps just resubscribe harmlessly. Also
                    # doubles as the periodic reload trigger above.
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    except asyncio.TimeoutError:
                        if session_deadline is None or (
                            asyncio.get_event_loop().time() < session_deadline
                        ):
                            log.warning(
                                "No AIS messages for %ds — resubscribing "
                                "(dead subscription or receiver coverage gap)",
                                idle_reconnect_s,
                            )
                        break
                    msg = json.loads(raw)
                    if msg.get("MessageType") != "PositionReport":
                        continue
                    meta = msg["MetaData"]
                    body = msg["Message"]["PositionReport"]
                    _append_ping(store_dir, {
                        "mmsi": meta["MMSI"],
                        "ts": _normalize_ts(
                            meta.get("time_utc")
                            or datetime.now(tz=timezone.utc).isoformat()
                        ),
                        "lat": meta["latitude"],
                        "lon": meta["longitude"],
                        "sog_knots": body.get("Sog"),
                        "cog_deg": body.get("Cog"),
                    })
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # network blips: reconnect with backoff
            log.warning("Stream dropped (%s) — reconnecting in %ds", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="aisstream.io → local store collector")
    parser.add_argument(
        "--bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        default=None,
        help="Watch one explicit area instead of every saved location "
             "(data/locations.json). Default: all saved locations, or "
             "AOI_BBOX/the built-in default if none are saved.",
    )
    parser.add_argument("--api-key", default=settings.ais_api_key,
                        help="aisstream.io key (default: AIS_API_KEY env)")
    parser.add_argument("--store", default=str(settings.ais_store_dir))
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Set AIS_API_KEY or pass --api-key (free key: aisstream.io)")

    static_bboxes = [tuple(args.bbox)] if args.bbox else None

    loop = asyncio.new_event_loop()
    task = loop.create_task(
        collect(args.api_key, Path(args.store), static_bboxes=static_bboxes)
    )
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, task.cancel)
    try:
        loop.run_until_complete(task)
    except asyncio.CancelledError:
        log.info("Collector stopped.")


if __name__ == "__main__":
    main()
