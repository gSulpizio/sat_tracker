"""End-to-end dark-vessel fusion pipeline (Step 2 deliverable).

    python -m backend.pipeline                             # live, AIS from API/store
    python -m backend.pipeline --ais fleet.csv             # live, AIS from a file
    python -m backend.pipeline --mode simulate             # synthetic test series

Processes EVERY Sentinel-1 pass found in the search window (up to
`max_scenes`) and writes one snapshot per pass to data/snapshots/ — the
dashboard navigates between them with prev/next. For each pass, AIS pings
are fetched from the configured provider (local aisstream store, historical
REST API, or an uploaded file) for ±temporal_window around THAT pass's
acquisition time, so image time and AIS time always coincide per snapshot.

LIVE mode (default) — real data only; mock detector rejected; simulated
runs are unmistakably flagged (`"simulated": true` + red banner in the UI).

Outputs:
    data/snapshots/index.json           run summary + ordered scene list
    data/snapshots/<scene>.json         one fused snapshot per pass
    data/snapshots/<scene>_sar.png      rendered SAR overlay per pass
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from pyproj import Geod

from backend.config import Settings, settings
from backend.detection.base import Detection, build_detector
from backend.detection.mock import MockDetector
from backend.fusion.matcher import (
    KNOTS_TO_MS,
    fuse,
    project_ais_to_image_time,
    summarize,
)
from backend.ingestion.ais_loader import generate_mock_ais
from backend.ingestion.ais_providers import FileAISProvider, build_ais_provider
from backend.ingestion.stac_client import (
    Scene,
    fetch_s2_rgb,
    group_into_passes,
    inject_vessel_signatures,
    load_scene,
    search_scenes,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pipeline")
GEOD = Geod(ellps="WGS84")


# Simulation ground truth, positioned as FRACTIONS of the AOI bbox at the
# FIRST simulated pass (fx: 0 = west → 1 = east; fy: 0 = south → 1 = north).
# Vessels then sail along their courses between passes, so each snapshot in
# the series shows the fleet in a different, physically consistent position.
# 6 AIS broadcasters: 5 SAR-visible + 1 small wooden hull SAR misses (→ green)
SIM_FLEET = [
    {"mmsi": 224000101, "fx": 0.533, "fy": 0.511, "sog_knots": 12.5, "cog_deg": 78.0,  "sar_visible": True},
    {"mmsi": 224000102, "fx": 0.722, "fy": 0.667, "sog_knots": 9.8,  "cog_deg": 255.0, "sar_visible": True},
    {"mmsi": 210000203, "fx": 0.244, "fy": 0.311, "sog_knots": 15.2, "cog_deg": 92.0,  "sar_visible": True},
    {"mmsi": 636000304, "fx": 0.878, "fy": 0.822, "sog_knots": 7.1,  "cog_deg": 310.0, "sar_visible": True},
    {"mmsi": 247000405, "fx": 0.800, "fy": 0.400, "sog_knots": 11.0, "cog_deg": 65.0,  "sar_visible": True},
    {"mmsi": 271000506, "fx": 0.167, "fy": 0.911, "sog_knots": 4.5,  "cog_deg": 180.0, "sar_visible": False},
]
# 3 dark targets (imaged, silent → red), also underway between passes
SIM_DARK = [
    {"fx": 0.389, "fy": 0.756, "sog_knots": 8.0,  "cog_deg": 120.0},
    {"fx": 0.644, "fy": 0.267, "sog_knots": 10.5, "cog_deg": 280.0},
    {"fx": 0.111, "fy": 0.467, "sog_knots": 6.0,  "cog_deg": 45.0},
]
# 1 static false positive (charted rock awash) — reappears in every pass,
# exactly like a real recurring clutter source an analyst learns to delete.
SIM_FP_FRAC = [(0.856, 0.178)]


def _frac_to_lonlat(bbox, fx: float, fy: float) -> tuple[float, float]:
    min_lon, min_lat, max_lon, max_lat = bbox
    return (
        round(min_lon + fx * (max_lon - min_lon), 6),
        round(min_lat + fy * (max_lat - min_lat), 6),
    )


def _advance(lon: float, lat: float, sog_knots: float, cog_deg: float,
             dt_s: float) -> tuple[float, float]:
    lon2, lat2, _ = GEOD.fwd(lon, lat, cog_deg, sog_knots * KNOTS_TO_MS * dt_s)
    return lon2, lat2


def _sim_fleet_at(bbox, epoch: datetime, t: datetime) -> list[dict]:
    dt_s = (t - epoch).total_seconds()
    fleet = []
    for v in SIM_FLEET:
        lon0, lat0 = _frac_to_lonlat(bbox, v["fx"], v["fy"])
        lon, lat = _advance(lon0, lat0, v["sog_knots"], v["cog_deg"], dt_s)
        fleet.append({**v, "lon": lon, "lat": lat})
    return fleet


def _sim_dark_at(bbox, epoch: datetime, t: datetime) -> list[tuple[float, float]]:
    dt_s = (t - epoch).total_seconds()
    out = []
    for d in SIM_DARK:
        lon0, lat0 = _frac_to_lonlat(bbox, d["fx"], d["fy"])
        out.append(_advance(lon0, lat0, d["sog_knots"], d["cog_deg"], dt_s))
    return out


def _safe_name(scene_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", scene_id)


def run(
    ais_file: str | None = None,
    cfg: Settings | None = None,
    search_start: datetime | None = None,
    search_end: datetime | None = None,
) -> dict:
    """Process all passes in the search window; returns the index payload."""
    cfg = cfg or settings
    simulated = cfg.mode == "simulate"
    cfg.snapshots_dir.mkdir(parents=True, exist_ok=True)

    if simulated:
        log.warning("=" * 62)
        log.warning("=== SIMULATED RUN — SYNTHETIC DATA, NOT REAL OBSERVATIONS ===")
        log.warning("=" * 62)
    else:
        if cfg.detector_backend == "mock":
            raise ValueError(
                "Live mode requires a real detector backend (yolo / roboflow "
                "/ vertex); 'mock' is only allowed in simulate mode."
            )

    # ------------------------------------------------------------ AIS source
    # Simulate mode fabricates AIS unless a file is supplied (then the
    # synthetic passes align to the file's coverage — useful for testing
    # fusion against a real export without imagery credentials).
    if simulated:
        provider = FileAISProvider(ais_file) if ais_file else None
    else:
        provider = build_ais_provider(cfg, ais_file)

    # ---------------------------------------------------- scene search window
    if search_start and search_end:
        start, end = search_start, search_end
    elif isinstance(provider, FileAISProvider):
        # A file fixes the fuseable period — search scenes inside its coverage.
        pad = timedelta(seconds=cfg.temporal_window_s)
        cov_start, cov_end = provider.coverage()
        start, end = cov_start - pad, cov_end + pad
        log.info("Scene search window derived from AIS file coverage: %s → %s",
                 start.isoformat(), end.isoformat())
    else:
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=cfg.search_days) if not simulated \
            else end - timedelta(hours=3)

    refs = search_scenes(cfg.aoi_bbox, start, end, cfg=cfg)
    if not refs:
        raise LookupError(
            f"No {cfg.stac_collection} scenes intersect bbox={cfg.aoi_bbox} "
            f"between {start.isoformat()} and {end.isoformat()} — widen the "
            "AOI, extend the search window, or check the collection name."
        )
    # A wide AOI or one straddling a frame boundary needs several adjacent
    # scene files mosaicked to fill it — group by overpass, not by file.
    passes = group_into_passes(refs)
    multi = [g for g in passes if len(g) > 1]
    if multi:
        log.info("Grouped %d scene files into %d overpasses (%d multi-frame: %s)",
                 len(refs), len(passes), len(multi),
                 ", ".join(f"{len(g)} frames @ {g[0].acquired_at:%m-%d %H:%M}"
                           for g in multi))
    if len(passes) > cfg.max_scenes:
        log.info("Limiting to the %d most recent of %d passes",
                 cfg.max_scenes, len(passes))
        passes = passes[-cfg.max_scenes:]

    detector = build_detector(cfg.detector_backend, cfg)
    epoch = passes[0][0].acquired_at  # simulation fleet evolution reference

    index_entries = []
    failed_passes = []  # [(scene_id, reason)] — one bad pass must not lose the rest
    current_files = {"index.json"}
    for group in passes:
        ref0 = group[0]
        t_img = ref0.acquired_at
        pad = timedelta(seconds=cfg.temporal_window_s)
        safe = _safe_name(ref0.scene_id)
        snap_file = cfg.snapshots_dir / f"{safe}.json"
        overlay_name = f"{safe}_sar.png"

        try:
            # AIS is fetched fresh on EVERY run (cheap), even for cached
            # passes — the store keeps filling, so a pass that had no
            # coverage yesterday may fuse today. Fetching before the
            # imagery also means an AIS misconfiguration fails cheaply,
            # without burning a COG download.
            if not simulated:
                ais = provider.fetch(cfg.aoi_bbox, t_img - pad, t_img + pad)

            # Detections are deterministic per (scene, detector, AOI) —
            # reuse them from the previous run instead of re-downloading
            # imagery and re-paying the detector API; only fusion re-runs
            # with fresh AIS.
            cached = None if simulated else _load_cached(snap_file, overlay_name, cfg)
            cached_s2 = None
            if cached is not None:
                scene_block, detections, cached_s2 = cached
                log.info("Pass %s: reusing %d cached detections — no imagery "
                         "download, no detector call", ref0.scene_id, len(detections))
            else:
                scene = load_scene(group, cfg.aoi_bbox, cfg=cfg)
                if simulated:
                    if provider is not None:
                        # Real AIS file: image ~70% of the fleet at their
                        # dead-reckoned positions; the rest become AIS_ONLY.
                        ais = provider.fetch(cfg.aoi_bbox, t_img - pad, t_img + pad)
                        proj = project_ais_to_image_time(ais, t_img, cfg.temporal_window_s)
                        pts = list(zip(proj["lon_proj"], proj["lat_proj"]))
                        visible = pts[: max(1, round(len(pts) * 0.7))] if pts else []
                    else:
                        fleet = _sim_fleet_at(cfg.aoi_bbox, epoch, t_img)
                        ais = generate_mock_ais(fleet, t_img)
                        visible = [(v["lon"], v["lat"]) for v in fleet if v["sar_visible"]]
                    dark = _sim_dark_at(cfg.aoi_bbox, epoch, t_img)
                    fps = [_frac_to_lonlat(cfg.aoi_bbox, fx, fy) for fx, fy in SIM_FP_FRAC]
                    if isinstance(detector, MockDetector):
                        detector.plant(visible + dark, false_positives=fps)
                    inject_vessel_signatures(scene, visible + dark + fps)

                detections = detector.detect(scene)
                log.info("Detector (%s): %d ship candidates",
                         cfg.detector_backend, len(detections))
                if not simulated and cfg.measure_lengths and detections:
                    from backend.measurement import measure_detections

                    n_sized = measure_detections(
                        [r.asset_href for r in group], detections, cfg=cfg)
                    log.info("Sized %d/%d detections from native-res chips",
                             n_sized, len(detections))
                _render_sar_overlay(scene, cfg.snapshots_dir / overlay_name)
                # Fraction of the AOI actually imaged by this pass. A
                # rotated frame footprint (or a seam between adjacent
                # frames) can leave most of a north-up AOI box empty even
                # after mosaicking every available frame — this makes that
                # visible instead of letting a mostly-transparent overlay
                # look like a rendering bug.
                coverage_pct = 100.0 if simulated else round(
                    float((~np.isclose(scene.image, 0.0)).mean()) * 100, 1)
                scene_block = {
                    "scene_id": scene.scene_id,
                    "platform": scene.platform,
                    "product_type": scene.product_type,
                    "acquired_at": scene.acquired_at.isoformat(),
                    "bbox": list(scene.bbox),
                    "crs": scene.crs,
                    "asset_href": scene.asset_href,
                    "aoi_coverage_pct": coverage_pct,
                    "frame_count": len(group),
                }

            payload = _assemble_payload(scene_block, overlay_name, detections,
                                        ais, cfg, simulated, ais_file)

            # Optical context: nearest cloud-free Sentinel-2 around this
            # pass. Best-effort — a failure never blocks the radar/AIS
            # result for this pass.
            if not simulated and cfg.include_s2:
                s2_name = f"{safe}_s2.png"
                if cached_s2 and (cfg.snapshots_dir / s2_name).exists():
                    payload["s2"] = cached_s2
                    current_files.add(s2_name)
                else:
                    try:
                        s2 = fetch_s2_rgb(cfg.aoi_bbox, t_img, cfg=cfg,
                                          max_cloud=cfg.s2_max_cloud)
                    except Exception as exc:
                        log.warning("S2 fetch failed for %s: %s", ref0.scene_id, exc)
                        s2 = None
                    if s2 is not None:
                        _save_display_png(s2.rgba, cfg.snapshots_dir / s2_name)
                        payload["s2"] = {
                            "scene_id": s2.scene_id,
                            "acquired_at": s2.acquired_at.isoformat(),
                            "cloud_cover": s2.cloud_cover,
                            "bbox": list(s2.bbox),
                            "overlay": s2_name,
                        }
                        current_files.add(s2_name)

            snap_file.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            # One pass with no real coverage (rotated swath entirely
            # missing the AOI), a detector hiccup, etc. must not cost the
            # analyst every OTHER pass in the run — log it, skip it, and
            # keep going. A run where every single pass fails still
            # surfaces below as a hard error.
            log.warning("Pass %s failed, skipping: %s", ref0.scene_id, exc)
            failed_passes.append({"scene_id": ref0.scene_id,
                                  "acquired_at": t_img.isoformat(),
                                  "reason": str(exc)})
            continue

        current_files |= {snap_file.name, overlay_name}
        index_entries.append({
            "scene_id": payload["scene"]["scene_id"],
            "acquired_at": t_img.isoformat(),
            "file": snap_file.name,
            "simulated": simulated,
            "stats": payload["stats"],
            "aoi_coverage_pct": payload["scene"].get("aoi_coverage_pct"),
        })
        log.info("Pass %s: %s", payload["scene"]["scene_id"], payload["stats"])

    if not index_entries:
        reasons = "; ".join(f"{f['scene_id']}: {f['reason']}" for f in failed_passes)
        raise RuntimeError(
            f"All {len(passes)} pass(es) in the search window failed — none "
            f"produced a usable snapshot. {reasons}"
        )

    # Prune snapshots that are not part of this run (old simulated series,
    # passes that dropped out of the search window) so the app's pass list
    # always mirrors the latest run.
    pruned = 0
    for f in cfg.snapshots_dir.iterdir():
        if f.name not in current_files:
            f.unlink()
            pruned += 1
    if pruned:
        log.info("Pruned %d stale snapshot files", pruned)

    index = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "simulated": simulated,
        "search_window": {"start": start.isoformat(), "end": end.isoformat()},
        "scenes": index_entries,  # chronological
        "failed_passes": failed_passes,
    }
    (cfg.snapshots_dir / "index.json").write_text(json.dumps(index, indent=2))
    if failed_passes:
        log.warning("%d/%d pass(es) failed and were skipped: %s",
                    len(failed_passes), len(passes),
                    ", ".join(f["scene_id"] for f in failed_passes))
    log.info("Wrote %d snapshot(s) to %s", len(index_entries), cfg.snapshots_dir)
    return index


def _detector_model(cfg: Settings) -> str:
    """The concrete model identity behind the backend — cache key component."""
    return {
        "yolo": cfg.yolo_weights,
        "roboflow": cfg.roboflow_model_id,
        "replicate": cfg.replicate_model,
    }.get(cfg.detector_backend, cfg.detector_backend)


def _load_cached(snap_file, overlay_name: str, cfg: Settings):
    """(scene_block, detections) from a previous run's snapshot, or None if
    absent or produced with different imagery/detector inputs."""
    if not snap_file.exists() or not (cfg.snapshots_dir / overlay_name).exists():
        return None
    try:
        old = json.loads(snap_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    oc = old.get("config", {})
    if (old.get("simulated")
            or oc.get("detector_backend") != cfg.detector_backend
            # snapshots from before this field existed pass on backend match
            or (oc.get("detector_model") or _detector_model(cfg)) != _detector_model(cfg)
            or oc.get("aoi_bbox") != list(cfg.aoi_bbox)):
        return None
    detections = [
        Detection(
            lat=d["lat"], lon=d["lon"],
            confidence=d.get("confidence") or 0.0,
            obb_lonlat=[tuple(p) for p in d.get("obb") or []],
            length_m=d.get("length_m"),
            width_m=d.get("width_m"),
            heading_deg=d.get("heading_deg"),
            source_model=d.get("source_model", "cached"),
        )
        for v in old.get("vessels", [])
        if (d := v.get("detection"))
    ]
    return old["scene"], detections, old.get("s2")


def _assemble_payload(scene_block: dict, overlay_name: str, detections: list,
                      ais: pd.DataFrame, cfg: Settings,
                      simulated: bool, ais_file: str | None) -> dict:
    """Fuse one pass's detections with AIS and assemble its snapshot payload."""
    t_img = datetime.fromisoformat(scene_block["acquired_at"])
    log.info("Scene %s acquired %s — %d AIS pings from %d vessels available",
             scene_block["scene_id"], t_img.isoformat(), len(ais),
             ais["mmsi"].nunique() if not ais.empty else 0)

    # ------------------------------------------------- time-coverage check
    if ais.empty:
        time_alignment = {
            "t_img": t_img.isoformat(),
            "ais_coverage_start": None,
            "ais_coverage_end": None,
            "temporal_window_s": cfg.temporal_window_s,
            "pings_in_window": 0,
            "vessels_in_window": 0,
            "vessels_total": 0,
        }
        log.warning("No AIS pings at all for ±%ds around T_img=%s — every "
                    "detection will be flagged DARK.",
                    cfg.temporal_window_s, t_img.isoformat())
    else:
        dt_abs = (pd.Timestamp(t_img) - ais["ts"]).dt.total_seconds().abs()
        in_window = dt_abs <= cfg.temporal_window_s
        time_alignment = {
            "t_img": t_img.isoformat(),
            "ais_coverage_start": ais["ts"].min().isoformat(),
            "ais_coverage_end": ais["ts"].max().isoformat(),
            "temporal_window_s": cfg.temporal_window_s,
            "pings_in_window": int(in_window.sum()),
            "vessels_in_window": int(ais.loc[in_window, "mmsi"].nunique()),
            "vessels_total": int(ais["mmsi"].nunique()),
        }

    fused = fuse(detections, ais, t_img,
                 window_s=cfg.temporal_window_s, gate_m=cfg.match_gate_m)
    stats = summarize(fused)

    return {
        "simulated": simulated,
        "scene": scene_block,
        "sar_overlay": overlay_name,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "config": {
            "mode": cfg.mode,
            "detector_backend": cfg.detector_backend,
            "detector_model": _detector_model(cfg),
            "temporal_window_s": cfg.temporal_window_s,
            "match_gate_m": cfg.match_gate_m,
            "aoi_bbox": list(cfg.aoi_bbox),
            "ais_source": ais_file or ("mock (simulated)" if simulated else cfg.ais_provider),
            "stac_url": cfg.stac_url,
            "stac_collection": cfg.stac_collection,
        },
        "time_alignment": time_alignment,
        "stats": stats,
        "vessels": [{"id": f"v{i:03d}", **f.to_dict()} for i, f in enumerate(fused)],
    }


DISPLAY_MAX_PX = 1280  # map overlay resolution — independent of detection input


def _save_display_png(rgba: np.ndarray, out_path) -> None:
    """Save an RGBA array as the map-overlay PNG, downsized for display.

    st_folium/Leaflet embeds this file as base64 inside the page HTML and
    it gets re-sent on every Streamlit rerun (not just the initial load).
    Detection already ran on the full-resolution array before this is
    called, so shrinking here only affects what's shipped to the browser —
    it cuts payload size substantially with no effect on ship detection.
    """
    from PIL import Image

    img = Image.fromarray(rgba, "RGBA")
    scale = max(img.width, img.height) / DISPLAY_MAX_PX
    if scale > 1:
        img = img.resize((max(1, int(img.width / scale)),
                          max(1, int(img.height / scale))),
                         Image.LANCZOS)
    img.save(out_path, optimize=True)


def _render_sar_overlay(scene: Scene, out_path) -> None:
    """Opaque, contrast-stretched grayscale render of the SAR backscatter.

    Contrast percentiles are computed over VALID pixels only, and the
    warp/swath fill (exactly 0 dB from the nodata path) becomes fully
    transparent — so the overlay reads like a crisp satellite image where
    there is data and leaves the basemap untouched where there is none.
    Blend strength is controlled by the app's opacity slider, not baked in.
    """
    img = scene.image
    valid = ~np.isclose(img, 0.0)
    if valid.any():
        lo, hi = np.percentile(img[valid], [2, 99.5])
    else:
        lo, hi = 0.0, 1.0
    norm = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    gray = (norm * 255).astype(np.uint8)
    alpha = np.where(valid, 255, 0).astype(np.uint8)
    rgba = np.dstack([gray, gray, gray, alpha])
    _save_display_png(rgba, out_path)


if __name__ == "__main__":
    import argparse
    import os
    from dataclasses import replace

    parser = argparse.ArgumentParser(description="Dark-vessel fusion pipeline")
    parser.add_argument("--ais", default=None,
                        help="AIS CSV/JSON export (overrides the API provider)")
    parser.add_argument("--mode", choices=["live", "simulate"], default=None,
                        help="override SAT_TRACKER_MODE")
    parser.add_argument("--detector", default=None,
                        help="override DETECTOR_BACKEND (yolo/roboflow/vertex/mock)")
    parser.add_argument("--provider", default=None,
                        help="override AIS_PROVIDER (store/rest/file)")
    parser.add_argument("--start", default=None, help="scene search start (ISO 8601)")
    parser.add_argument("--end", default=None, help="scene search end (ISO 8601)")
    args = parser.parse_args()

    from backend.config import load_user_settings

    overrides = {}
    if args.mode:
        overrides["mode"] = args.mode
    if args.detector:
        overrides["detector_backend"] = args.detector
    elif args.mode == "simulate" and "DETECTOR_BACKEND" not in os.environ:
        overrides["detector_backend"] = "mock"  # sensible test-run default
    if args.provider:
        overrides["ais_provider"] = args.provider
    # CLI runs resume from the same saved config the app writes (if any)
    base = load_user_settings()
    cfg = replace(base, **overrides) if overrides else base

    result = run(
        ais_file=args.ais,
        cfg=cfg,
        search_start=datetime.fromisoformat(args.start) if args.start else None,
        search_end=datetime.fromisoformat(args.end) if args.end else None,
    )
    print(json.dumps(result, indent=2))
