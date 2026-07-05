"""Dark Vessel Detection dashboard (Step 3 deliverable).

    streamlit run app/streamlit_app.py

The pipeline produces one snapshot per satellite pass; navigate between them
with ◀ / ▶ — each pass carries its own scene, AIS window, fusion result, and
analyst corrections. AIS comes from an API provider (local aisstream store or
a historical REST API) with file upload as fallback. All options configurable
in the sidebar; simulate mode renders under a large SIMULATED banner.

Targets: blue = AIS+SAR verified · green = AIS only · red = dark vessel.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import datetime, time as dtime, timezone
from pathlib import Path

import folium
import streamlit as st
from folium.plugins import MiniMap
from streamlit_folium import st_folium

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (  # noqa: E402
    DATA_DIR,
    USER_CONFIG_PATH,
    clear_user_settings,
    delete_location,
    has_user_settings,
    load_locations,
    load_user_settings,
    save_location,
    save_user_settings,
    settings,
)
from backend.detection.hosted_api import normalize_roboflow_model_id  # noqa: E402

STATUS_STYLE = {
    "VERIFIED": {"color": "#1f77d0", "label": "Verified (AIS + SAR)"},
    "AIS_ONLY": {"color": "#2ca02c", "label": "AIS only (not imaged)"},
    "DARK":     {"color": "#d62728", "label": "DARK vessel (no AIS)"},
}

st.set_page_config(page_title="Dark Vessel Detection", layout="wide", page_icon="🛰️")


# ---------------------------------------------------------------- data layer

def scan_snapshots() -> list[dict]:
    """Load every per-pass snapshot for the ACTIVE location, chronological
    by acquisition time. Each AOI has its own snapshot folder (see
    backend.config.aoi_slug), so this only ever lists the current area's
    passes — switching locations naturally switches what's shown here.
    """
    snaps = []
    snap_dir = load_user_settings().snapshots_dir
    if snap_dir.exists():
        for f in snap_dir.glob("*.json"):
            if f.name == "index.json":
                continue
            try:
                snaps.append(json.loads(f.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
    snaps.sort(key=lambda s: s["scene"]["acquired_at"])
    return snaps


def init_state() -> None:
    # Re-scan whenever the active AOI differs from what's loaded — this is
    # what makes switching locations (preset select, drawn rectangle, or a
    # pipeline run under a new AOI) immediately show that location's own
    # snapshots on the main map, instead of leaving the previous location's
    # data on screen until a manual reload.
    active_aoi = load_user_settings().aoi_bbox
    if "snaps" not in st.session_state or st.session_state.get("snaps_aoi") != active_aoi:
        st.session_state.snaps = scan_snapshots()
        st.session_state.snap_idx = max(0, len(st.session_state.snaps) - 1)
        st.session_state.edits = {}  # scene_id -> {vessels, corrections, last_click}
        st.session_state.snaps_aoi = active_aoi


def reload_snapshots() -> None:
    for k in ("snaps", "snap_idx", "edits", "snaps_aoi"):
        st.session_state.pop(k, None)
    init_state()


def edits_for(snap: dict) -> dict:
    """Per-scene working copy: analyst edits survive prev/next navigation."""
    scene_id = snap["scene"]["scene_id"]
    if scene_id not in st.session_state.edits:
        st.session_state.edits[scene_id] = {
            "vessels": {v["id"]: dict(v) for v in snap["vessels"]},
            "corrections": [],
            "last_click": None,
            "last_marker_click": None,
        }
    return st.session_state.edits[scene_id]


def delete_vessel(vessels: dict, edits: dict, vessel_id: str) -> None:
    """Remove a detection the model got wrong. A VERIFIED target keeps its
    AIS track as a green 'AIS only' marker — deleting the radar detection
    shouldn't also erase a real ship's broadcast history."""
    removed = vessels.pop(vessel_id)
    if removed["status"] == "VERIFIED" and removed.get("mmsi"):
        vessels[vessel_id + "_ais"] = {
            "id": vessel_id + "_ais",
            "status": "AIS_ONLY",
            "lat": removed["lat"],
            "lon": removed["lon"],
            "mmsi": removed["mmsi"],
            "match_dist_m": None,
        }
    log_correction(edits, "DELETE", vessel_id=vessel_id)


def relink_vessel(vessels: dict, edits: dict, dark_id: str, ais_id: str) -> None:
    """Merge a DARK detection with an AIS-only track the matcher missed —
    the dark vessel becomes VERIFIED, carrying the track's MMSI."""
    dark_v = vessels[dark_id]
    ais_v = vessels.pop(ais_id)
    dark_v["status"] = "VERIFIED"
    dark_v["mmsi"] = ais_v["mmsi"]
    dark_v["manual_link"] = True
    log_correction(edits, "RELINK", vessel_id=dark_id, mmsi=ais_v["mmsi"])


def merge_vessels(vessels: dict, edits: dict, id_a: str, id_b: str) -> str:
    """Merge two detections of the SAME physical ship (double counting from
    chip-overlap seams, split radar blobs, …). Keeps the better one —
    VERIFIED beats DARK, then an AIS identity, then detector confidence —
    and folds the other in, preferring non-null attributes from the keeper.
    Returns the surviving vessel id."""
    a, b = vessels[id_a], vessels[id_b]

    def rank(v):
        det = v.get("detection") or {}
        return (v["status"] == "VERIFIED", v.get("mmsi") is not None,
                det.get("confidence") or 0)

    keep, drop = (a, b) if rank(a) >= rank(b) else (b, a)
    # Inherit anything the keeper lacks
    if keep.get("mmsi") is None and drop.get("mmsi") is not None:
        keep["mmsi"] = drop["mmsi"]
        if drop["status"] == "VERIFIED":
            keep["status"] = "VERIFIED"
    kd, dd = keep.get("detection") or {}, drop.get("detection") or {}
    for field in ("length_m", "width_m", "heading_deg", "obb"):
        if kd.get(field) is None and dd.get(field) is not None:
            kd[field] = dd[field]
    vessels.pop(drop["id"])
    log_correction(edits, "MERGE", kept=keep["id"], removed=drop["id"])
    return keep["id"]


def resize_vessel(vessels: dict, edits: dict, vessel_id: str,
                  length_m: float, width_m: float) -> None:
    """Analyst override of the estimated ship size — recorded in the audit
    trail like every other correction."""
    det = vessels[vessel_id].setdefault("detection", {})
    log_correction(edits, "RESIZE", vessel_id=vessel_id,
                   old_length_m=det.get("length_m"), new_length_m=length_m,
                   old_width_m=det.get("width_m"), new_width_m=width_m)
    det["length_m"] = round(length_m, 1)
    det["width_m"] = round(width_m, 1)
    det["size_manual"] = True


def is_simulated(snap: dict) -> bool:
    return bool(snap.get("simulated")) or "SIM" in snap["scene"]["scene_id"]


def vessel_label(v: dict) -> str:
    mmsi = f"MMSI {v['mmsi']}" if v.get("mmsi") else "no AIS"
    return f"{v['id']} · {v['status']} · {mmsi}"


def log_correction(edits: dict, action: str, **kw) -> None:
    edits["corrections"].append(
        {"action": action, "ts": datetime.now(tz=timezone.utc).isoformat(), **kw}
    )


# ------------------------------------------------------- pipeline config UI

def config_panel() -> None:
    snaps = st.session_state.snaps
    cur = snaps[st.session_state.snap_idx]["config"] if snaps else {}
    # Saved user config (data/user_config.json) wins over the last snapshot's
    # config, which wins over env defaults — so a restart resumes exactly
    # where the analyst left off.
    saved = load_user_settings()
    have_saved = has_user_settings()

    def default(fld: str, cur_key: str | None = None):
        if have_saved:
            return getattr(saved, fld)
        return cur.get(cur_key or fld, getattr(saved, fld))

    with st.expander("⚙️ Pipeline configuration", expanded=not snaps):
        if have_saved:
            st.caption(f"Loaded saved config from `{USER_CONFIG_PATH.name}` "
                       "(git-ignored). Settings save automatically on each run.")
        with st.form("pipeline_cfg"):
            mode = st.selectbox(
                "Data mode", ["live", "simulate"],
                index=["live", "simulate"].index(default("mode")),
                help="**live** — fetches real Sentinel-1 passes from the STAC "
                     "catalog and fuses each with AIS from the configured "
                     "provider. Requires credentials and a real detector.\n\n"
                     "**simulate** — generates a synthetic 3-pass series with "
                     "known ground truth, for testing only. All outputs are "
                     "flagged and shown under a large red SIMULATED banner.",
            )
            backends = ["replicate", "yolo", "roboflow", "vertex", "mock"]
            backend = st.selectbox(
                "Detector backend", backends,
                index=backends.index(default("detector_backend"))
                if default("detector_backend") in backends else 0,
                help="Which model finds ships in the SAR image:\n\n"
                     "- **replicate** — the xView3 challenge 2nd-place model "
                     "(purpose-built for Sentinel-1 dark-vessel detection; "
                     "also estimates vessel length directly), self-hosted on "
                     "Replicate via this repo's replicate_xview3/ package. "
                     "Pay-per-second with scale-to-zero: cents/month at this "
                     "workload. Reads native-resolution VV+VH imagery.\n"
                     "- **yolo** — open-source YOLOv8-OBB running locally. Free, "
                     "private, needs `pip install ultralytics`, trained weights, "
                     "and ideally a GPU.\n"
                     "- **roboflow** — hosted inference API. No local GPU needed; "
                     "point it at a model YOU trained on Roboflow (paste its "
                     "URL or id in the field below), or use the built-in "
                     "public default.\n"
                     "- **vertex** — Google Cloud Vertex AI endpoint; requires a "
                     "deployed model and GCP credentials.\n"
                     "- **mock** — returns planted test targets. Only accepted "
                     "in simulate mode.",
            )

            st.markdown("**AIS source**")
            providers = ["store", "rest", "file"]
            ais_provider = st.selectbox(
                "AIS provider", providers,
                index=providers.index(
                    saved.ais_provider if have_saved
                    else (cur.get("ais_source")
                          if cur.get("ais_source") in providers
                          else saved.ais_provider)
                ),
                help="Where ship position reports come from — fetched "
                     "automatically per pass, ±window around each image time:\n\n"
                     "- **store** — local recording continuously written by the "
                     "aisstream.io collector daemon "
                     "(`python -m backend.ingestion.ais_collector`, free API "
                     "key). Live AIS can't be queried retroactively, so you "
                     "record it and the pipeline queries the recording.\n"
                     "- **rest** — one HTTP call per pass to a historical AIS "
                     "API (Datalastic, Spire, in-house) via the URL template "
                     "below.\n"
                     "- **file** — manual CSV/JSON upload (fallback). An "
                     "uploaded file always takes precedence over the API.",
            )
            ais_api_url = st.text_input(
                "AIS REST URL template", default("ais_api_url"),
                help="Only for provider 'rest'. Placeholders are filled per "
                     "request: {min_lon} {min_lat} {max_lon} {max_lat} "
                     "{start_iso} {end_iso} {api_key}. Example:\n\n"
                     "`https://api.example.com/ais?bbox={min_lon},{min_lat},"
                     "{max_lon},{max_lat}&from={start_iso}&to={end_iso}`\n\n"
                     "The JSON response is normalized through the same column "
                     "aliases as file imports (MMSI/LAT/LON/SOG/COG variants).",
            )
            ais_api_key = st.text_input(
                "AIS API key", saved.ais_api_key, type="password",
                help="Bearer token for the REST provider (also used as the "
                     "aisstream.io key by the collector daemon). Saved with "
                     "the rest of the config to data/user_config.json — "
                     "git-ignored, file permissions 0600, local machine only.",
            )
            ais_up = st.file_uploader(
                "AIS file (CSV / JSON / JSONL) — overrides the provider",
                type=["csv", "json", "jsonl"],
                help="Ship position reports with columns for MMSI, timestamp, "
                     "lat, lon, speed (SOG, knots) and course (COG, degrees). "
                     "Common header variants (MarineCadastre, Spire, …) are "
                     "recognized automatically. When a file is uploaded, the "
                     "scene search window is derived from its time coverage, "
                     "so image and AIS times always coincide.",
            )

            st.markdown("**Fusion parameters**")
            window_min = st.number_input(
                "AIS temporal window ± (minutes)", min_value=0.5, max_value=120.0,
                value=default("temporal_window_s") / 60,
                step=0.5,
                help="Only AIS pings within this many minutes of each image "
                     "acquisition time are used for matching (each vessel's "
                     "position is then dead-reckoned to the exact image time "
                     "using its speed and course). Larger values tolerate sparse "
                     "AIS feeds but the projection gets less accurate the "
                     "further a ping is from the image time — at 15 knots a "
                     "vessel moves ~2.3 km in 5 minutes, so course changes "
                     "start to matter. ±5 min is a good default for Sentinel-1.",
            )
            gate_m = st.number_input(
                "Match gate — max detection↔AIS distance (m)",
                min_value=50, max_value=20000,
                value=int(default("match_gate_m")), step=50,
                help="After dead reckoning, a radar detection and an AIS track "
                     "are only paired if they lie within this distance of each "
                     "other. Detections with no AIS track inside the gate are "
                     "flagged DARK. Too small → real matches are missed and "
                     "honest ships show up as dark (false alarms); too large → "
                     "a dark vessel can 'steal' a nearby ship's AIS identity. "
                     "1000 m suits Sentinel-1 GRD accuracy plus typical AIS "
                     "GPS/projection error.",
            )

            st.markdown("**Area of interest** (EPSG:4326) — or draw it on "
                        "the map: 🗺 picker in the main panel")
            bbox0 = list(default("aoi_bbox"))
            ca, cb = st.columns(2)
            min_lon = ca.number_input(
                "Min lon", -180.0, 180.0, float(bbox0[0]), format="%.4f",
                help="Western edge of the analysis box, decimal degrees "
                     "(negative = west of Greenwich).",
            )
            min_lat = ca.number_input(
                "Min lat", -90.0, 90.0, float(bbox0[1]), format="%.4f",
                help="Southern edge of the analysis box, decimal degrees "
                     "(negative = southern hemisphere).",
            )
            max_lon = cb.number_input(
                "Max lon", -180.0, 180.0, float(bbox0[2]), format="%.4f",
                help="Eastern edge of the analysis box. Must be greater than "
                     "Min lon. Keep the box modest (≲ 1–2°) — imagery is read "
                     "at full resolution within it.",
            )
            max_lat = cb.number_input(
                "Max lat", -90.0, 90.0, float(bbox0[3]), format="%.4f",
                help="Northern edge of the analysis box. Must be greater than "
                     "Min lat. Only scenes intersecting this box are considered, "
                     "and only AIS traffic inside it is fused.",
            )

            include_s2 = st.toggle(
                "Fetch Sentinel-2 optical context", value=saved.include_s2,
                help="For each radar pass, also download the nearest "
                     "cloud-free Sentinel-2 true-color image (±5 days) as a "
                     "map overlay. Adds one search plus one image window "
                     "download per pass; disable for faster runs.",
            )
            s2_max_cloud = st.number_input(
                "Max S2 cloud cover (%)", min_value=0, max_value=100,
                value=int(saved.s2_max_cloud),
                help="Sentinel-2 scenes cloudier than this are skipped when "
                     "picking the optical context image. Lower = clearer "
                     "image but more passes end up with no optical layer.",
            )
            measure_lengths = st.toggle(
                "Measure ship sizes", value=saved.measure_lengths,
                help="After detection, re-read a small NATIVE-resolution "
                     "(~10 m/px) chip around each ship and estimate its "
                     "length, beam, and axis orientation from the radar "
                     "signature — shown in the marker popup as e.g. "
                     "'Size: ~180 × 30 m'. Costs a few extra tiny S3 reads "
                     "per pass, no API credits. Accuracy ≈ ±10–25 m and "
                     "faint hull ends get clipped, so treat as an estimate; "
                     "large ships' sidelobes are suppressed but anchored "
                     "clusters can still merge into one oversized blob.",
            )

            st.markdown("**Scene search**")
            search_days = st.number_input(
                "Search lookback (days)", min_value=1, max_value=30,
                value=saved.search_days,
                help="How far back from now to search for Sentinel-1 passes "
                     "when using an API provider (ignored when an AIS file "
                     "fixes the window, or when the manual override below is "
                     "set). Sentinel-1 revisits a given area every 1–3 days, "
                     "and publication can lag — 7 days is a safe default; if "
                     "a run finds no scenes, widen this first.",
            )
            max_scenes = st.number_input(
                "Max passes per run", min_value=1, max_value=20,
                value=saved.max_scenes,
                help="Upper limit on how many passes are processed (newest "
                     "kept). Each pass costs one imagery download, one "
                     "detector run, and one AIS fetch — raise it to build a "
                     "longer navigable history, lower it for quick runs.",
            )
            manual_window = st.checkbox(
                "Override scene search window",
                value=False,
                help="Force a specific date range instead of the automatic "
                     "one (AIS file coverage, or now − lookback). Useful to "
                     "revisit a known incident period. If the range doesn't "
                     "overlap your AIS data, every detection will flag DARK.",
            )
            cw1, cw2 = st.columns(2)
            search_start_d = cw1.date_input(
                "Search start (UTC)", value=None,
                help="Earliest acquisition date to consider (00:00 UTC). Used "
                     "only when the override above is ticked.",
            )
            search_end_d = cw2.date_input(
                "Search end (UTC)", value=None,
                help="Latest acquisition date to consider (23:59 UTC). Used "
                     "only when the override above is ticked.",
            )

            st.markdown("**Data sources & credentials**")
            stac_url = st.text_input(
                "STAC catalog URL", default("stac_url"),
                help="SpatioTemporal Asset Catalog endpoint to search for "
                     "scenes. Default is the Copernicus Data Space Ecosystem "
                     "(free, registration required). Any STAC-compliant "
                     "provider works, e.g. Microsoft Planetary Computer.",
            )
            stac_coll = st.text_input(
                "STAC collection", default("stac_collection"),
                help="Collection name inside the catalog — default "
                     "sentinel-1-grd on the Copernicus Data Space, whose COG "
                     "assets are read directly (ground-control-point products "
                     "are warped to EPSG:4326 on the fly; accurate over ocean). "
                     "For land-adjacent precision work a terrain-corrected "
                     "(RTC) collection is preferable.",
            )
            cdse_id = st.text_input(
                "CDSE client id", saved.cdse_client_id,
                help="OAuth2 client id for the Copernicus Data Space Ecosystem, "
                     "needed to download imagery from the default catalog. "
                     "Create one for free: dataspace.copernicus.eu → account → "
                     "OAuth clients. Leave empty for catalogs that serve "
                     "public/unsigned assets.",
            )
            cdse_secret = st.text_input(
                "CDSE client secret", saved.cdse_client_secret, type="password",
                help="Secret paired with the client id above. Shown once when "
                     "you create the OAuth client. Saved with the rest of the "
                     "config to data/user_config.json — git-ignored, file "
                     "permissions 0600, local machine only.",
            )
            cdse_s3_key = st.text_input(
                "CDSE S3 key", saved.cdse_s3_key,
                help="Access key for streaming imagery windows directly from "
                     "CDSE object storage (s3://eodata) — this is a DIFFERENT "
                     "credential from the OAuth client above. Generate a key "
                     "pair for free at eodata-s3keysmanager.dataspace."
                     "copernicus.eu. Required for the default sentinel-1-grd "
                     "collection, whose COG assets live on S3.",
            )
            cdse_s3_secret = st.text_input(
                "CDSE S3 secret", saved.cdse_s3_secret, type="password",
                help="Secret paired with the S3 key above. Saved with the "
                     "rest of the config to data/user_config.json — "
                     "git-ignored, file permissions 0600, local machine only.",
            )
            yolo_w = st.text_input(
                "YOLO weights path", saved.yolo_weights,
                help="Path to the fine-tuned YOLOv8-OBB weights file (.pt) used "
                     "by the local detector. Train on a SAR ship dataset "
                     "(xView3-SAR, HRSID, SSDD — recipe in the README). Only "
                     "used when the detector backend is 'yolo'.",
            )
            rf_key = st.text_input(
                "Roboflow API key", saved.roboflow_api_key, type="password",
                help="Private API key from your Roboflow workspace settings. "
                     "Only used when the detector backend is 'roboflow'. Note: "
                     "imagery is uploaded to Roboflow's servers for inference. "
                     "Saved with the rest of the config to data/user_config.json "
                     "— git-ignored, file permissions 0600, local machine only.",
            )
            rf_model = st.text_input(
                "Roboflow model — yours or public", saved.roboflow_model_id,
                help="Point this at a model YOU trained on Roboflow, or "
                     "leave the default public one. Paste whatever Roboflow "
                     "gives you — any of these work, auto-detected:\n\n"
                     "- A Universe page URL: "
                     "`universe.roboflow.com/<workspace>/<project>/model/<version>`\n"
                     "- The Deploy-tab curl/SDK URL: "
                     "`serverless.roboflow.com/<project>/<version>?api_key=...`\n"
                     "- The bare form: `project/version`\n\n"
                     "**The trailing /<version> number is required** — a "
                     "bare project slug with no version fails. To use your "
                     "own: open your project on Roboflow → Deploy tab → "
                     "copy either the page URL or the API snippet's URL and "
                     "paste it here as-is; don't type it by hand. It's "
                     "normalized and validated on ▶ Run pipeline (or check "
                     "🔍 Run diagnostics first without running the full "
                     "pipeline).\n\n"
                     "⚠ Roboflow's 'Model URL' copy button can show a "
                     "descriptive name that ISN'T the real id — e.g. "
                     "`find-ship-and-water-1-rfdetr-medium-t1` when the "
                     "actual working id is `find-ship-and-water/1`. If "
                     "diagnostics 404s on every version, open the project's "
                     "**Versions** page for the exact slug/version pair, or "
                     "check the Python/cURL code snippet on Deploy instead "
                     "of the page URL.\n\n"
                     "The default 'sar-ship-dataset-ogvbz/1' is a public "
                     "model trained on the 39k-image SAR-Ship-Dataset. Only "
                     "used when the detector backend is 'roboflow'.",
            )

            repl_token = st.text_input(
                "Replicate API token", saved.replicate_api_token, type="password",
                help="Token from replicate.com/account/api-tokens. Only used "
                     "when the detector backend is 'replicate'. Saved with "
                     "the rest of the config to data/user_config.json — "
                     "git-ignored, file permissions 0600, local machine only.",
            )
            repl_model = st.text_input(
                "Replicate model (owner/name)", saved.replicate_model,
                help="Which Replicate deployment to call, in 'owner/name' "
                     "form. The default 'gsulpizio/xview3-vessel-detect' is "
                     "the published public build of this repo's "
                     "replicate_xview3/ package — works with any Replicate "
                     "token, no model setup needed (you pay your own usage). "
                     "To self-host instead, push the package to your account "
                     "with cog (see README) and put your owner/name here. "
                     "Only used when the detector backend is 'replicate'.",
            )

            submitted = st.form_submit_button("▶ Run pipeline", use_container_width=True)

        if have_saved and st.button(
            "🗑 Clear saved config", use_container_width=True,
            help="Delete data/user_config.json (including stored credentials) "
                 "and fall back to environment-variable defaults on the next "
                 "restart.",
        ):
            clear_user_settings()
            st.rerun()

        if not submitted:
            return
        if min_lon >= max_lon or min_lat >= max_lat:
            st.error("Invalid AOI: min lon/lat must be smaller than max lon/lat.")
            return

        ais_path = None
        if ais_up is not None:
            ais_path = DATA_DIR / f"uploaded_ais{Path(ais_up.name).suffix.lower()}"
            DATA_DIR.mkdir(exist_ok=True)
            ais_path.write_bytes(ais_up.getvalue())
        elif mode == "live" and ais_provider == "file":
            st.error("AIS provider 'file' selected but no file was uploaded.")
            return

        search_start = search_end = None
        if manual_window and search_start_d and search_end_d:
            search_start = datetime.combine(search_start_d, dtime.min, tzinfo=timezone.utc)
            search_end = datetime.combine(search_end_d, dtime.max, tzinfo=timezone.utc)

        cfg = replace(
            saved,
            mode=mode,
            detector_backend=backend,
            ais_provider=ais_provider,
            ais_api_url=ais_api_url,
            ais_api_key=ais_api_key,
            temporal_window_s=int(window_min * 60),
            match_gate_m=float(gate_m),
            include_s2=bool(include_s2),
            s2_max_cloud=float(s2_max_cloud),
            measure_lengths=bool(measure_lengths),
            aoi_bbox=(min_lon, min_lat, max_lon, max_lat),
            search_days=int(search_days),
            max_scenes=int(max_scenes),
            stac_url=stac_url or saved.stac_url,
            stac_collection=stac_coll or saved.stac_collection,
            cdse_client_id=cdse_id,
            cdse_client_secret=cdse_secret,
            cdse_s3_key=cdse_s3_key,
            cdse_s3_secret=cdse_s3_secret,
            yolo_weights=yolo_w or saved.yolo_weights,
            roboflow_api_key=rf_key,
            roboflow_model_id=normalize_roboflow_model_id(rf_model) or saved.roboflow_model_id,
            replicate_api_token=repl_token,
            replicate_model=repl_model.strip().strip("/"),
        )
        # Persist before running: even a failed attempt (bad credentials,
        # empty search window) keeps what was typed for the next session.
        save_user_settings(cfg)
        # A manually-edited AOI may no longer match whichever preset the
        # picker last considered "applied" — clear the stale label so the
        # 📍 Saved locations dropdown doesn't misleadingly keep it selected.
        al = st.session_state.get("applied_location")
        if al is not None:
            al_bbox = load_locations().get(al)
            if al_bbox is None or tuple(al_bbox) != tuple(cfg.aoi_bbox):
                st.session_state.applied_location = None
        try:
            with st.spinner("Running fusion pipeline (all passes)…"):
                from backend.pipeline import run
                result = run(
                    ais_file=str(ais_path) if ais_path else None,
                    cfg=cfg,
                    search_start=search_start,
                    search_end=search_end,
                )
        except Exception as exc:  # surface config/credential/data errors in the UI
            st.error(f"Pipeline failed: {exc}")
        else:
            # A run can partially fail (e.g. one pass's swath genuinely
            # doesn't overlap the AOI) without losing the other passes —
            # stash the skipped ones so they're shown once after the
            # rerun, instead of vanishing silently.
            if result.get("failed_passes"):
                st.session_state.last_run_failed_passes = result["failed_passes"]
            else:
                st.session_state.pop("last_run_failed_passes", None)
            reload_snapshots()
            st.rerun()


# ---------------------------------------------------------- source status UI

def status_panel() -> None:
    from backend.diagnostics import run_all, store_coverage

    saved = load_user_settings()
    with st.expander("📡 Data source status", expanded=False):
        # Passive, always-on view of the local AIS store (cheap file read)
        if saved.ais_provider == "store":
            cov = store_coverage(saved)
            if cov is None:
                st.error(
                    "**AIS store: empty.** No pings recorded yet — start the "
                    "collector:\n`python -m backend.ingestion.ais_collector`"
                )
            else:
                from backend.diagnostics import _age_minutes
                age = _age_minutes(cov["end"])
                line = (f"**AIS store:** {cov['pings']} pings · "
                        f"{cov['start'][:16]} → {cov['end'][:16]} UTC")
                if age is not None and age < 15:
                    st.success(line + f" · last ping {age:.0f} min ago 🟢 live")
                elif age is not None:
                    st.warning(
                        line + f" · **last ping {age/60:.1f} h ago** — the "
                        "collector is down, or aisstream's volunteer receivers "
                        "have a coverage gap in your AOI. Run diagnostics below "
                        "to tell which."
                    )
                else:
                    st.info(line)

        if st.button(
            "🔍 Run diagnostics", use_container_width=True,
            help="Actively probes every configured source: STAC catalog "
                 "reachability, imagery access (opens a real scene header "
                 "with your S3 keys), the AIS provider (store freshness plus "
                 "a 15 s live subscription to aisstream, or a REST test "
                 "call), and the detector (one tiny Roboflow test inference, "
                 "or local YOLO weights/library check). Takes ~20–30 s.",
        ):
            with st.spinner("Probing data sources…"):
                st.session_state.diagnostics = run_all(saved)

        for r in st.session_state.get("diagnostics", []):
            icon = "✅" if r.ok else "❌"
            lat = f" `{r.latency_s:.1f}s`" if r.latency_s else ""
            st.markdown(f"{icon} **{r.name}**{lat} — {r.detail}")


# ------------------------------------------------------------- AOI picker UI

MAX_READ_PX = 2048  # mirrors stac_client.MAX_READ_PX — imagery read cap


def validate_aoi(bbox: tuple[float, float, float, float]) -> tuple[str, str]:
    """('ok'|'warn'|'error', message) for a candidate AOI, based on the
    pipeline's real constraints rather than arbitrary numbers."""
    import math

    min_lon, min_lat, max_lon, max_lat = bbox
    if not (-180 <= min_lon < max_lon <= 180 and -85 <= min_lat < max_lat <= 85):
        return "error", ("Invalid rectangle (check bounds; boxes crossing the "
                         "±180° antimeridian are not supported).")

    mid_lat = (min_lat + max_lat) / 2
    w_km = (max_lon - min_lon) * 111.32 * math.cos(math.radians(mid_lat))
    h_km = (max_lat - min_lat) * 110.57
    # Effective ground resolution after the 2048 px read cap (S1 native ~10 m)
    m_per_px = max(10.0, max(w_km, h_km) * 1000 / MAX_READ_PX)
    size_line = f"{w_km:.0f} × {h_km:.0f} km · ~{m_per_px:.0f} m/px read resolution"

    if max(w_km, h_km) > 400:
        return "error", (
            f"Too large ({size_line}). Imagery reads are capped at "
            f"{MAX_READ_PX} px, so at this size even large ships shrink "
            "below a pixel and detection is meaningless. Split the region "
            "into multiple AOIs of ≲80 × 80 km instead."
        )
    if m_per_px > 40:
        return "warn", (
            f"{size_line}. Above ~40 m/px small and mid-size vessels start "
            "to vanish for the detector — expect misses. ≲80 × 80 km keeps "
            "full detection quality."
        )
    if max(w_km, h_km) < 2:
        return "warn", (f"{size_line}. Very small — fine for a single "
                        "anchorage, but AIS matching benefits from context; "
                        "consider ≥5 km.")
    return "ok", f"{size_line} — good for detection."


def aoi_picker() -> None:
    from folium.plugins import Draw

    saved = load_user_settings()
    cur = saved.aoi_bbox
    locations = load_locations()

    # Which saved name is "active" is tracked EXPLICITLY by name in
    # session_state, not re-derived from bbox equality on every rerun.
    # Reverse-deriving it from coordinates breaks the moment two locations
    # share a bbox (e.g. a duplicate bookmark) — the lookup then always
    # resolves to whichever name comes first, so picking the OTHER one can
    # never register as "applied", and the app saves + reruns forever
    # trying to catch up. Name-based tracking has no such ambiguity.
    if "applied_location" not in st.session_state:
        st.session_state.applied_location = next(
            (n for n, b in locations.items()
             if all(abs(b[i] - cur[i]) < 1e-6 for i in range(4))), None,
        )

    with st.expander("🗺 Area of interest — draw on map", expanded=False):
        # -------------------------------------------------- location presets
        # One row: switch, delete, or bookmark-the-current-AOI — every
        # "manage the location list" action lives together here.
        lc1, lc2, lc3 = st.columns([3, 1, 1])
        if locations:
            names = list(locations.keys())
            default_name = st.session_state.applied_location
            chosen = lc1.selectbox(
                "📍 Saved locations", names,
                index=names.index(default_name) if default_name in names else 0,
                key="location_select",
                help="Switch the AOI to a bookmarked location. Selecting one "
                     "immediately updates the AOI everywhere — config panel, "
                     "this map, and the main dashboard map below (showing that "
                     "location's own cached passes, if any).",
            )
            if lc2.button(
                "🗑", use_container_width=True, help="Delete this bookmark "
                "(does not change the current AOI).",
            ):
                delete_location(chosen)
                if st.session_state.applied_location == chosen:
                    st.session_state.applied_location = None
                st.rerun()
            if chosen != st.session_state.applied_location:
                save_user_settings(replace(saved, aoi_bbox=locations[chosen]))
                st.session_state.applied_location = chosen
                st.rerun()
        else:
            lc1.caption("No saved locations yet — bookmark the current AOI, "
                       "or draw a box below.")

        with lc3.popover("💾", use_container_width=True, help="Bookmark the "
                         "AOI currently in effect under a name."):
            bookmark_name = st.text_input(
                "Name", key="bookmark_current_name",
                placeholder="e.g. Strait of Gibraltar",
                help="Name for the AOI currently in effect (shown above the "
                     "map below). You'll pick it from 📍 Saved locations.",
            )
            if st.button("Save", key="bookmark_current_btn"):
                if not bookmark_name.strip():
                    st.error("Enter a name.")
                else:
                    save_location(bookmark_name.strip(), cur)
                    st.session_state.applied_location = bookmark_name.strip()
                    st.rerun()

        st.divider()
        st.caption(
            "Use the ▭ rectangle tool (top-left of the map) to drag a new "
            "area of interest. The blue dashed box is the current AOI."
        )
        m = folium.Map(
            location=[(cur[1] + cur[3]) / 2, (cur[0] + cur[2]) / 2],
            zoom_start=8, tiles="OpenStreetMap", control_scale=True,
        )
        folium.Rectangle(
            bounds=[[cur[1], cur[0]], [cur[3], cur[2]]],
            color="#1f77d0", weight=2, dash_array="6", fill=False,
            tooltip="Current AOI",
        ).add_to(m)
        Draw(
            draw_options={
                "rectangle": {"shapeOptions": {"color": "#d62728", "weight": 2}},
                "polygon": False, "polyline": False, "circle": False,
                "marker": False, "circlemarker": False,
            },
            edit_options={"edit": False, "remove": True},
        ).add_to(m)
        state = st_folium(
            m, height=380, use_container_width=True,
            # Key includes the current AOI so saving a new box forces the
            # browser-side Leaflet map to fully remount (recentered on the
            # new box, old rectangle gone). A static key would leave
            # streamlit-folium's client-side map mounted with its previous
            # view/layers untouched — the map "doesn't update" until you
            # reload the whole page, since only a fresh page load creates a
            # brand-new component instance.
            key=f"aoi_picker_{cur[0]:.4f}_{cur[1]:.4f}_{cur[2]:.4f}_{cur[3]:.4f}",
            returned_objects=["last_active_drawing"],  # see main map for why
        )

        drawing = (state or {}).get("last_active_drawing")
        if not (drawing and drawing.get("geometry", {}).get("type") == "Polygon"):
            return
        coords = drawing["geometry"]["coordinates"][0]
        bbox = (
            round(min(c[0] for c in coords), 4),
            round(min(c[1] for c in coords), 4),
            round(max(c[0] for c in coords), 4),
            round(max(c[1] for c in coords), 4),
        )
        level, msg = validate_aoi(bbox)
        st.markdown(f"Drawn: `{bbox[0]}, {bbox[1]} → {bbox[2]}, {bbox[3]}`")
        {"ok": st.success, "warn": st.warning, "error": st.error}[level](msg)
        if level != "error":
            nc, bc = st.columns([3, 2])
            draw_name = nc.text_input(
                "Name (optional)", key="draw_location_name",
                placeholder="leave blank to just switch, or name it to bookmark",
                label_visibility="collapsed",
                help="Optional: give this drawn rectangle a name to also "
                     "save it under 📍 Saved locations. Leave blank to just "
                     "switch the active AOI without bookmarking it.",
            )
            if bc.button(
                "✔ Use this rectangle", use_container_width=True,
                help="Makes this the active AOI — the config panel fields "
                     "update to match, and the next ▶ Run pipeline uses it. "
                     "If you gave it a name above, it's also saved under "
                     "📍 Saved locations for later. Note: changing the AOI "
                     "invalidates cached detections, so the next run "
                     "downloads imagery and calls the detector again. "
                     "Remember to point the AIS collector at the new box "
                     "too (--bbox).",
            ):
                if draw_name.strip():
                    save_location(draw_name.strip(), bbox)
                    st.session_state.applied_location = draw_name.strip()
                else:
                    st.session_state.applied_location = None  # ad-hoc AOI
                save_user_settings(replace(saved, aoi_bbox=bbox))
                st.rerun()


# ----------------------------------------------------------------- map build
#
# Split so that clicking a marker doesn't feel like the app is reloading:
# the BASE map (tiles + multi-MB SAR/S2 overlays) is built once per
# (scene, layer settings) and cached in session state; the vessel markers
# ride in a folium FeatureGroup passed via st_folium's feature_group_to_add,
# which streamlit-folium re-renders WITHOUT re-sending the base map. The
# whole map pane also runs inside a st.fragment, so a click reruns only
# this pane instead of the entire app.

def get_base_map(snap: dict, show_sar: bool, show_s2: bool,
                 sar_opacity: float) -> folium.Map:
    # Built FRESH on every (fragment) rerun — never cache folium objects:
    # rendering mutates them, and re-rendering a cached Map across reruns
    # accumulates broken HTML until the component blanks out entirely.
    # streamlit-folium doesn't re-render an unchanged-key base map in the
    # browser anyway; only the feature_group_to_add layer updates.
    snap_dir = load_user_settings().snapshots_dir  # active location's folder
    min_lon, min_lat, max_lon, max_lat = snap["scene"]["bbox"]
    m = folium.Map(
        location=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2],
        zoom_start=10,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    overlay = snap_dir / snap.get("sar_overlay", "")
    if show_sar and snap.get("sar_overlay") and overlay.exists():
        folium.raster_layers.ImageOverlay(
            image=str(overlay),
            bounds=[[min_lat, min_lon], [max_lat, max_lon]],
            opacity=sar_opacity,
            name="Sentinel-1 SAR (VV backscatter)",
        ).add_to(m)

    s2 = snap.get("s2")
    if show_s2 and s2 and (snap_dir / s2["overlay"]).exists():
        s2b = s2["bbox"]
        folium.raster_layers.ImageOverlay(
            image=str(snap_dir / s2["overlay"]),
            bounds=[[s2b[1], s2b[0]], [s2b[3], s2b[2]]],
            opacity=sar_opacity,
            name=(f"Sentinel-2 RGB ({s2['acquired_at'][:10]}, "
                  f"{s2['cloud_cover']:.0f}% cloud)"),
        ).add_to(m)

    MiniMap(toggle_display=True).add_to(m)
    return m


def build_vessel_group(vessels: dict) -> folium.FeatureGroup:
    fg = folium.FeatureGroup(name="vessels")
    for v in vessels.values():
        style = STATUS_STYLE[v["status"]]
        det = v.get("detection") or {}
        size_line = ""
        if det.get("length_m"):
            size_line = (f"<br>Size: ~{det['length_m']:.0f} × "
                         f"{det.get('width_m') or 0:.0f} m")
            if det.get("heading_deg") is not None:
                size_line += f" · axis {det['heading_deg']:.0f}°/{(det['heading_deg'] + 180) % 360:.0f}°"
            if det.get("obb"):
                size_line += "<br><i>outline = measured radar footprint</i>"
        popup_html = (
            f"<b>{v['id']}</b> — {style['label']}<br>"
            f"MMSI: {v.get('mmsi') or '—'}<br>"
            f"Position: {v['lat']:.5f}, {v['lon']:.5f}<br>"
            f"Match dist: {v.get('match_dist_m') or '—'} m<br>"
            f"Model conf: {det.get('confidence', '—')}"
            + size_line
            + ("<br><i>manually added</i>" if v.get("manual") else "")
        )
        folium.CircleMarker(
            location=[v["lat"], v["lon"]],
            radius=9 if v["status"] == "DARK" else 7,
            color=style["color"],
            weight=3,
            fill=True,
            fill_opacity=0.35,
            tooltip=vessel_label(v),
            popup=folium.Popup(popup_html, max_width=280),
        ).add_to(fg)

        # Measured radar footprint / detector OBB — the visual answer to
        # "what did it actually measure for this size estimate?"
        if det.get("obb"):
            folium.Polygon(
                locations=[[la, lo] for lo, la in det["obb"]],
                color=style["color"],
                weight=2,
                fill=False,
                dash_array="4",
            ).add_to(fg)
    return fg


# ------------------------------------------------------------------- UI body

init_state()
snaps: list = st.session_state.snaps

with st.sidebar:
    config_panel()
    status_panel()

st.title("🛰️ Dark Vessel Detection")

aoi_picker()

failed = st.session_state.get("last_run_failed_passes")
if failed:
    with st.expander(
        f"⚠ {len(failed)} pass(es) skipped in the last run — the rest "
        "processed normally", expanded=True,
    ):
        st.caption(
            "A pass is skipped (not the whole run aborted) when it fails "
            "individually — most commonly because its radar swath, even "
            "mosaicked across all available frames, genuinely doesn't "
            "overlap the AOI for that particular overpass."
        )
        for f in failed:
            st.markdown(f"**{f['scene_id']}** ({f['acquired_at'][:16].replace('T',' ')} UTC) "
                       f"— {f['reason']}")
    if st.button("Dismiss", key="dismiss_failed_passes",
                 help="Hide this notice. Doesn't affect the run — those "
                     "passes are already skipped either way."):
        st.session_state.pop("last_run_failed_passes", None)
        st.rerun()

if not snaps:
    st.info(
        "No snapshots yet. Draw your area of interest above, then open "
        "**⚙️ Pipeline configuration** in the sidebar, choose your AIS "
        "provider (or pick simulate mode for a test run), and hit "
        "**▶ Run pipeline**."
    )
    st.stop()

# ---------------------------------------------------------- pass navigation
idx = st.session_state.snap_idx
nav_prev, nav_sel, nav_next = st.columns([1, 3, 1])
if nav_prev.button(
    "◀ Previous pass", disabled=idx <= 0, use_container_width=True,
    help="Go to the previous (older) satellite pass. Each pass has its own "
         "scene, AIS window, fusion result, and analyst corrections — your "
         "edits are kept when you navigate away.",
):
    st.session_state.snap_idx = idx - 1
    st.rerun()
if nav_next.button(
    "Next pass ▶", disabled=idx >= len(snaps) - 1, use_container_width=True,
    help="Go to the next (newer) satellite pass.",
):
    st.session_state.snap_idx = idx + 1
    st.rerun()
def _pass_label(i: int) -> str:
    s = snaps[i]
    cov = s["scene"].get("aoi_coverage_pct")
    cov_str = f" · {cov:.0f}% imaged" if cov is not None else ""
    warn = " ⚠" if cov is not None and cov < 20 else ""
    return (f"Pass {i + 1}/{len(snaps)} · "
            f"{s['scene']['acquired_at'][:16].replace('T', ' ')} UTC · "
            f"{s['stats']['dark_vessels']} dark{cov_str}{warn}")


sel = nav_sel.selectbox(
    "Satellite pass", options=list(range(len(snaps))), index=idx,
    format_func=_pass_label,
    label_visibility="collapsed",
    help="Jump directly to any processed pass, ordered oldest → newest. "
         "'% imaged' is how much of your AOI this pass actually covers — "
         "⚠ flags passes where the swath only clips a corner.",
)
if sel != idx:
    st.session_state.snap_idx = sel
    st.rerun()

snap = snaps[st.session_state.snap_idx]
edits = edits_for(snap)
vessels: dict = edits["vessels"]

# ---- BIG simulated-data banner — impossible to mistake for real output
if is_simulated(snap):
    st.markdown(
        """
        <div style="background:repeating-linear-gradient(45deg,#d62728,#d62728 24px,#8f1a1b 24px,#8f1a1b 48px);
                    color:white;text-align:center;padding:14px;border-radius:8px;
                    font-size:2.2em;font-weight:900;letter-spacing:0.25em;margin-bottom:8px;">
            ⚠ SIMULATED DATA ⚠
        </div>
        <p style="text-align:center;color:#d62728;font-weight:600;margin-top:0;">
            Synthetic test scene and AIS traffic — not real observations.
        </p>
        """,
        unsafe_allow_html=True,
    )

frame_note = (f" ({snap['scene']['frame_count']} frames mosaicked)"
             if snap["scene"].get("frame_count", 1) > 1 else "")
st.caption(
    f"Scene **{snap['scene']['scene_id']}**{frame_note} · "
    f"{snap['scene']['platform']} · acquired {snap['scene']['acquired_at']} · "
    f"AIS source `{snap['config']['ais_source']}` · detector "
    f"`{snap['config']['detector_backend']}` · gate "
    f"{snap['config']['match_gate_m']:.0f} m / "
    f"±{snap['config']['temporal_window_s'] // 60} min"
)

cov = snap["scene"].get("aoi_coverage_pct")
if cov is not None and cov < 20:
    st.warning(
        f"📐 **This pass images only {cov:.0f}% of your AOI.** Sentinel-1 "
        "frames are ~250×170 km rotated rectangles along the orbit track; "
        "when an AOI sits near a frame edge or the seam between two "
        "consecutive frames, most of the box falls outside the swath for "
        "that particular overpass — not a rendering issue. Detections "
        "outside the imaged sliver simply don't exist for this pass. Use "
        "◀ ▶ to check other passes (the % imaged is shown per pass), which "
        "may have fuller coverage from a different orbit geometry."
    )

# ---- time-alignment status: does AIS coverage actually bracket T_img?
ta = snap.get("time_alignment")
if ta:
    if ta["pings_in_window"] == 0:
        msg = (
            f"⏱ **No AIS for this pass:** zero pings within "
            f"±{ta['temporal_window_s'] // 60} min of image time {ta['t_img']}"
            + (f" (AIS covers {ta['ais_coverage_start']} → {ta['ais_coverage_end']})."
               if ta["ais_coverage_start"] else ".")
            + " All detections are flagged DARK."
        )
        # Explain WHY when the store provider is in play: a pass from before
        # recording began can never fuse; a pass inside coverage hit a gap.
        if snap["config"].get("ais_source") == "store":
            from backend.diagnostics import store_coverage
            cov = store_coverage(load_user_settings())
            if cov is None:
                msg += (" **Cause: the AIS store is empty** — the collector "
                        "has never recorded anything. Start it and check "
                        "📡 Data source status.")
            elif snap["scene"]["acquired_at"] < cov["start"]:
                msg += (f" **Cause: this pass predates your AIS recording** "
                        f"(store starts {cov['start'][:16]} UTC). Historical "
                        "pings can't be recorded retroactively — this pass "
                        "will stay detection-only. Passes acquired while the "
                        "collector runs will fuse normally.")
            else:
                msg += (f" **Cause: receiver gap** — your store covers "
                        f"{cov['start'][:16]} → {cov['end'][:16]} UTC but has "
                        "no pings around this pass (aisstream volunteer "
                        "coverage is intermittent). See 📡 Data source status.")
        st.error(msg)
    elif ta["vessels_in_window"] < ta["vessels_total"]:
        st.warning(
            f"⏱ Time alignment partial: {ta['vessels_in_window']}/{ta['vessels_total']} "
            f"AIS vessels have pings inside ±{ta['temporal_window_s'] // 60} min "
            f"of image time; the rest cannot be matched."
        )
    else:
        st.caption(
            f"⏱ Time alignment OK — image {ta['t_img']} inside AIS coverage "
            f"({ta['ais_coverage_start']} → {ta['ais_coverage_end']}), "
            f"{ta['pings_in_window']} pings usable."
        )

# ---- analytics panel: placeholder here (top of page), filled after the
# sidebar runs so it can respect the size/status filters defined there
analytics_box = st.container()

# --------------------------------------------------------------- sidebar HIL
with st.sidebar:
    st.header("Layers")
    show_sar = st.toggle(
        "Sentinel-1 SAR", value=True,
        help="Radar backscatter overlay from the analyzed scene. Ships appear "
             "as bright points on dark ocean; works day/night and through "
             "cloud. This is the layer the detector actually saw.",
    )
    show_s2 = st.toggle(
        "Sentinel-2 RGB", value=False,
        help="True-color optical image from the nearest cloud-free "
             "Sentinel-2 pass (fetched automatically, up to ±5 days from the "
             "radar pass). CONTEXT ONLY: it was taken at a different time "
             "than the radar image, so moving ships will NOT be in the same "
             "positions — use it for coastline, ports, anchorages, and "
             "confirming static targets like rigs or buoys.",
    )
    if show_s2 and not snap.get("s2"):
        st.caption("⚠ No cloud-free Sentinel-2 found near this pass "
                   "(or the optical layer is disabled in the pipeline config).")
    sar_opacity = st.slider(
        "SAR opacity", 0.2, 1.0, 0.9, 0.05,
        help="Blend between the radar image and the basemap. 1.0 shows the "
             "pure satellite image (best for eyeballing ship signatures); "
             "lower values let coastlines and place names show through for "
             "orientation. Areas outside the radar swath stay transparent "
             "regardless.",
    )

    st.header("Filters")
    min_ship_len = st.number_input(
        "Min ship length (m)", min_value=0, max_value=500, value=0, step=5,
        help="Hide detections shorter than this from the map and the "
             "analytics counts. 0 = show everything. Detections without a "
             "size estimate stay visible regardless (unknown ≠ small), and "
             "green AIS-only tracks are never filtered by size. Nothing is "
             "deleted — clear the filter to see them again.",
    )
    fcols = st.columns(3)
    show_status = {
        "VERIFIED": fcols[0].checkbox("🔵", value=True, key="flt_verified",
                                      help="Show verified targets (AIS + radar)."),
        "AIS_ONLY": fcols[1].checkbox("🟢", value=True, key="flt_ais",
                                      help="Show AIS-only tracks (broadcasting, not imaged)."),
        "DARK": fcols[2].checkbox("🔴", value=True, key="flt_dark",
                                  help="Show dark vessels (imaged, no AIS)."),
    }

    st.header("Analyst tools")
    mode_click = st.radio(
        "Map click mode",
        ["Inspect", "Add missed vessel (false negative)"],
        help="**Inspect** — clicking the map only opens marker popups.\n\n"
             "**Add missed vessel** — each click on empty water drops a new "
             "DARK contact there. Use when you can see a ship in the SAR "
             "layer that the model failed to detect. Added contacts are "
             "recorded in the correction audit trail.\n\n"
             "In either mode, clicking directly ON a marker selects it and "
             "shows actions under the map: 🗑 Delete, 🔗 Link to AIS track "
             "(dark targets), 🔀 Merge duplicate, and ✏️ size correction.",
    )

    st.divider()
    st.download_button(
        "⬇ Export corrected snapshot",
        data=json.dumps(
            {
                **snap,
                "vessels": list(vessels.values()),
                "corrections": edits["corrections"],
                "exported_at": datetime.now(tz=timezone.utc).isoformat(),
            },
            indent=2,
        ),
        file_name=f"corrected_{snap['scene']['scene_id']}.json",
        mime="application/json",
        use_container_width=True,
        help="Download this pass's snapshot with your corrections applied, "
             "plus the full audit trail — usable as retraining labels.",
    )
    if st.button(
        "↻ Reset this pass to pipeline output", use_container_width=True,
        help="Discard your corrections for the CURRENT pass only and restore "
             "the raw fusion result. Other passes keep their edits.",
    ):
        st.session_state.edits.pop(snap["scene"]["scene_id"], None)
        st.rerun()

def layer_info_panel(snap: dict, show_sar: bool, show_s2: bool) -> None:
    """Metadata for whichever raster layer(s) are currently on the map.

    Critical when both are visible at once: SAR and optical are never from
    the same instant (S1/S2 don't fly together), so an analyst looking at
    two overlapping images needs the acquisition gap front and center, not
    buried in a tooltip.
    """
    if not (show_sar or show_s2):
        return
    s2 = snap.get("s2")
    cols = st.columns(2) if (show_sar and show_s2) else [st.container()]

    if show_sar:
        with cols[0]:
            cov = snap["scene"].get("aoi_coverage_pct")
            frames = snap["scene"].get("frame_count", 1)
            line = f"**📡 Radar — Sentinel-1**  \n{snap['scene']['acquired_at'][:16].replace('T', ' ')} UTC"
            if frames > 1:
                line += f" · {frames} frames mosaicked"
            if cov is not None:
                line += f" · {cov:.0f}% of AOI imaged"
            st.markdown(line)

    if show_s2:
        with cols[1] if (show_sar and show_s2) else cols[0]:
            if s2:
                t_sar = datetime.fromisoformat(snap["scene"]["acquired_at"])
                t_s2 = datetime.fromisoformat(s2["acquired_at"])
                gap_h = abs((t_s2 - t_sar).total_seconds()) / 3600
                gap_str = f"{gap_h / 24:.1f} days" if gap_h >= 24 else f"{gap_h:.0f} hours"
                when = "before" if t_s2 < t_sar else "after"
                st.markdown(
                    f"**🖼️ Optical — Sentinel-2**  \n"
                    f"{s2['acquired_at'][:16].replace('T', ' ')} UTC · "
                    f"{s2['cloud_cover']:.0f}% cloud · "
                    f"**{gap_str} {when} the radar pass**"
                )
            else:
                st.markdown("**🖼️ Optical — Sentinel-2**  \nnot available for this pass")

    if show_sar and show_s2 and s2:
        st.caption(
            "⚠ Different acquisition times (gap shown above) — ship "
            "positions will NOT match between the two layers. Use optical "
            "only for static context: coastline, ports, anchorages, rigs."
        )


# ------------------------------------------------------- filters + analytics

def _passes_filters(v: dict) -> bool:
    if not show_status.get(v["status"], True):
        return False
    if min_ship_len and v["status"] != "AIS_ONLY":
        length = (v.get("detection") or {}).get("length_m")
        if length is not None and length < min_ship_len:
            return False  # unknown sizes stay visible: unknown ≠ small
    return True


visible_vessels = {k: v for k, v in vessels.items() if _passes_filters(v)}

with analytics_box:
    n_ver = sum(1 for v in visible_vessels.values() if v["status"] == "VERIFIED")
    n_dark = sum(1 for v in visible_vessels.values() if v["status"] == "DARK")
    n_ais = sum(1 for v in visible_vessels.values() if v["status"] == "AIS_ONLY")
    total_detected = n_ver + n_dark
    hidden = len(vessels) - len(visible_vessels)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total detected (SAR)", total_detected)
    c2.metric("AIS active", n_ver + n_ais)
    c3.metric("Verified", n_ver)
    c4.metric("Dark vessels", n_dark, delta=None if n_dark == 0 else "⚠ review")
    c5.metric(
        "Dark fleet multiplier",
        f"{(total_detected / n_ver):.2f}×" if n_ver else "—",
        help="Radar-detected fleet size relative to the AIS-verified fleet",
    )
    if hidden:
        st.caption(f"🔎 {hidden} target(s) hidden by the sidebar filters "
                   "(size / status) — nothing is deleted.")

# ------------------------------------------------------------------ map pane
layer_info_panel(snap, show_sar, show_s2)


@st.fragment
def map_pane():
    """Everything a marker click touches lives in this fragment: a click
    reruns ONLY this block, and the heavy base map (tiles + SAR/S2
    overlays) is cached and shipped once — so selecting a ship no longer
    kicks off a whole-app 'loading' cycle. Actual data mutations (delete/
    link/merge/add/resize) escalate to a full rerun on purpose, because
    they change the analytics above."""
    base = get_base_map(snap, show_sar, show_s2, sar_opacity)
    fg = build_vessel_group(visible_vessels)
    map_state = st_folium(
        base, height=640, use_container_width=True,
        key=f"map_{snap['scene']['scene_id']}",
        feature_group_to_add=fg,
        # Only sync clicks back to Python — pan/zoom stay client-side.
        returned_objects=["last_clicked", "last_object_clicked_tooltip"],
    )

    # Click-to-add: act once per distinct click coordinate
    clicked = map_state.get("last_clicked") if map_state else None
    if clicked and mode_click.startswith("Add") and clicked != edits["last_click"]:
        edits["last_click"] = clicked
        new_id = f"m{sum(1 for v in vessels if v.startswith('m')):03d}"
        vessels[new_id] = {
            "id": new_id,
            "status": "DARK",
            "lat": round(clicked["lat"], 6),
            "lon": round(clicked["lng"], 6),
            "mmsi": None,
            "match_dist_m": None,
            "manual": True,
        }
        log_correction(edits, "ADD", vessel_id=new_id,
                       lat=clicked["lat"], lon=clicked["lng"])
        st.rerun(scope="app")

    # Click-to-select: the marker tooltip identifies the vessel
    marker_tooltip = map_state.get("last_object_clicked_tooltip") if map_state else None
    if marker_tooltip and marker_tooltip != edits["last_marker_click"]:
        edits["last_marker_click"] = marker_tooltip
        st.session_state["_selected_marker_id"] = marker_tooltip.split(" · ")[0].strip()

    # Stale IDs (already deleted, other pass, …) resolve to nothing — drop.
    selected_id = st.session_state.get("_selected_marker_id")
    if selected_id and selected_id not in vessels:
        st.session_state["_selected_marker_id"] = None
        selected_id = None
    for key in ("_link_source_id", "_merge_source_id"):
        if st.session_state.get(key) and st.session_state[key] not in vessels:
            st.session_state[key] = None
    link_source_id = st.session_state.get("_link_source_id")
    merge_source_id = st.session_state.get("_merge_source_id")

    def clear_selection():
        for key in ("_selected_marker_id", "_link_source_id", "_merge_source_id"):
            st.session_state[key] = None

    if link_source_id:
        src = vessels[link_source_id]
        lc1, lc2 = st.columns([4, 1])
        lc1.info(f"🔗 Linking **{vessel_label(src)}** — click a green "
                 "AIS-only marker on the map to complete the link.")
        if lc2.button("Cancel", key="cancel_linking", use_container_width=True,
                      help="Abort linking; nothing changes."):
            clear_selection()
            st.rerun(scope="fragment")
        elif selected_id and selected_id != link_source_id:
            target = vessels[selected_id]
            if target["status"] == "AIS_ONLY":
                if st.button(
                    f"✔ Confirm: link {vessel_label(src)} ↔ {vessel_label(target)}",
                    key="confirm_link", use_container_width=True,
                    help="Merges these into one VERIFIED target carrying the "
                         "AIS track's MMSI. Recorded in the audit trail.",
                ):
                    relink_vessel(vessels, edits, link_source_id, selected_id)
                    clear_selection()
                    st.rerun(scope="app")
            else:
                st.warning(f"{vessel_label(target)} isn't an AIS-only (green) "
                           "track — click a green marker to complete the link.")
    elif merge_source_id:
        src = vessels[merge_source_id]
        mc1, mc2 = st.columns([4, 1])
        mc1.info(f"🔀 Merging **{vessel_label(src)}** — click the duplicate "
                 "detection (🔵/🔴 marker) it should be merged with.")
        if mc2.button("Cancel", key="cancel_merge", use_container_width=True,
                      help="Abort merging; nothing changes."):
            clear_selection()
            st.rerun(scope="fragment")
        elif selected_id and selected_id != merge_source_id:
            target = vessels[selected_id]
            if target.get("detection") or target.get("manual"):
                if st.button(
                    f"✔ Confirm: merge {vessel_label(src)} + {vessel_label(target)}",
                    key="confirm_merge", use_container_width=True,
                    help="Treats these two markers as ONE physical ship "
                         "(double counting happens on chip seams or split "
                         "radar blobs). The stronger record survives — "
                         "verified > has-AIS > higher confidence — and "
                         "inherits the other's attributes. Recorded in the "
                         "audit trail.",
                ):
                    merge_vessels(vessels, edits, merge_source_id, selected_id)
                    clear_selection()
                    st.rerun(scope="app")
            else:
                st.warning(f"{vessel_label(target)} is an AIS-only track — "
                           "merging is for duplicate radar detections. Use "
                           "🔗 Link for detection↔AIS pairing.")
    elif selected_id:
        sv = vessels[selected_id]
        style = STATUS_STYLE[sv["status"]]
        det = sv.get("detection")
        can_delete = sv["status"] in ("DARK", "VERIFIED")
        can_link = sv["status"] == "DARK" and any(
            v["status"] == "AIS_ONLY" for v in vessels.values())
        can_merge = bool(det or sv.get("manual"))
        n_btns = 1 + can_delete + can_link + can_merge + bool(det)
        cols = st.columns([3] + [1] * n_btns)
        cols[0].markdown(f"**Selected:** {vessel_label(sv)} — {style['label']}")
        ci = 1
        if can_delete:
            if cols[ci].button(
                "🗑 Delete", key="delete_selected_marker", use_container_width=True,
                help="Remove this detection. A VERIFIED target's AIS track is "
                     "kept as a green 'AIS only' marker, not deleted with it.",
            ):
                delete_vessel(vessels, edits, selected_id)
                clear_selection()
                st.rerun(scope="app")
            ci += 1
        if can_link:
            if cols[ci].button(
                "🔗 Link", key="start_link", use_container_width=True,
                help="Pair this unmatched radar detection with an AIS track: "
                     "click this, then click the green (AIS-only) marker it "
                     "belongs to.",
            ):
                st.session_state["_link_source_id"] = selected_id
                st.session_state["_selected_marker_id"] = None
                st.rerun(scope="fragment")
            ci += 1
        if can_merge:
            if cols[ci].button(
                "🔀 Merge", key="start_merge", use_container_width=True,
                help="This ship was counted twice? Click this, then click "
                     "the duplicate marker — the two records merge into one.",
            ):
                st.session_state["_merge_source_id"] = selected_id
                st.session_state["_selected_marker_id"] = None
                st.rerun(scope="fragment")
            ci += 1
        if det:
            with cols[ci].popover("✏️ Size", use_container_width=True,
                                  help="Correct the estimated ship size. The "
                                       "dashed outline on the map shows the "
                                       "radar footprint the estimate came "
                                       "from — if it grabbed a sidelobe or "
                                       "merged two hulls, fix the numbers "
                                       "here (recorded in the audit trail)."):
                new_len = st.number_input(
                    "Length (m)", min_value=5.0, max_value=500.0,
                    value=float(det.get("length_m") or 50.0), step=5.0,
                    key="resize_len")
                new_beam = st.number_input(
                    "Beam (m)", min_value=2.0, max_value=80.0,
                    value=float(det.get("width_m") or 15.0), step=1.0,
                    key="resize_beam")
                if st.button("Save size", key="resize_save"):
                    resize_vessel(vessels, edits, selected_id, new_len, new_beam)
                    st.rerun(scope="app")
            ci += 1
        if cols[ci].button("Deselect", key="deselect_marker",
                           use_container_width=True,
                           help="Clear the selection without changing anything."):
            clear_selection()
            st.rerun(scope="fragment")


map_pane()

if edits["corrections"]:
    with st.expander(f"Correction audit trail ({len(edits['corrections'])})"):
        st.json(edits["corrections"])
