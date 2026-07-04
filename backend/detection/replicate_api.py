"""Replicate-hosted detector backend — xView3 2nd-place TimmUnet.

Talks to the model WE package and push via Cog (replicate_xview3/ in this
repo), so the I/O contract is fixed:

    input:  chip = .npz file with float32 'vv_db'/'vh_db' (sigma0 dB,
            native ~10 m/px), confidence = center threshold (0..255)
    output: {"detections": [{"y","x","score","is_vessel","length_m"}, ...]}

Unlike the Roboflow backend (which tiles the decimated display image),
this one re-reads NATIVE-resolution VV+VH windows from the source COGs —
the xView3 model was trained on native 10 m pixels and both polarizations,
and it also regresses vessel length directly, which the pipeline carries
into the UI.

Replicate bills per second with scale-to-zero: for a bursty on-demand
pipeline (tens of CPU chips per pass, a few passes a week) that's cents
per month; cold starts of ~10–30 s don't matter here.
"""
from __future__ import annotations

import io
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from backend.config import Settings, settings
from backend.detection.base import Detection, VesselDetector
from backend.ingestion.stac_client import Scene, sibling_polarization_href

log = logging.getLogger(__name__)

API_ROOT = "https://api.replicate.com/v1"

CHIP = 800            # matches the shard size the model was trained on
STRIDE = 704          # 96 px overlap so ships on seams aren't cut
CONFIDENCE = 100.0    # center-head threshold the original solution used
DEDUPE_M = 120.0      # cross-chip/frame duplicate radius, metres
# All radiometric preprocessing (amplitude→dB, sea-percentile pseudo-
# calibration, normalization) lives INSIDE the Replicate predictor now —
# the client just ships raw-amplitude 2-band float32 TIFF chips, the same
# format any other user of the public model would send.


class ReplicateDetector(VesselDetector):
    def __init__(self, api_token: str | None = None, model: str | None = None,
                 cfg: Settings | None = None):
        self.cfg = cfg or settings
        self.api_token = api_token or self.cfg.replicate_api_token
        self.model = (model or self.cfg.replicate_model).strip().strip("/")
        if not self.api_token:
            raise ValueError("Replicate API token not set.")
        if not self.model or "/" not in self.model:
            raise ValueError(
                "Replicate model not set — use 'owner/name' as shown on "
                "the model's Replicate page."
            )
        self._version: str | None = None  # resolved lazily, cached

    def _latest_version(self) -> str:
        """Private/community models can't use the model-scoped predictions
        endpoint (that's official-models only — it 404s otherwise); we must
        create predictions against an explicit version id instead."""
        if self._version:
            return self._version
        import requests

        r = requests.get(f"{API_ROOT}/models/{self.model}",
                         headers={"Authorization": f"Bearer {self.api_token}"},
                         timeout=30)
        if r.status_code == 404:
            raise ValueError(
                f"Replicate model '{self.model}' not found — check "
                "owner/name and that your token belongs to that account.")
        r.raise_for_status()
        version = (r.json().get("latest_version") or {}).get("id")
        if not version:
            raise ValueError(
                f"Replicate model '{self.model}' has no pushed version yet "
                "— run `cog push` first (see replicate_xview3/).")
        self._version = version
        return version

    # ------------------------------------------------------------- chips
    def _collect_chips(self, scene: Scene) -> list[dict]:
        """Serially read native-res VV+VH chip pairs covering AOI∩frame
        (VRT reads aren't thread-safe; the network inference is what's
        worth parallelizing, not these small window reads)."""
        import rasterio
        from rasterio.errors import WindowError
        from rasterio.vrt import WarpedVRT
        from rasterio.windows import Window, from_bounds

        from backend.ingestion.stac_client import _gdal_env

        chips = []
        for vv_href in (scene.asset_hrefs or [scene.asset_href]):
            with rasterio.Env(**_gdal_env(vv_href, self.cfg)):
                vh_href = next(
                    (c for c in sibling_polarization_href(vv_href, "vh")
                     if _openable(c)), None)
                if vh_href is None:
                    log.warning("No VH sibling for %s — skipping frame "
                                "(model needs both polarizations)",
                                vv_href.rsplit("/", 1)[-1][:40])
                    continue
                with rasterio.open(vv_href) as vv_src, \
                        rasterio.open(vh_href) as vh_src:
                    vv_vrt = _to_4326(vv_src, WarpedVRT)
                    vh_vrt = _to_4326(vh_src, WarpedVRT)
                    try:
                        try:
                            win = from_bounds(*scene.bbox,
                                              transform=vv_vrt.transform)
                            win = win.intersection(
                                Window(0, 0, vv_vrt.width, vv_vrt.height)
                            ).round_offsets().round_lengths()
                        except WindowError:
                            continue
                        for r0 in range(int(win.row_off),
                                        int(win.row_off + win.height), STRIDE):
                            for c0 in range(int(win.col_off),
                                            int(win.col_off + win.width), STRIDE):
                                h = min(CHIP, vv_vrt.height - r0)
                                w = min(CHIP, vv_vrt.width - c0)
                                if h < 64 or w < 64:
                                    continue
                                vv = vv_vrt.read(1, window=Window(c0, r0, w, h),
                                                 out_dtype="float32")
                                if (vv > 0).mean() < 0.05:
                                    continue  # outside the swath
                                vh = vh_vrt.read(1, window=Window(c0, r0, w, h),
                                                 out_dtype="float32")
                                chips.append({
                                    "vv": vv, "vh": vh,
                                    "transform": vv_vrt.window_transform(
                                        Window(c0, r0, w, h)),
                                })
                    finally:
                        for v, s in ((vv_vrt, vv_src), (vh_vrt, vh_src)):
                            if v is not s:
                                v.close()
        return chips

    # --------------------------------------------------------------- API
    def _upload_tiff(self, vv: np.ndarray, vh: np.ndarray) -> str:
        import requests
        import tifffile

        buf = io.BytesIO()
        tifffile.imwrite(buf, np.stack([vv, vh]).astype("float32"))
        buf.seek(0)
        resp = requests.post(
            f"{API_ROOT}/files",
            headers={"Authorization": f"Bearer {self.api_token}"},
            files={"content": ("chip.tif", buf, "image/tiff")},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["urls"]["get"]

    def _predict_chip(self, chip: dict) -> list[Detection]:
        import requests

        # Raw amplitude straight from the COGs — the predictor owns
        # amplitude→dB conversion and sea-anchored pseudo-calibration.
        file_url = self._upload_tiff(chip["vv"], chip["vh"])

        # Exact ground sampling of this (warped) chip, so the model's
        # length regression is scaled correctly.
        tfm = chip["transform"]
        _, center_lat = tfm * (chip["vv"].shape[1] / 2, chip["vv"].shape[0] / 2)
        mpp_x = abs(tfm.a) * 111_320.0 * math.cos(math.radians(center_lat))
        mpp_y = abs(tfm.e) * 110_540.0
        pixel_spacing = round((mpp_x + mpp_y) / 2, 2)

        resp = requests.post(
            f"{API_ROOT}/predictions",
            headers={"Authorization": f"Bearer {self.api_token}",
                     "Prefer": "wait=60"},
            json={"version": self._latest_version(),
                  "input": {"image": file_url, "confidence": CONFIDENCE,
                            "pixel_spacing_m": pixel_spacing}},
            timeout=90,
        )
        if resp.status_code == 402:
            raise ValueError(
                "Replicate returned 402 Payment Required — add billing at "
                "replicate.com/account/billing.")
        resp.raise_for_status()
        pred = resp.json()

        deadline = time.monotonic() + 300  # allow for a cold start
        while pred.get("status") in ("starting", "processing"):
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Replicate prediction {pred.get('id')} still "
                    f"'{pred['status']}' after 5 min.")
            time.sleep(3)
            poll = requests.get(pred["urls"]["get"],
                                headers={"Authorization": f"Bearer {self.api_token}"},
                                timeout=30)
            poll.raise_for_status()
            pred = poll.json()
        if pred.get("status") != "succeeded":
            raise RuntimeError(
                f"Replicate prediction failed: "
                f"{pred.get('error') or pred.get('status')}")

        out = []
        vv = chip["vv"]
        h_chip, w_chip = vv.shape
        for d in (pred.get("output") or {}).get("detections", []):
            if not d.get("is_vessel", True):
                continue  # fixed structure (platform/rig), not a ship
            # Belt-and-braces against swath-edge phantoms: a ship can't be
            # where the radar recorded nothing. (The predictor filters
            # these too, but the client has the raw chip right here.)
            cy, cx = int(round(d["y"])), int(round(d["x"]))
            if not (0 <= cy < h_chip and 0 <= cx < w_chip) or vv[cy, cx] == 0:
                continue
            lon, lat = tfm * (d["x"], d["y"])
            length = d.get("length_m")
            out.append(Detection(
                lat=lat, lon=lon,
                confidence=round(min(float(d["score"]) / 255.0, 1.0), 3),
                length_m=round(length, 1) if length and length > 0 else None,
                source_model=f"replicate:{self.model}",
            ))
        return out

    # ------------------------------------------------------------ detect
    def detect(self, scene: Scene) -> list[Detection]:
        chips = self._collect_chips(scene)
        log.info("Replicate: %d native-res chips to infer", len(chips))
        all_dets: list[Detection] = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for dets in pool.map(self._predict_chip, chips):
                all_dets.extend(dets)

        # Cross-chip/frame dedupe on ground distance
        kept: list[Detection] = []
        for cand in sorted(all_dets, key=lambda d: -d.confidence):
            if all(_ground_dist_m(cand, k) > DEDUPE_M for k in kept):
                kept.append(cand)
        log.info("Replicate: %d detections after dedupe (%d raw)",
                 len(kept), len(all_dets))
        return kept


def _to_4326(src, WarpedVRT):
    gcps, gcp_crs = src.gcps
    if src.crs is None and gcps:
        return WarpedVRT(src, src_crs=gcp_crs, crs="EPSG:4326")
    if src.crs is not None and src.crs.to_epsg() != 4326:
        return WarpedVRT(src, crs="EPSG:4326")
    return src


def _openable(href: str) -> bool:
    import rasterio

    try:
        with rasterio.open(href):
            return True
    except Exception:
        return False


def _ground_dist_m(a: Detection, b: Detection) -> float:
    mean_lat = math.radians((a.lat + b.lat) / 2)
    dx = (a.lon - b.lon) * 111_320.0 * math.cos(mean_lat)
    dy = (a.lat - b.lat) * 110_540.0
    return math.hypot(dx, dy)
