"""Hosted-inference adapters — same `VesselDetector` contract as the local
YOLO backend, so teams preferring managed infra flip `DETECTOR_BACKEND` and
provide credentials; nothing downstream changes.
"""
from __future__ import annotations

import base64
import io
import logging
import re

import numpy as np

from backend.config import settings
from backend.detection.base import Detection, VesselDetector
from backend.ingestion.stac_client import Scene

log = logging.getLogger(__name__)


ROBOFLOW_INFER_HOST = "serverless.roboflow.com"  # current v2 endpoint — required
# for newer architectures (e.g. RF-DETR); confirmed a drop-in replacement for
# the legacy detect.roboflow.com v1 host on classic models too.


def normalize_roboflow_model_id(raw: str) -> str:
    """Accept whatever Roboflow's own UI actually hands a user, and reduce
    it to the bare 'project/version' the hosted inference API expects.

    Roboflow never shows a user that bare form directly — they get one of:
      - a Universe page URL:   universe.roboflow.com/<workspace>/<project>/model/<version>
      - a Deploy-tab curl/SDK snippet built around:
                                serverless.roboflow.com/<project>/<version>?api_key=...
      - or just 'project/version' if copied by hand.
    This strips scheme/host, query string, and a leading workspace segment
    (recognizable because Universe URLs insert a literal 'model' segment
    before the version) so any of the above pastes into the same field.
    """
    s = raw.strip()
    if not s:
        return s
    s = re.sub(r"^\w+://", "", s)                      # drop scheme
    s = re.sub(r"^(universe|app|detect|serverless)\.roboflow\.com/", "", s)
    s = s.split("?")[0].strip("/")                      # drop query string
    parts = [p for p in s.split("/") if p]
    if "model" in parts:                                # Universe URL shape
        parts = [p for p in parts if p != "model"]
        parts = parts[-2:]                              # project, version
    elif len(parts) >= 3:                                # workspace/project/version
        parts = parts[-2:]
    return "/".join(parts)


def validate_roboflow_model_id(model_id: str) -> str | None:
    """None if `model_id` looks like a valid 'project/version' reference,
    else a human-readable reason. Catches the single most common mistake —
    pasting a project slug with no trailing version number — BEFORE it
    reaches the API, where it produces an opaque 405 rather than a clear
    'you forgot the version' message.
    """
    if not model_id:
        return "no model id set"
    if "/" not in model_id:
        return (f"'{model_id}' has no version number — Roboflow ids need a "
                f"trailing /<version>, e.g. '{model_id}/1'. Check the exact "
                "id in your project's Deploy tab.")
    project, _, version = model_id.rpartition("/")
    if not version.isdigit():
        return (f"'{model_id}' — the segment after the last '/' should be a "
                f"version number (got '{version}'). Check the exact id in "
                "your project's Deploy tab.")
    return None


def _scene_to_png_bytes(scene: Scene) -> bytes:
    from PIL import Image

    lo, hi = np.percentile(scene.image, [2, 98])
    img8 = (np.clip((scene.image - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img8).save(buf, format="PNG")
    return buf.getvalue()


class RoboflowDetector(VesselDetector):
    """Roboflow Hosted Inference — cheapest managed option; upload chips,
    get boxes back. Works with your own trained model or any public
    Universe model, e.g. 'sar-ship-dataset-ogvbz/1' (trained on the
    39k-image SAR-Ship-Dataset).

    Scenes are tiled before upload: Roboflow resizes each request to the
    model's input size, so a full 2048 px scene sent whole would shrink a
    20 px ship below detectability. Tiles overlap and duplicates are removed
    with a centroid NMS. One API call per tile.
    """

    TILE = 1024
    OVERLAP = 128
    NMS_RADIUS_PX = 12
    CONFIDENCE = 35  # percent

    def __init__(self, api_key: str | None = None, model_id: str | None = None):
        self.api_key = api_key or settings.roboflow_api_key
        self.model_id = normalize_roboflow_model_id(model_id or settings.roboflow_model_id)
        problem = validate_roboflow_model_id(self.model_id)
        if problem:
            raise ValueError(f"Roboflow model id invalid: {problem}")
        if not self.api_key:
            raise ValueError("Roboflow API key not set.")

    def detect(self, scene: Scene) -> list[Detection]:
        lo, hi = np.percentile(scene.image, [2, 98])
        img8 = (np.clip((scene.image - lo) / max(hi - lo, 1e-6), 0, 1) * 255
                ).astype(np.uint8)
        h, w = img8.shape

        raw: list[tuple[float, float, float]] = []  # (row, col, conf)
        step = self.TILE - self.OVERLAP
        calls = 0
        for r0 in range(0, max(1, h - self.OVERLAP), step):
            for c0 in range(0, max(1, w - self.OVERLAP), step):
                tile = img8[r0:r0 + self.TILE, c0:c0 + self.TILE]
                if tile.shape[0] < 32 or tile.shape[1] < 32:
                    continue
                for p in self._infer(tile):
                    raw.append((r0 + p["y"], c0 + p["x"], p["confidence"]))
                calls += 1

        kept: list[tuple[float, float, float]] = []
        for cand in sorted(raw, key=lambda t: -t[2]):
            if all((cand[0] - k[0]) ** 2 + (cand[1] - k[1]) ** 2
                   > self.NMS_RADIUS_PX ** 2 for k in kept):
                kept.append(cand)

        detections = []
        for cy, cx, conf in kept:
            lon, lat = scene.pixel_to_lonlat(cy, cx)
            detections.append(
                Detection(lat=lat, lon=lon, confidence=float(conf),
                          source_model=f"roboflow:{self.model_id}")
            )
        log.info("Roboflow: %d detections from %d tile calls", len(detections), calls)
        return detections

    def _infer(self, tile: np.ndarray) -> list[dict]:
        import io
        import requests
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(tile).save(buf, format="PNG")
        resp = requests.post(
            f"https://{ROBOFLOW_INFER_HOST}/{self.model_id}",
            params={"api_key": self.api_key, "confidence": self.CONFIDENCE},
            data=base64.b64encode(buf.getvalue()),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=120,
        )
        if resp.status_code == 404:
            raise ValueError(
                f"Roboflow model '{self.model_id}' not found (404) — this "
                "API key's workspace doesn't have a project/version by that "
                "exact id. Roboflow's 'Model URL' copy button can show a "
                "descriptive name that ISN'T the real id (e.g. "
                "'find-ship-and-water-1-rfdetr-medium-t1' when the working "
                "id is 'find-ship-and-water/1') — check the project's "
                "Versions page or its Python/cURL snippet for the exact "
                "slug/version, or use a public Universe model such as "
                "'sar-ship-dataset-ogvbz/1'."
            )
        if resp.status_code == 405:
            raise ValueError(
                f"Roboflow model '{self.model_id}' returned 405 Method Not "
                "Allowed — Roboflow didn't accept this id/version shape. "
                "Copy the id straight from the project's Deploy tab rather "
                "than typing it by hand."
            )
        resp.raise_for_status()
        return resp.json().get("predictions", [])


class VertexAIDetector(VesselDetector):
    """Google Cloud Vertex AI custom-trained image object detection endpoint.

    Assumes an AutoML/custom model deployed to an endpoint; set
    GOOGLE_APPLICATION_CREDENTIALS and the endpoint name below or via env.
    """

    def __init__(self, endpoint: str | None = None):
        import os

        self.endpoint_name = endpoint or os.getenv("VERTEX_ENDPOINT", "")
        if not self.endpoint_name:
            raise ValueError("VERTEX_ENDPOINT not configured")

    def detect(self, scene: Scene) -> list[Detection]:
        from google.cloud import aiplatform

        endpoint = aiplatform.Endpoint(self.endpoint_name)
        instance = {
            "content": base64.b64encode(_scene_to_png_bytes(scene)).decode()
        }
        prediction = endpoint.predict(instances=[instance]).predictions[0]

        h, w = scene.image.shape
        detections = []
        for box, conf in zip(
            prediction.get("bboxes", []), prediction.get("confidences", [])
        ):
            # Vertex returns [xMin, xMax, yMin, yMax] normalized 0..1
            cx_px = (box[0] + box[1]) / 2 * w
            cy_px = (box[2] + box[3]) / 2 * h
            lon, lat = scene.pixel_to_lonlat(cy_px, cx_px)
            detections.append(
                Detection(lat=lat, lon=lon, confidence=conf, source_model="vertex-ai")
            )
        log.info("Vertex AI: %d detections", len(detections))
        return detections
