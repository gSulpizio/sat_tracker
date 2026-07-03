"""Open-source detector: YOLOv8-OBB fine-tuned on SAR ship datasets.

Training recipe (run once, offline):

    # 1. Get data — xView3-SAR (best fit: S1 GRD + dark-vessel labels),
    #    HRSID, or LS-SSDD. Convert labels to YOLO OBB format
    #    (class cx cy w h angle, normalized).
    # 2. Preprocess: dB-scale chips, 2–98 percentile stretch to uint8,
    #    tile to 1024x1024 with 128 px overlap (ships near tile edges).
    # 3. Train:
    #       yolo obb train model=yolov8m-obb.pt data=sar_ships.yaml \
    #            imgsz=1024 epochs=100 batch=8 degrees=180 mosaic=0.5
    #    (rotation augmentation matters — SAR scenes have arbitrary
    #    orbit-relative headings.)
    # 4. Export weights to models/yolov8m-obb-sar.pt (YOLO_WEIGHTS env var).

Inference here tiles the scene the same way, runs the model per tile, maps
OBB pixel corners back to lon/lat via the scene geotransform, and applies
cross-tile non-max suppression on centroids.
"""
from __future__ import annotations

import logging

import numpy as np

from backend.config import settings
from backend.detection.base import Detection, VesselDetector
from backend.ingestion.stac_client import Scene

log = logging.getLogger(__name__)

TILE = 1024
OVERLAP = 128
NMS_RADIUS_PX = 12  # duplicate suppression across tile overlaps


class YoloObbDetector(VesselDetector):
    def __init__(self, weights: str | None = None, conf: float = 0.35):
        from ultralytics import YOLO  # deferred: heavy import, optional dep

        self.model = YOLO(weights or settings.yolo_weights)
        self.conf = conf

    def detect(self, scene: Scene) -> list[Detection]:
        img8 = _to_uint8(scene.image)
        rgb = np.stack([img8] * 3, axis=-1)  # YOLO expects 3 channels
        h, w = img8.shape

        raw: list[tuple[float, float, float, np.ndarray]] = []  # (row, col, conf, corners_px)
        step = TILE - OVERLAP
        for r0 in range(0, max(1, h - OVERLAP), step):
            for c0 in range(0, max(1, w - OVERLAP), step):
                tile = rgb[r0 : r0 + TILE, c0 : c0 + TILE]
                if tile.size == 0:
                    continue
                results = self.model.predict(tile, conf=self.conf, verbose=False)
                for res in results:
                    if res.obb is None:
                        continue
                    corners = res.obb.xyxyxyxy.cpu().numpy()   # (n, 4, 2) x,y px
                    confs = res.obb.conf.cpu().numpy()
                    for pts, cf in zip(corners, confs):
                        pts = pts + [c0, r0]                   # tile → scene px
                        cy, cx = pts[:, 1].mean(), pts[:, 0].mean()
                        raw.append((cy, cx, float(cf), pts))

        detections = []
        for cy, cx, cf, pts in _nms_by_centroid(raw):
            lon, lat = scene.pixel_to_lonlat(cy, cx)
            detections.append(
                Detection(
                    lat=lat,
                    lon=lon,
                    confidence=cf,
                    obb_lonlat=[scene.pixel_to_lonlat(y, x) for x, y in pts],
                    length_m=_obb_length_m(pts, scene),
                    source_model="yolov8m-obb-sar",
                )
            )
        log.info("YOLO OBB: %d detections after NMS", len(detections))
        return detections


def _to_uint8(db_image: np.ndarray) -> np.ndarray:
    """Percentile stretch matching the training preprocessing."""
    lo, hi = np.percentile(db_image, [2, 98])
    scaled = np.clip((db_image - lo) / max(hi - lo, 1e-6), 0, 1)
    return (scaled * 255).astype(np.uint8)


def _nms_by_centroid(raw: list[tuple]) -> list[tuple]:
    """Greedy centroid NMS to dedupe detections in tile-overlap zones."""
    kept: list[tuple] = []
    for cand in sorted(raw, key=lambda t: -t[2]):
        if all(
            (cand[0] - k[0]) ** 2 + (cand[1] - k[1]) ** 2 > NMS_RADIUS_PX**2
            for k in kept
        ):
            kept.append(cand)
    return kept


def _obb_length_m(pts_px: np.ndarray, scene: Scene) -> float:
    """Longest OBB side in metres (coarse — assumes small chips)."""
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    lengths = []
    for i in range(4):
        lon1, lat1 = scene.pixel_to_lonlat(pts_px[i, 1], pts_px[i, 0])
        j = (i + 1) % 4
        lon2, lat2 = scene.pixel_to_lonlat(pts_px[j, 1], pts_px[j, 0])
        _, _, d = geod.inv(lon1, lat1, lon2, lat2)
        lengths.append(d)
    return round(max(lengths), 1)
