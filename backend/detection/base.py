"""Detector abstraction: every backend (local YOLO, hosted APIs, mock)
implements `VesselDetector`, so swapping infra is a config change, not a
code change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from backend.ingestion.stac_client import Scene


@dataclass
class Detection:
    lat: float
    lon: float
    confidence: float
    # Oriented bounding box corners as [(lon, lat), ...] — optional; point
    # detectors (or CFAR prescreeners) may only produce a centroid.
    obb_lonlat: list[tuple[float, float]] = field(default_factory=list)
    length_m: float | None = None
    source_model: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "type": "detected_ship",
            "confidence": round(self.confidence, 3),
            "obb": [[round(x, 6), round(y, 6)] for x, y in self.obb_lonlat],
            "length_m": self.length_m,
            "source_model": self.source_model,
        }


class VesselDetector(ABC):
    """Contract: take a georeferenced Scene, return geographic detections."""

    @abstractmethod
    def detect(self, scene: Scene) -> list[Detection]:
        ...


def build_detector(backend: str, cfg=None) -> VesselDetector:
    """Factory keyed on config — the only place backends are named."""
    from backend.config import settings

    cfg = cfg or settings
    if backend == "yolo":
        from backend.detection.yolo_sar import YoloObbDetector
        return YoloObbDetector(weights=cfg.yolo_weights)
    if backend == "roboflow":
        from backend.detection.hosted_api import RoboflowDetector
        return RoboflowDetector(api_key=cfg.roboflow_api_key, model_id=cfg.roboflow_model_id)
    if backend == "vertex":
        from backend.detection.hosted_api import VertexAIDetector
        return VertexAIDetector()
    if backend == "mock":
        from backend.detection.mock import MockDetector
        return MockDetector()
    raise ValueError(f"Unknown detector backend: {backend!r}")
