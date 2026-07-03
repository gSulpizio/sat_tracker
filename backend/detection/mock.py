"""Placeholder detector for offline/simulation runs.

In simulate mode the pipeline injects vessel signatures into the synthetic
scene and registers their true positions here; `detect` returns them with
pixel-quantization noise, mimicking a real model's localization error.
Also demonstrates a false positive (a moored buoy / rock the model would
plausibly fire on) so the delete workflow in the UI has something to do.
"""
from __future__ import annotations

import numpy as np

from backend.detection.base import Detection, VesselDetector
from backend.ingestion.stac_client import Scene


class MockDetector(VesselDetector):
    def __init__(self):
        self._planted: list[tuple[float, float]] = []  # (lon, lat) truly imaged
        self._false_positives: list[tuple[float, float]] = []

    def plant(self, lonlats: list[tuple[float, float]],
              false_positives: list[tuple[float, float]] | None = None) -> None:
        self._planted = lonlats
        self._false_positives = false_positives or []

    def detect(self, scene: Scene) -> list[Detection]:
        rng = np.random.default_rng(11)
        out = []
        for lon, lat in self._planted:
            out.append(
                Detection(
                    lat=lat + float(rng.normal(0, 4e-5)),   # ~5 m localization noise
                    lon=lon + float(rng.normal(0, 4e-5)),
                    confidence=float(rng.uniform(0.62, 0.97)),
                    source_model="mock",
                )
            )
        for lon, lat in self._false_positives:
            out.append(
                Detection(lat=lat, lon=lon, confidence=float(rng.uniform(0.4, 0.6)),
                          source_model="mock"),
            )
        return out
