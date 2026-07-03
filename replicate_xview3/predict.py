"""Replicate predictor: xView3 2nd-place TimmUnet (resnet34) vessel detector.

Input contract (fixed — the sat_tracker client depends on it):
    chip:       .npz file containing float32 arrays 'vv_db' and 'vh_db'
                (calibrated Sentinel-1 sigma0 in dB, same H×W, native
                ~10 m/px resolution; 800×800 recommended)
    confidence: center-head threshold 0..255 (default 100, the value the
                original solution used)

Output:
    {"detections": [{"y": row, "x": col, "score": mean center activation,
                     "is_vessel": bool, "length_m": float}, ...],
     "height": H, "width": W}
"""
from __future__ import annotations

import numpy as np
import torch
from cog import BasePredictor, Input, Path
from scipy.ndimage import binary_dilation, label as ndlabel

from unet import TimmUnet


def normalize_band(band: np.ndarray) -> np.ndarray:
    """Exact normalization the solution trained with: (dB + 40) / 15,
    nodata → 0."""
    band = band.copy()
    band[band < -32760] = -100
    ignored = band == -100
    band = (band + 40.0) / 15.0
    band[ignored] = 0.0
    return band


class Predictor(BasePredictor):
    def setup(self):
        model = TimmUnet(encoder="resnet34", in_chans=2, pretrained=False)
        # weights_only=False: the checkpoint (official xView3 2nd-place
        # release) embeds numpy scalars, which PyTorch ≥2.6 rejects under
        # its safe-load default. Trusted source, baked into this image.
        ckpt = torch.load("model.bin", map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        model.eval()
        self.model = model
        torch.set_num_threads(4)

    def predict(
        self,
        chip: Path = Input(description=".npz with float32 'vv_db' and 'vh_db' (sigma0 dB)"),
        confidence: float = Input(default=100.0, ge=0, le=255,
                                  description="center-head threshold"),
    ) -> dict:
        data = np.load(str(chip))
        vv = normalize_band(data["vv_db"].astype("float32"))
        vh = normalize_band(data["vh_db"].astype("float32"))
        x = torch.from_numpy(np.stack([vv, vh]))[None].float()

        with torch.no_grad():
            out = self.model(x)
        center = out["center_mask"][0, 0].numpy()
        vessel = torch.sigmoid(out["vessel_mask"][0, 0]).numpy() * 255
        length = out["length_mask"][0, 0].numpy()

        labeled, n = ndlabel(binary_dilation(center > confidence))
        detections = []
        for i in range(1, n + 1):
            region = labeled == i
            ys, xs = np.nonzero(region)
            detections.append({
                "y": float(ys.mean()),
                "x": float(xs.mean()),
                "score": float(center[region].mean()),
                "is_vessel": bool((vessel[region] > 100).sum() > 8),
                "length_m": float(length[region].max()),
            })
        h, w = vv.shape
        return {"detections": detections, "height": h, "width": w}
