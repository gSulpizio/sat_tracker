"""Replicate predictor: xView3 2nd-place TimmUnet (resnet34) vessel detector.

Accepts real-world inputs and owns ALL preprocessing, so the public model
page works by drag-and-drop:

    image:  one of
            - GeoTIFF/TIFF (.tif/.tiff): 1 or 2 bands, float or integer.
              Band 1 = VV, band 2 = VH (if present). Values may be raw
              amplitude/DN, linear intensity, or already-calibrated σ0 dB —
              detected automatically.
            - PNG/JPEG: grayscale quicklook (8-bit). Works, but a
              display-stretched image compresses radar dynamic range, so
              expect degraded accuracy vs. a float TIFF.
            - .npz with float32 'vv_db'/'vh_db' (legacy sat_tracker chips).
    pixel_spacing_m: ground sampling distance of the input (default 10,
            Sentinel-1 IW GRD). Length estimates scale with this.
    confidence: detection threshold on the center head, 0–255.

Preprocessing pipeline (applied per band):
    1. amplitude/DN → dB when the value range says the data isn't dB yet
    2. pseudo-calibration: anchor the 30th percentile (sea is the darkest
       extended surface, robust to ≲2/3 land in frame) to typical calm-sea
       σ0 (VV −20.5 dB, VH −28 dB)
    3. single-pol inputs synthesize VH = VV − 7.5 dB (typical sea offset);
       expect reduced accuracy vs. true dual-pol
    4. the model's training normalization: (dB + 40) / 15, nodata → 0

Output: detections JSON (position px, score, vessel classification,
length in metres) plus an annotated PNG for instant visual feedback.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from cog import BasePredictor, Input, Path
from scipy.ndimage import binary_dilation, label as ndlabel

from unet import TimmUnet

SEA_PCTL = 30
SEA_VV_DB = -20.5
SEA_VH_DB = -28.0
VV_MINUS_VH_DB = 7.5   # typical sea-level offset used for single-pol inputs


def _to_db(band: np.ndarray) -> tuple[np.ndarray, str]:
    """Return (dB array, description). Detects whether the band is already
    in dB (calibrated σ0 is negative, small magnitude) or amplitude/DN."""
    band = band.astype("float32")
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.full_like(band, -100.0), "empty"
    p1, p99 = np.percentile(finite, [1, 99])
    if p99 <= 15.0 and p1 >= -60.0 and p1 < 0:
        return band, "sigma0 dB (used as-is before anchoring)"
    return 20.0 * np.log10(np.maximum(band, 1.0)), "amplitude/DN → 20·log10"


def _anchor(db: np.ndarray, sea_db: float) -> np.ndarray:
    valid = np.isfinite(db) & (db > -99)
    if valid.mean() < 0.02:
        return db
    return db - (float(np.percentile(db[valid], SEA_PCTL)) - sea_db)


LAND_BG_DB = -15.0     # local background brighter than this ⇒ land clutter
BG_INNER_PX = 8        # exclude the target itself from the background ring
BG_OUTER_PX = 40


def _bright_background(vv_db: np.ndarray, nodata: np.ndarray,
                       cy: int, cx: int) -> bool:
    """True when a detection sits on a BRIGHT local background — urban/land
    radar clutter rather than a ship on dark water. Measured as the median
    of an annulus around the target (excluding the target core and nodata),
    in anchored dB where calm sea ≈ -20.5."""
    h, w = vv_db.shape
    y0, y1 = max(0, cy - BG_OUTER_PX), min(h, cy + BG_OUTER_PX + 1)
    x0, x1 = max(0, cx - BG_OUTER_PX), min(w, cx + BG_OUTER_PX + 1)
    patch = vv_db[y0:y1, x0:x1]
    mask = ~nodata[y0:y1, x0:x1]
    iy0, iy1 = max(0, cy - BG_INNER_PX) - y0, min(h, cy + BG_INNER_PX + 1) - y0
    ix0, ix1 = max(0, cx - BG_INNER_PX) - x0, min(w, cx + BG_INNER_PX + 1) - x0
    mask[iy0:iy1, ix0:ix1] = False
    ring = patch[mask]
    if ring.size < 50:
        return False  # not enough context to judge — keep the detection
    return float(np.median(ring)) > LAND_BG_DB


def _normalize(db: np.ndarray) -> np.ndarray:
    db = db.copy()
    db[~np.isfinite(db)] = -100.0
    ignored = db <= -99
    out = (db + 40.0) / 15.0
    out[ignored] = 0.0
    return out


QUICKLOOK_SPAN_DB = 40.0  # assumed dynamic range of a display stretch


def _quicklook_to_db(gray: np.ndarray) -> np.ndarray:
    """8-bit quicklooks are display stretches — LINEAR IN dB, not amplitude.
    Running them through 20·log10 again double-compresses the contrast and
    kills detections (verified on real chips). Map [0,255] onto a typical
    ~40 dB display span instead; the sea-percentile anchor then places the
    absolute level."""
    return gray.astype("float32") / 255.0 * QUICKLOOK_SPAN_DB - 30.0


def _load_bands(path: str) -> tuple[np.ndarray, np.ndarray | None, str, bool]:
    """→ (vv_raw, vh_raw or None, source description, is_quicklook).

    Detection is by CONTENT, never by filename: files arrive through
    Replicate's Files API under extension-less download URLs, so an
    extension check silently routed float32 2-band TIFFs into the PIL
    quicklook branch (page 1 only, clipped to 8-bit, VH synthesized) —
    degrading every prediction while still 'working'. Sniff instead.
    """
    # 1) npz with our known keys?
    try:
        d = np.load(path)
        if hasattr(d, "files") and "vv_db" in d.files:
            return (d["vv_db"].astype("float32"), d["vh_db"].astype("float32"),
                    "npz (pre-computed dB)", False)
    except Exception:
        pass
    # 2) TIFF? tifffile reads by magic bytes, keeps dtype and all pages.
    # (Read attempt kept separate from shape handling: TiffFileError
    # subclasses ValueError, so a combined try-block can't distinguish
    # "not a TIFF, try PIL" from "TIFF with an unsupported layout".)
    import tifffile

    try:
        arr = np.asarray(tifffile.imread(path))
    except Exception:
        arr = None
    if arr is not None:
        quicklook = arr.dtype == np.uint8
        if arr.ndim == 2:
            return arr, None, f"single-band TIFF ({arr.dtype})", quicklook
        if arr.ndim == 3:
            # accept (C,H,W) or (H,W,C)
            if arr.shape[0] <= 4 and arr.shape[0] < arr.shape[-1]:
                bands = [arr[i] for i in range(arr.shape[0])]
            else:
                bands = [arr[..., i] for i in range(arr.shape[-1])]
            vh = bands[1] if len(bands) > 1 else None
            return (bands[0], vh, f"{len(bands)}-band TIFF ({arr.dtype})",
                    quicklook)
        raise ValueError(f"Unsupported TIFF shape {arr.shape}")
    # 3) PNG/JPEG quicklook via PIL
    from PIL import Image

    img = Image.open(path)
    if img.mode == "F":  # float TIFF that tifffile couldn't parse
        return np.asarray(img).astype("float32"), None, "float image (PIL)", False
    if img.mode == "LA":  # legacy 2-channel (VV in L, VH in A)
        arr = np.asarray(img).astype("float32")
        return arr[..., 0], arr[..., 1], "LA PNG", True
    gray = np.asarray(img.convert("L")).astype("float32")
    return gray, None, f"{img.format or 'image'} grayscale quicklook", True


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
        image: Path = Input(
            description="SAR chip: GeoTIFF (VV or VV+VH bands; amplitude, "
                        "intensity or σ0 dB — auto-detected), grayscale "
                        "PNG/JPEG quicklook, or legacy .npz. Best results: "
                        "float TIFF at ~10 m/px (Sentinel-1 IW GRD)."),
        pixel_spacing_m: float = Input(
            default=10.0, ge=1.0, le=100.0,
            description="Ground sampling distance of the input in metres/px "
                        "(10 for Sentinel-1 IW GRD). Length estimates scale "
                        "with this."),
        confidence: float = Input(
            default=100.0, ge=0, le=255,
            description="Detection threshold on the model's center head."),
        include_land_targets: bool = Input(
            default=False,
            description="Keep detections whose local background is bright "
                        "(land/urban clutter). Default off: a real ship at "
                        "sea sits on dark water (~-20 dB); radar-bright "
                        "cities trigger many false 'ships'. Ships moored "
                        "directly against piers are also suppressed when "
                        "off — turn on for harbour analysis."),
        vv_anchor_db: Optional[float] = Input(
            default=None,
            description="Advanced: the raw-dB value corresponding to open "
                        "water at FULL-SCENE scale (e.g. p30 over the whole "
                        "scene). Used only by the land-clutter test as an "
                        "absolute water reference — a chip that is mostly "
                        "city self-normalizes its rooftops to 'sea level', "
                        "which would otherwise blind the land filter. Model "
                        "detection itself always uses per-chip anchoring "
                        "(more robust: sea state varies several dB between "
                        "basins of one scene)."),
        vh_anchor_db: Optional[float] = Input(
            default=None,
            description="Reserved for symmetry with vv_anchor_db; currently "
                        "unused (the land test runs on VV)."),
    ) -> dict:
        vv_raw, vh_raw, source, quicklook = _load_bands(str(image))

        if quicklook:
            vv_db, vv_kind = _quicklook_to_db(vv_raw), \
                f"8-bit display stretch → linear dB over {QUICKLOOK_SPAN_DB:.0f} dB span"
        else:
            vv_db, vv_kind = _to_db(vv_raw)

        # Nodata (swath edges) must be masked BEFORE the sea anchor is
        # computed — otherwise a half-empty edge chip anchors its
        # percentile inside the nodata mass, mis-calibrating the whole
        # chip and provoking phantom detections along the swath boundary.
        nodata = (vv_raw == 0) if (vv_kind.startswith("amplitude") or quicklook) \
            else ~np.isfinite(vv_raw)
        vv_db[nodata] = -100.0
        # Model input is ALWAYS per-chip anchored — detection runs on local
        # contrast and this is the empirically proven path. The scene-level
        # anchor (if given) feeds ONLY the land-clutter test below: it maps
        # this chip into an absolute frame where open water ≈ -20.5 dB even
        # when the chip itself is wall-to-wall city (per-chip anchoring
        # would normalize rooftops to sea level and blind the land test).
        vv_land = vv_db.copy()
        if vv_anchor_db is not None:
            validl = vv_land > -99
            vv_land[validl] = vv_land[validl] - (vv_anchor_db - SEA_VV_DB)
            anchor_kind = "per-chip (model) + scene-level (land test)"
        else:
            anchor_kind = f"per-chip p{SEA_PCTL} anchor (model and land test)"
        vv_db = _anchor(vv_db, SEA_VV_DB)
        if vv_anchor_db is None:
            vv_land = vv_db

        if vh_raw is not None:
            vh_db = (_quicklook_to_db(vh_raw) if quicklook
                     else _to_db(vh_raw)[0])
            vh_db[nodata] = -100.0
            vh_db[vh_raw == 0] = -100.0
            vh_db = _anchor(vh_db, SEA_VH_DB)
            pol = "dual-pol (VV+VH)"
        else:
            vh_db = vv_db - VV_MINUS_VH_DB
            vh_db[nodata] = -100.0
            pol = "single-pol (VH synthesized from VV — reduced accuracy)"

        x = torch.from_numpy(
            np.stack([_normalize(vv_db), _normalize(vh_db)]))[None].float()
        with torch.no_grad():
            out = self.model(x)
        center = out["center_mask"][0, 0].numpy()
        vessel = torch.sigmoid(out["vessel_mask"][0, 0]).numpy() * 255
        length = out["length_mask"][0, 0].numpy()

        length_scale = pixel_spacing_m / 10.0
        labeled, n = ndlabel(binary_dilation(center > confidence))
        # Ships can't exist where the radar didn't image: reject any
        # detection whose neighbourhood is mostly nodata. This kills the
        # phantom targets the UNet emits along the sharp valid/nodata
        # boundary of swath-edge chips.
        h_img, w_img = vv_db.shape
        detections = []
        n_land = 0
        for i in range(1, n + 1):
            region = labeled == i
            ys, xs = np.nonzero(region)
            cy, cx = int(round(ys.mean())), int(round(xs.mean()))
            y0, y1 = max(0, cy - 3), min(h_img, cy + 4)
            x0, x1 = max(0, cx - 3), min(w_img, cx + 4)
            if nodata[y0:y1, x0:x1].mean() > 0.5:
                continue
            on_land = _bright_background(vv_land, nodata, cy, cx)
            if on_land and not include_land_targets:
                n_land += 1
                continue
            detections.append({
                "y": float(ys.mean()),
                "x": float(xs.mean()),
                "score": float(center[region].mean()),
                "is_vessel": bool((vessel[region] > 100).sum() > 8),
                "length_m": float(length[region].max() * length_scale),
                "on_land": on_land,
            })

        h, w = vv_db.shape
        return {
            "count": len(detections),
            "detections": detections,
            "annotated": self._annotate(vv_db, detections, pixel_spacing_m),
            "preprocessing": {
                "source": source, "vv_interpretation": vv_kind,
                "polarization": pol,
                "calibration": anchor_kind,
                "land_clutter_suppressed": n_land,
            },
            "height": h, "width": w,
        }

    def _annotate(self, vv_db: np.ndarray, detections: list[dict],
                  pixel_spacing_m: float) -> Path:
        """Detections drawn over a display stretch of VV — instant visual
        feedback on the Replicate playground."""
        from PIL import Image, ImageDraw

        valid = vv_db > -99
        lo, hi = (np.percentile(vv_db[valid], [2, 99.5]) if valid.any()
                  else (-30.0, 0.0))
        gray = (np.clip((vv_db - lo) / max(hi - lo, 1e-6), 0, 1) * 255
                ).astype(np.uint8)
        img = Image.fromarray(gray).convert("RGB")
        draw = ImageDraw.Draw(img)
        for d in detections:
            color = (255, 60, 60) if d["is_vessel"] else (255, 200, 0)
            r = max(8.0, d["length_m"] / pixel_spacing_m / 2 + 4)
            x, y = d["x"], d["y"]
            draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=2)
            draw.text((x + r + 2, y - 6), f"{d['length_m']:.0f} m", fill=color)
        out = "/tmp/annotated.png"
        img.save(out)
        return Path(out)
