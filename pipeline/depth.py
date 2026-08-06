"""
Depth estimation (Depth Anything V2) and the relative-depth -> usable-depth
conversion.

The important subtlety here: Depth Anything V2's *relative* checkpoints do
not predict depth. They predict inverse depth (disparity) in arbitrary
units — bigger means nearer, and the scale is per-image, not metric. Feeding
that straight into a pinhole unprojection produces a scene that is turned
inside out (far things end up nearest the camera). `disparity_to_depth`
below is the conversion that has to happen first, and it is the single most
common place this kind of pipeline goes wrong.

Two model families are supported:
  - relative (`depth-anything/Depth-Anything-V2-Small-hf` and friends)
    -> disparity, needs the conversion, scale is arbitrary
  - metric (`depth-anything/Depth-Anything-V2-metric-indoor-small-hf`)
    -> depth in metres directly, no conversion, scale is meaningful-ish

The metric indoor checkpoint is worth preferring for room scenes because it
removes one of the two ambiguities (scale), leaving only focal length. It is
the same size as the relative Small model, so it costs nothing extra to run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

ModelKind = Literal["relative", "metric"]

# Checkpoint -> whether it emits metric depth or relative disparity.
KNOWN_CHECKPOINTS: dict[str, ModelKind] = {
    "depth-anything/Depth-Anything-V2-Small-hf": "relative",
    "depth-anything/Depth-Anything-V2-Base-hf": "relative",
    "depth-anything/Depth-Anything-V2-Large-hf": "relative",
    "depth-anything/Depth-Anything-V2-metric-indoor-small-hf": "metric",
    "depth-anything/Depth-Anything-V2-metric-indoor-base-hf": "metric",
    "depth-anything/Depth-Anything-V2-metric-indoor-large-hf": "metric",
}

DEFAULT_CHECKPOINT = "depth-anything/Depth-Anything-V2-metric-indoor-small-hf"


@dataclass
class DepthResult:
    """Depth for one image, in a form the rest of the pipeline can consume."""

    depth: np.ndarray  # (H, W) float32, distance along the optical axis
    raw: np.ndarray  # (H, W) float32, whatever the model actually emitted
    kind: ModelKind
    checkpoint: str

    @property
    def is_metric(self) -> bool:
        return self.kind == "metric"

    @property
    def shape(self) -> tuple[int, int]:
        return self.depth.shape[:2]

    def normalized_for_preview(self) -> np.ndarray:
        """(H, W) uint8 depth visualisation, near = bright."""
        d = self.depth
        finite = np.isfinite(d)
        if not finite.any():
            return np.zeros(d.shape, dtype=np.uint8)
        lo, hi = np.percentile(d[finite], [2, 98])
        if hi - lo < 1e-9:
            return np.zeros(d.shape, dtype=np.uint8)
        norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
        return ((1.0 - norm) * 255).astype(np.uint8)


def disparity_to_depth(
    disparity: np.ndarray,
    near: float = 0.6,
    far: float = 12.0,
    percentile_clip: tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    """Relative inverse depth -> depth, mapped into a plausible [near, far].

    Depth Anything's relative output is affine-invariant disparity: it is
    proportional to 1/depth up to an unknown scale and shift. We cannot
    recover the true scale from one image, so we choose one — we assume the
    nearest surface in the photo is `near` metres away and the farthest is
    `far` metres, and solve for the affine map that makes that true.

    Concretely, after normalising disparity to [0, 1]:

        1/depth = t * (1/near) + (1 - t) * (1/far)

    i.e. we interpolate in *inverse* depth, not in depth. That matters: a
    linear-in-depth mapping would squash all the near geometry together and
    stretch the background, visibly bending flat walls.

    The percentile clip keeps a handful of outlier pixels (specular
    highlights, sky through a window) from dragging the whole normalisation.
    """
    d = np.asarray(disparity, dtype=np.float64)
    if near <= 0 or far <= near:
        raise ValueError(f"need 0 < near < far, got near={near} far={far}")

    finite = np.isfinite(d)
    if not finite.any():
        raise ValueError("disparity map has no finite values")

    lo, hi = np.percentile(d[finite], percentile_clip)
    if hi - lo < 1e-12:
        # Degenerate (flat) prediction — return a constant plane rather
        # than dividing by zero.
        return np.full(d.shape, (near + far) / 2.0, dtype=np.float32)

    t = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    inv_depth = t * (1.0 / near) + (1.0 - t) * (1.0 / far)
    depth = 1.0 / inv_depth

    depth[~finite] = far
    return depth.astype(np.float32)


class DepthAnythingV2:
    """Thin wrapper over the HF `transformers` Depth Anything V2 pipeline.

    Kept deliberately thin — this is a pretrained building block, the code
    that matters is what we do with its output. Loading is lazy so importing
    this module stays cheap on a machine with no GPU.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str | None = None,
        kind: ModelKind | None = None,
    ):
        self.checkpoint = checkpoint
        self.kind: ModelKind = kind or KNOWN_CHECKPOINTS.get(checkpoint, "relative")
        self._device = device
        self._model = None
        self._processor = None

    @property
    def device(self) -> str:
        if self._device is None:
            try:
                import torch

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:  # pragma: no cover
                self._device = "cpu"
        return self._device

    def load(self) -> "DepthAnythingV2":
        if self._model is not None:
            return self
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._processor = AutoImageProcessor.from_pretrained(self.checkpoint)
        model = AutoModelForDepthEstimation.from_pretrained(self.checkpoint)
        # fp16 on GPU roughly halves both VRAM and latency, and depth is
        # nowhere near precision-critical enough to notice the difference.
        if self.device == "cuda":
            model = model.half()
        self._model = model.to(self.device).eval()
        return self

    def predict(
        self,
        image: Image.Image,
        near: float = 0.6,
        far: float = 12.0,
        max_side: int | None = 700,
    ) -> DepthResult:
        """Run depth on a PIL image, returned at the image's own resolution.

        `max_side` caps the resolution fed to the model (the output is
        upsampled back to full size). The 4GB-VRAM laptop case needs this;
        on a T4 it can be raised or set to None.
        """
        import torch

        self.load()
        rgb = image.convert("RGB")
        full_w, full_h = rgb.size

        infer_img = rgb
        if max_side and max(full_w, full_h) > max_side:
            scale = max_side / max(full_w, full_h)
            infer_img = rgb.resize(
                (max(1, round(full_w * scale)), max(1, round(full_h * scale))),
                Image.BICUBIC,
            )

        inputs = self._processor(images=infer_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.device == "cuda":
            inputs = {
                k: (v.half() if v.dtype == torch.float32 else v)
                for k, v in inputs.items()
            }

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Upsample the prediction back to the original image resolution so
        # depth and pixels index the same grid everywhere downstream.
        prediction = torch.nn.functional.interpolate(
            outputs.predicted_depth.unsqueeze(1).float(),
            size=(full_h, full_w),
            mode="bicubic",
            align_corners=False,
        )
        raw = prediction.squeeze().cpu().numpy().astype(np.float32)

        return self._to_result(raw, near=near, far=far)

    def _to_result(self, raw: np.ndarray, near: float, far: float) -> DepthResult:
        if self.kind == "metric":
            depth = np.asarray(raw, dtype=np.float32).copy()
            # Metric checkpoints occasionally emit a few non-positive
            # pixels at borders; clamp rather than let them fold the
            # geometry through the camera centre.
            depth[~np.isfinite(depth)] = far
            depth = np.clip(depth, 1e-3, None)
        else:
            depth = disparity_to_depth(raw, near=near, far=far)
        return DepthResult(
            depth=depth, raw=np.asarray(raw, dtype=np.float32),
            kind=self.kind, checkpoint=self.checkpoint,
        )

    def unload(self) -> None:
        """Free VRAM. The Colab notebook holds three models at once, so
        being able to drop one matters on a 15GB T4."""
        self._model = None
        self._processor = None
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass


def smooth_depth_edge_preserving(
    depth: np.ndarray, diameter: int = 7, sigma_depth: float = 0.05
) -> np.ndarray:
    """Bilateral-filter the depth map, keeping object boundaries sharp.

    Depth Anything output is smooth already, but the *upsample* back to full
    resolution reintroduces ringing along high-contrast edges, which shows
    up in the Tier 1 mesh as a rippled halo around every object. A bilateral
    filter removes the ripple without rounding off the depth discontinuities
    that the mesher relies on for its edge-aware culling.

    No-op (with a warning return) if OpenCV isn't installed.
    """
    try:
        import cv2
    except ImportError:
        return depth

    d = depth.astype(np.float32)
    scale = float(np.median(d[np.isfinite(d)])) or 1.0
    return cv2.bilateralFilter(
        d / scale, d=diameter, sigmaColor=sigma_depth, sigmaSpace=diameter
    ) * scale
