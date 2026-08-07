"""
Crop enhancement: give the generator the best possible look at each object.

TripoSR reconstructs a clean, well-lit, reasonably large object well. What
it actually receives from a room photo is often a 90x120 pixel patch of a
dim shelf, upscaled to 512 and therefore soft, low-contrast and half in
shadow. The blobby meshes that come back are not really a model failure —
they are the honest output for that input.

Nothing here invents detail. Upscaling a 90px crop to 512 cannot create
information that was never captured, and any module claiming otherwise is
lying. What these steps do is make the information that *is* present easier
for the model to use:

  - **CLAHE** pulls local contrast out of shadowed regions. A dark sofa
    against a dark wall carries plenty of shape cues; they are just
    compressed into a few levels of the histogram.
  - **Unsharp masking** restores the acutance that any upscale removes,
    which matters because the generator keys off edges.
  - **Honest reporting** of the native crop size, so a genuinely
    unrecoverable object can be identified as such rather than silently
    reconstructed into a blob.

That last one is the important part. `assess_crop` exists so the pipeline
can say "this object was only 70 pixels across, do not trust the mesh"
instead of pretending every input is equal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class EnhanceParams:
    """Tunables for crop enhancement. Conservative by default."""

    # Local contrast. 2.0 is a moderate lift; higher starts amplifying
    # sensor noise in dark regions, which the generator reads as texture.
    clahe_clip_limit: float = 2.0
    clahe_grid: int = 8
    use_clahe: bool = True

    # Unsharp mask. Applied *after* the resize, since the resize is what
    # costs the acutance.
    unsharp_amount: float = 0.6
    unsharp_radius: float = 1.4
    use_unsharp: bool = True

    # Below this many pixels on the crop's longest side, the object simply
    # was not captured in enough detail for a meaningful reconstruction.
    # Not enforced here — reported, so the caller can decide.
    min_native_px: int = 96
    # Upscaling further than this is pure interpolation; flagged as such.
    max_useful_upscale: float = 4.0


@dataclass
class CropQuality:
    """How much real information this crop actually carries."""

    native_px: int          # longest side in the source image
    output_px: int          # longest side after resizing for the generator
    upscale: float          # output / native
    sharpness: float        # variance of Laplacian, higher = crisper
    mean_luma: float        # 0-255; very low means a shadowed object
    contrast: float         # std of luma

    @property
    def too_small(self) -> bool:
        return self.native_px < 96

    @property
    def over_upscaled(self) -> bool:
        return self.upscale > 4.0

    def summary(self) -> dict:
        return {
            "native_px": int(self.native_px),
            "upscale": round(float(self.upscale), 2),
            "sharpness": round(float(self.sharpness), 1),
            "mean_luma": round(float(self.mean_luma), 1),
            "contrast": round(float(self.contrast), 1),
            "too_small": bool(self.too_small),
            "over_upscaled": bool(self.over_upscaled),
        }

    def warnings(self) -> list[str]:
        out = []
        if self.too_small:
            out.append(
                f"only {self.native_px}px across natively — too little detail "
                f"for a reliable reconstruction"
            )
        if self.over_upscaled:
            out.append(f"upscaled {self.upscale:.1f}x — mostly interpolation")
        if self.mean_luma < 45:
            out.append(f"very dark (mean luma {self.mean_luma:.0f})")
        if self.contrast < 18:
            out.append(f"very low contrast (std {self.contrast:.0f})")
        return out


def measure_quality(native_px: int, output_px: int, image: Image.Image) -> CropQuality:
    """Quality metrics for a prepared crop."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    luma = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    sharpness = 0.0
    try:
        import cv2

        sharpness = float(cv2.Laplacian(luma.astype(np.float32), cv2.CV_32F).var())
    except ImportError:
        gy, gx = np.gradient(luma)
        sharpness = float((gx**2 + gy**2).mean())

    return CropQuality(
        native_px=int(native_px),
        output_px=int(output_px),
        upscale=float(output_px) / max(float(native_px), 1.0),
        sharpness=sharpness,
        mean_luma=float(luma.mean()),
        contrast=float(luma.std()),
    )


def apply_clahe(image: Image.Image, params: EnhanceParams) -> Image.Image:
    """Contrast-limited adaptive histogram equalisation on luminance only.

    Operating on L in LAB rather than on RGB is what keeps colours intact —
    equalising the channels independently shifts hue, and a sofa that comes
    back the wrong colour is worse than one that comes back dark.
    """
    try:
        import cv2
    except ImportError:
        return image

    rgb = np.asarray(image.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(
        clipLimit=params.clahe_clip_limit,
        tileGridSize=(params.clahe_grid, params.clahe_grid),
    )
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def apply_unsharp(image: Image.Image, params: EnhanceParams) -> Image.Image:
    """Unsharp mask: original + amount * (original - blurred)."""
    try:
        import cv2
    except ImportError:
        return image

    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    blurred = cv2.GaussianBlur(rgb, (0, 0), params.unsharp_radius)
    sharpened = rgb + params.unsharp_amount * (rgb - blurred)
    return Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8))


def enhance(
    image: Image.Image,
    params: EnhanceParams | None = None,
    protect_mask: np.ndarray | None = None,
) -> Image.Image:
    """Enhance a prepared object crop for the generator.

    `protect_mask` marks the flat background region a crop was composited
    onto. CLAHE run over that flat grey would stretch its noise into visible
    banding, and the generator reads banding as geometry — so the background
    is restored afterwards.
    """
    params = params or EnhanceParams()
    out = image.convert("RGB")
    before = np.asarray(out).copy()

    if params.use_clahe:
        out = apply_clahe(out, params)
    if params.use_unsharp:
        out = apply_unsharp(out, params)

    if protect_mask is not None:
        arr = np.asarray(out).copy()
        keep = ~np.asarray(protect_mask, dtype=bool)
        if keep.shape == arr.shape[:2]:
            arr[keep] = before[keep]
            out = Image.fromarray(arr)

    return out
