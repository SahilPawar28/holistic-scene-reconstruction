"""
Fill in the background where an object was, instead of cutting a hole.

Earlier, the background mesh for a placed object was built by *excluding*
that object's pixels entirely — cutting the triangles out and letting the
generated 3D object fill the gap from the original camera position. That
works from exactly the photo's own viewpoint, because the object really was
there covering that background. Orbit even slightly and it stops working:
the excluded region has no geometry at all, so the wall or table behind the
object is a black gap rather than a wall.

This module replaces "cut a hole" with "fill it plausibly first, then mesh
normally". Both the colour and the depth under an object's mask are
inpainted before the Tier 1 mesher ever sees them, so what gets built is a
continuous surface: the real photo everywhere the object wasn't, a
plausible low-detail continuation of the surrounding wall/table/floor where
it was. No exclusion step is needed afterwards — the edge-aware culling in
`meshing.depth_to_mesh` already drops stretched triangles at genuine depth
jumps, and it does not fire here because the inpainted region is smooth by
construction.

Two different inpainting methods for two different signals:

  - **Colour** uses OpenCV's Telea algorithm, which propagates texture
    inward from the region boundary — reasonable for filling in a patch of
    wall or tabletop, since those tend to be locally uniform or gently
    gradiented.
  - **Depth** uses nearest-valid-pixel extrapolation (a Euclidean distance
    transform), not Telea. Depth is not a texture; smoothly *interpolating*
    depth across a hole that spans a real object would invent a slope where
    there should be a flat wall. Nearest-neighbour is blockier but does not
    fabricate structure that was not observed at the boundary.

Nothing here invents what was actually behind the object — it cannot, the
photograph does not contain that information. It only makes the *absence*
look like a wall instead of a hole, which is the honest amount of
information to present.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class InpaintParams:
    # Grown outward from the object's exact silhouette before inpainting.
    # Segmentation masks sit a pixel or two inside the true edge, and the
    # generated object's placed footprint rarely matches the mask exactly
    # either — filling a slightly larger region than filling exactly the
    # mask is what keeps a visible seam from showing at the object's edge.
    grow_px: int = 4
    # Telea inpainting radius, in pixels. Larger softens the fill more but
    # costs more time; small holes (most objects) do not need much.
    color_radius: int = 7


def _grow_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    try:
        import cv2

        k = np.ones((px * 2 + 1, px * 2 + 1), np.uint8)
        return cv2.dilate(mask.astype(np.uint8), k, iterations=1).astype(bool)
    except ImportError:
        return mask


def inpaint_depth(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill masked depth pixels with their nearest unmasked neighbour.

    Uses `scipy.ndimage.distance_transform_edt`'s index-return mode, which
    for every masked pixel finds the nearest unmasked one directly — the
    standard trick for depth-hole filling, and it needs no dependency this
    project does not already have.
    """
    from scipy.ndimage import distance_transform_edt

    hole = np.asarray(mask, dtype=bool)
    if not hole.any():
        return depth.copy()
    if hole.all():
        # Nothing valid to extrapolate from; leave it untouched rather than
        # inventing a depth from nothing.
        return depth.copy()

    _, (iy, ix) = distance_transform_edt(hole, return_indices=True)
    filled = depth.copy()
    filled[hole] = depth[iy[hole], ix[hole]]
    return filled


def inpaint_color(image: Image.Image, mask: np.ndarray, radius: int = 7) -> Image.Image:
    """Fill masked RGB pixels using OpenCV's Telea texture propagation."""
    try:
        import cv2
    except ImportError:
        return image

    rgb = np.asarray(image.convert("RGB"))
    hole = np.asarray(mask, dtype=np.uint8) * 255
    if not hole.any():
        return image
    filled = cv2.inpaint(rgb, hole, radius, cv2.INPAINT_TELEA)
    return Image.fromarray(filled)


def inpaint_background(
    image: Image.Image,
    depth: np.ndarray,
    mask: np.ndarray,
    params: InpaintParams | None = None,
) -> tuple[Image.Image, np.ndarray]:
    """Fill an object's footprint in both colour and depth.

    Returns (inpainted_image, inpainted_depth), the same shapes as the
    inputs. Feed these straight to `meshing.depth_to_mesh` with no
    `exclude_mask` — the region is now a smooth, texturable surface rather
    than something that needs to be cut away.
    """
    params = params or InpaintParams()
    grown = _grow_mask(np.asarray(mask, dtype=bool), params.grow_px)

    new_depth = inpaint_depth(np.asarray(depth, dtype=np.float32), grown)
    new_image = inpaint_color(image, grown, radius=params.color_radius)
    return new_image, new_depth
