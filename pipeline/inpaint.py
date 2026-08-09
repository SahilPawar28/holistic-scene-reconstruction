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
  - **Depth** solves a harmonic (Laplace) fill with the surrounding depth as
    a boundary condition, not Telea. Depth is not a texture, and it must be
    continuous with its border and free of interior steps or the mesher's
    own depth-jump cull will punch the filled region straight back out. See
    `inpaint_depth` — the nearest-neighbour approach this originally used
    was not good enough, and its failure mode was visible as star-shaped
    black patches.

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


def _nearest_fill(depth: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Fill masked pixels with their nearest unmasked neighbour's value."""
    from scipy.ndimage import distance_transform_edt

    _, (iy, ix) = distance_transform_edt(hole, return_indices=True)
    filled = depth.copy()
    filled[hole] = depth[iy[hole], ix[hole]]
    return filled


def inpaint_depth(
    depth: np.ndarray,
    mask: np.ndarray,
    smooth: bool = True,
    coarse_side: int = 128,
    iterations: int = 600,
) -> np.ndarray:
    """Fill masked depth with a smooth surface continuous with its border.

    Nearest-neighbour filling alone is NOT sufficient here, and the way it
    fails is worth being precise about because it caused visible black
    patches. Every filled pixel takes the value of the closest border pixel,
    so the filled region ends up partitioned into cells, each fed by a
    different part of the border. Where two cells meet — along the mask's
    medial axis — the two border depths collide and leave a hard step. The
    Tier 1 mesher's depth-jump cull then correctly removes the triangles
    straddling that step, and the "filled" region comes out with a
    branching, star-shaped hole through the middle of it. Measured on the
    test room: relative steps up to 0.386 inside the filled area, against a
    cull threshold of 0.06.

    The fix is to solve for a *harmonic* fill instead — the discrete Laplace
    equation with the surrounding depth as a boundary condition, which is
    the smoothest surface that meets the border continuously and has no
    interior discontinuity to cull. Nearest-neighbour is kept as the initial
    guess since it converges far faster from there.

    Solved at reduced resolution and upsampled. Jacobi relaxation needs
    iterations proportional to the square of the region's radius, which is
    prohibitive at full resolution for a large object — and pointless, since
    there is no high-frequency detail to recover in a region the camera
    never saw. A smooth low-resolution solution is the honest answer.
    """
    hole = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth, dtype=np.float32)
    if not hole.any() or hole.all():
        return depth.copy()

    filled = _nearest_fill(depth, hole)
    if not smooth:
        return filled

    h, w = depth.shape[:2]
    scale = min(1.0, coarse_side / max(h, w))
    ch, cw = max(8, int(round(h * scale))), max(8, int(round(w * scale)))

    coarse = np.asarray(
        Image.fromarray(filled, mode="F").resize((cw, ch), Image.BILINEAR),
        dtype=np.float32,
    )
    coarse_hole = np.asarray(
        Image.fromarray((hole * 255).astype(np.uint8)).resize((cw, ch), Image.NEAREST)
    ) > 127
    if not coarse_hole.any():
        return filled

    # Jacobi relaxation of the Laplace equation: each interior pixel becomes
    # the average of its four neighbours, with known pixels held fixed.
    work = coarse.copy()
    for _ in range(iterations):
        padded = np.pad(work, 1, mode="edge")
        neighbour_mean = 0.25 * (
            padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
        )
        work[coarse_hole] = neighbour_mean[coarse_hole]

    smoothed = np.asarray(
        Image.fromarray(work, mode="F").resize((w, h), Image.BILINEAR),
        dtype=np.float32,
    )
    out = depth.copy()
    out[hole] = smoothed[hole]
    return out


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
