"""
Tier 1: turn one depth map into one continuous, textured scene mesh.

This is the "always works" path — no segmentation, no generative model, no
placement solver. Every pixel becomes a vertex, adjacent pixels become
triangles, the source image becomes the texture. What you get back is
exactly the visible surface of the photo, pushed out into 3D.

The only genuinely interesting part is deciding which triangles *not* to
emit. Naively triangulating a depth map welds every foreground object to
whatever is behind it, producing the rubber-sheet "skin" that makes these
reconstructions look melted. Two independent tests kill those triangles:

  1. Depth-jump test  — the triangle straddles a depth discontinuity.
  2. Grazing-angle test — the triangle is nearly edge-on to the camera,
     i.e. it spans a lot of depth over very few pixels.

They catch overlapping but not identical failure cases: (1) misses gentle
ramps that are still artefacts (a wall seen at a glancing angle behind a
chair leg), (2) misses genuine near-perpendicular jumps at low resolution.
Both thresholds are exposed because the right value is scene-dependent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from PIL import Image

from .camera import PinholeCamera


@dataclass
class MeshingParams:
    """Tunables for the Tier 1 mesher. Defaults are for indoor room photos."""

    # Longest side of the vertex grid. 480 gives ~170k vertices / ~340k
    # triangles, which three.js handles comfortably and a .glb of a few MB.
    max_grid_side: int = 480

    # Depth-jump cull. A triangle is dropped when the relative depth spread
    # across it, (max - min) / min, exceeds this. Relative rather than
    # absolute so the same threshold works near the camera and far from it.
    max_relative_depth_jump: float = 0.06

    # Grazing-angle cull. Drop triangles whose surface normal is more than
    # this many degrees away from facing the camera. 87 keeps legitimately
    # oblique surfaces (floors, side walls) while killing the stretched
    # connective tissue between depth layers.
    max_grazing_angle_deg: float = 87.0

    # Drop pixels closer/farther than these (metres, or arbitrary units for
    # relative depth). Mostly catches sky-through-window blowouts.
    min_depth: float = 1e-3
    max_depth: float = 1e4

    # Bilateral-filter the depth before meshing. Removes upsampling ripple.
    smooth_depth: bool = True


@dataclass
class Tier1Result:
    mesh: object  # trimesh.Trimesh
    camera: PinholeCamera  # intrinsics at the *grid* resolution
    stats: dict = field(default_factory=dict)


def _resize_to_grid(
    image: Image.Image, depth: np.ndarray, max_side: int
) -> tuple[Image.Image, np.ndarray]:
    """Bring image and depth onto one common, size-capped grid."""
    h, w = depth.shape[:2]
    if max_side is None or max(h, w) <= max_side:
        target_w, target_h = w, h
    else:
        scale = max_side / max(h, w)
        target_w = max(2, round(w * scale))
        target_h = max(2, round(h * scale))

    img = image.convert("RGB")
    if img.size != (target_w, target_h):
        img = img.resize((target_w, target_h), Image.LANCZOS)

    if (target_h, target_w) != (h, w):
        # Depth is resampled with PIL rather than a plain stride so we
        # don't alias away thin structures; BILINEAR (not LANCZOS) because
        # we do not want the overshoot ringing that a windowed-sinc kernel
        # produces at depth edges.
        depth = np.asarray(
            Image.fromarray(depth.astype(np.float32), mode="F").resize(
                (target_w, target_h), Image.BILINEAR
            ),
            dtype=np.float32,
        )

    return img, depth


def build_face_grid(h: int, w: int) -> np.ndarray:
    """Every quad of the pixel grid, split into two triangles.

    Returns (2 * (h-1) * (w-1), 3) vertex indices into a row-major (h, w)
    grid, wound counter-clockwise as seen from the camera so that glTF
    front-face culling and the default three.js material behave.
    """
    idx = np.arange(h * w, dtype=np.int64).reshape(h, w)
    a = idx[:-1, :-1].ravel()  # top-left
    b = idx[:-1, 1:].ravel()  # top-right
    c = idx[1:, :-1].ravel()  # bottom-left
    d = idx[1:, 1:].ravel()  # bottom-right

    lower = np.stack([a, c, b], axis=1)
    upper = np.stack([b, c, d], axis=1)
    return np.concatenate([lower, upper], axis=0)


def depth_jump_mask(depth_flat: np.ndarray, faces: np.ndarray, threshold: float) -> np.ndarray:
    """True for faces to KEEP under the relative-depth-spread test."""
    tri_depth = depth_flat[faces]  # (F, 3)
    dmin = tri_depth.min(axis=1)
    dmax = tri_depth.max(axis=1)
    spread = (dmax - dmin) / np.maximum(dmin, 1e-6)
    return spread <= threshold


def grazing_mask(
    points_flat: np.ndarray, faces: np.ndarray, max_angle_deg: float
) -> np.ndarray:
    """True for faces to KEEP under the grazing-angle test.

    The test is on the angle between the triangle's normal and the ray from
    the camera to the triangle's centroid. A surface facing the camera has
    0 degrees; a surface seen exactly edge-on has 90.
    """
    tri = points_flat[faces]  # (F, 3, 3)
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    n_len = np.linalg.norm(normals, axis=1)

    centroids = tri.mean(axis=1)
    c_len = np.linalg.norm(centroids, axis=1)

    # Degenerate (zero-area) triangles have no meaningful normal; drop them.
    valid = (n_len > 1e-12) & (c_len > 1e-12)

    cos_angle = np.zeros(len(faces), dtype=np.float64)
    np.divide(
        np.abs(np.einsum("ij,ij->i", normals, centroids)),
        np.maximum(n_len * c_len, 1e-12),
        out=cos_angle,
        where=valid,
    )
    return valid & (cos_angle >= math.cos(math.radians(max_angle_deg)))


def prune_unused_vertices(
    vertices: np.ndarray, uv: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop vertices no surviving face references, and reindex."""
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return vertices[used], uv[used], remap[faces]


def depth_to_mesh(
    image: Image.Image,
    depth: np.ndarray,
    camera: PinholeCamera,
    params: MeshingParams | None = None,
) -> Tier1Result:
    """Depth map + source image -> one textured trimesh scene mesh.

    `camera` must describe the *full-resolution* image; it is rescaled
    internally to whatever grid resolution the mesh ends up at.
    """
    import trimesh

    params = params or MeshingParams()

    if depth.shape[:2] != (camera.height, camera.width):
        raise ValueError(
            f"depth {depth.shape[:2]} does not match camera "
            f"{(camera.height, camera.width)}"
        )

    if params.smooth_depth:
        from .depth import smooth_depth_edge_preserving

        depth = smooth_depth_edge_preserving(depth)

    img, grid_depth = _resize_to_grid(image, depth, params.max_grid_side)
    grid_w, grid_h = img.size
    grid_cam = camera.scaled(grid_w, grid_h)

    # Invalid depth is pushed to NaN so it fails every downstream test
    # rather than quietly producing a point at the origin.
    d = grid_depth.astype(np.float64)
    invalid = ~np.isfinite(d) | (d < params.min_depth) | (d > params.max_depth)
    d = np.where(invalid, np.nan, d)

    points = grid_cam.unproject_depth_map(d)  # (H, W, 3)
    points_flat = points.reshape(-1, 3)
    depth_flat = d.reshape(-1)

    faces = build_face_grid(grid_h, grid_w)
    n_total = len(faces)

    finite_vertex = np.isfinite(depth_flat)
    keep = finite_vertex[faces].all(axis=1)
    n_after_finite = int(keep.sum())

    keep &= depth_jump_mask(depth_flat, faces, params.max_relative_depth_jump)
    n_after_jump = int(keep.sum())

    keep &= grazing_mask(points_flat, faces, params.max_grazing_angle_deg)
    n_after_grazing = int(keep.sum())

    faces = faces[keep]
    if len(faces) == 0:
        raise RuntimeError(
            "every triangle was culled — depth map is probably degenerate, "
            "or the thresholds are far too tight"
        )

    # UVs address the source image directly, so the texture is the photo
    # itself with no resampling: u = x / (W-1), v flipped for glTF's
    # bottom-left texture origin.
    vs, us = np.mgrid[0:grid_h, 0:grid_w]
    uv = np.stack(
        [us.ravel() / (grid_w - 1), 1.0 - vs.ravel() / (grid_h - 1)], axis=1
    ).astype(np.float32)

    # NaN vertices survive pruning only if referenced, which they can't be,
    # but they still need finite coordinates for trimesh to accept them.
    vertices = np.nan_to_num(points_flat, nan=0.0)
    vertices, uv, faces = prune_unused_vertices(vertices, uv, faces)

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=img,
        metallicFactor=0.0,
        roughnessFactor=1.0,
        # Depth-map meshes have plenty of holes where triangles were culled;
        # rendering them double-sided stops those holes from looking like
        # missing geometry when the camera orbits behind them.
        doubleSided=True,
    )
    mesh = trimesh.Trimesh(
        vertices=vertices.astype(np.float32),
        faces=faces,
        visual=trimesh.visual.TextureVisuals(uv=uv, material=material),
        process=False,
    )

    stats = {
        "grid": [grid_w, grid_h],
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "faces_before_culling": int(n_total),
        "culled_invalid_depth": int(n_total - n_after_finite),
        "culled_depth_jump": int(n_after_finite - n_after_jump),
        "culled_grazing": int(n_after_jump - n_after_grazing),
        "kept_fraction": round(len(faces) / n_total, 4),
        "depth_range": [
            float(np.nanmin(d)),
            float(np.nanmax(d)),
        ],
        "hfov_deg": round(camera.hfov_deg, 2),
    }
    return Tier1Result(mesh=mesh, camera=grid_cam, stats=stats)


def backproject_mask(
    mask: np.ndarray,
    depth: np.ndarray,
    camera: PinholeCamera,
    max_points: int | None = 20000,
    depth_percentile_trim: tuple[float, float] | None = (2.0, 98.0),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Pixels under a boolean mask -> (N, 3) point cloud.

    This is the bridge between 2D segmentation and 3D: it answers "where in
    space does this object actually sit, according to the photo". The room
    shell fitter uses it on the background mask; the assembly solver uses it
    per object as the target cloud to align a generated mesh against.

    The percentile trim matters more than it looks. Segmentation masks leak
    a few pixels onto whatever is behind the object, and because those
    pixels are much farther away they drag a fitted scale or centroid
    disproportionately. Trimming the depth tails removes them cheaply
    without needing a full outlier-rejection pass.
    """
    if mask.shape[:2] != depth.shape[:2]:
        raise ValueError(f"mask {mask.shape[:2]} != depth {depth.shape[:2]}")
    if depth.shape[:2] != (camera.height, camera.width):
        raise ValueError("camera resolution does not match depth map")

    vs, us = np.nonzero(mask.astype(bool))
    if len(vs) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    d = depth[vs, us].astype(np.float64)
    good = np.isfinite(d) & (d > 0)
    vs, us, d = vs[good], us[good], d[good]
    if len(d) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    if depth_percentile_trim is not None and len(d) > 50:
        lo, hi = np.percentile(d, depth_percentile_trim)
        inliers = (d >= lo) & (d <= hi)
        vs, us, d = vs[inliers], us[inliers], d[inliers]

    if max_points is not None and len(d) > max_points:
        rng = rng or np.random.default_rng(0)
        pick = rng.choice(len(d), size=max_points, replace=False)
        vs, us, d = vs[pick], us[pick], d[pick]

    return camera.unproject(us, vs, d)
