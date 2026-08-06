"""
An analytically ray-traced toy room, used as ground truth for testing the
geometry code without a GPU or any pretrained model.

Every neural component in this pipeline is slow, non-deterministic and only
available on Colab, which makes the geometry underneath them miserable to
debug: if an object ends up floating a metre above the floor, is that the
depth model, the segmentation, or the placement solver?

This module removes that ambiguity. It builds a room out of exact planes and
boxes, renders a perfect depth map, a colour image and per-object masks by
closed-form ray intersection, and hands back the ground-truth plane
equations and object transforms. Anything the pipeline computes from that
input can be checked against a number that is known to be right.

Camera convention matches camera.py exactly: +X right, +Y up, -Z forward,
camera at the origin looking down -Z.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .camera import PinholeCamera


@dataclass
class Plane:
    """Half-space boundary: points p with dot(normal, p) + d == 0."""

    normal: np.ndarray
    d: float
    color: tuple[int, int, int]
    name: str
    # Optional axis-aligned extent used to bound an otherwise infinite
    # plane, as (min_xyz, max_xyz) with None meaning unbounded on that axis.
    bounds: tuple[np.ndarray, np.ndarray] | None = None

    def intersect(self, origin: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        """Ray-plane distance t per ray; inf where there is no valid hit."""
        n = self.normal
        denom = dirs @ n
        t = np.full(dirs.shape[:-1], np.inf)
        hit = np.abs(denom) > 1e-9
        t[hit] = -(origin @ n + self.d) / denom[hit]
        t[t <= 1e-6] = np.inf

        if self.bounds is not None:
            # inf * 0 is nan, so evaluate hit points only where t is finite.
            finite_t = np.where(np.isfinite(t), t, 0.0)
            pts = origin + dirs * finite_t[..., None]
            lo, hi = self.bounds
            inside = np.ones(t.shape, dtype=bool)
            for axis in range(3):
                if np.isfinite(lo[axis]):
                    inside &= pts[..., axis] >= lo[axis] - 1e-6
                if np.isfinite(hi[axis]):
                    inside &= pts[..., axis] <= hi[axis] + 1e-6
            t[~inside] = np.inf
        return t


@dataclass
class Box:
    """Axis-aligned box, used as a stand-in for a piece of furniture."""

    center: np.ndarray
    half_extents: np.ndarray
    color: tuple[int, int, int]
    name: str

    @property
    def min_corner(self) -> np.ndarray:
        return self.center - self.half_extents

    @property
    def max_corner(self) -> np.ndarray:
        return self.center + self.half_extents

    def intersect(self, origin: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        """Slab method, vectorised over a whole image of rays."""
        inv = np.divide(
            1.0, dirs, out=np.full_like(dirs, np.inf), where=np.abs(dirs) > 1e-12
        )
        t0 = (self.min_corner - origin) * inv
        t1 = (self.max_corner - origin) * inv
        t_near = np.minimum(t0, t1).max(axis=-1)
        t_far = np.maximum(t0, t1).min(axis=-1)

        t = np.where((t_far >= t_near) & (t_far > 1e-6), t_near, np.inf)
        # A ray starting inside the box has t_near < 0; use the exit point.
        t = np.where((t < 1e-6) & np.isfinite(t_far) & (t_far > 1e-6), t_far, t)
        t[t <= 1e-6] = np.inf
        return t


@dataclass
class SyntheticScene:
    image: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32, along the optical axis
    camera: PinholeCamera
    masks: dict[str, np.ndarray] = field(default_factory=dict)  # name -> bool
    planes: list[Plane] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)

    @property
    def object_masks(self) -> dict[str, np.ndarray]:
        return {b.name: self.masks[b.name] for b in self.boxes}

    @property
    def background_mask(self) -> np.ndarray:
        bg = np.ones(self.depth.shape, dtype=bool)
        for b in self.boxes:
            bg &= ~self.masks[b.name]
        return bg

    def ground_truth_floor(self) -> Plane | None:
        return next((p for p in self.planes if p.name == "floor"), None)

    def pil_image(self):
        from PIL import Image

        return Image.fromarray(self.image)


def default_room(
    width: int = 640,
    height: int = 480,
    hfov_deg: float = 60.0,
    include_ceiling: bool = True,
) -> tuple[list[Plane], list[Box]]:
    """A 4m x 2.6m room with a table and a pot on it, seen from ~1.4m up.

    Deliberately the "clean, simple test scene" the project brief names as
    the Tier 2 target: well-separated objects, one flat back wall, an
    unambiguous floor.
    """
    inf = np.inf
    floor_y = -1.4  # camera eye height above the floor
    ceil_y = 1.2
    back_z = -4.5
    left_x = -2.2
    right_x = 2.2

    planes = [
        Plane(np.array([0.0, 1.0, 0.0]), -floor_y, (150, 138, 120), "floor",
              (np.array([left_x, -inf, back_z]), np.array([right_x, inf, 0.0]))),
        Plane(np.array([0.0, 0.0, 1.0]), -back_z, (198, 192, 182), "back_wall",
              (np.array([left_x, floor_y, -inf]), np.array([right_x, ceil_y, inf]))),
        Plane(np.array([1.0, 0.0, 0.0]), -left_x, (176, 172, 165), "left_wall",
              (np.array([-inf, floor_y, back_z]), np.array([inf, ceil_y, 0.0]))),
        Plane(np.array([-1.0, 0.0, 0.0]), right_x, (182, 178, 170), "right_wall",
              (np.array([-inf, floor_y, back_z]), np.array([inf, ceil_y, 0.0]))),
    ]
    if include_ceiling:
        planes.append(
            Plane(np.array([0.0, -1.0, 0.0]), ceil_y, (226, 224, 220), "ceiling",
                  (np.array([left_x, -inf, back_z]), np.array([right_x, inf, 0.0])))
        )

    table_top_y = floor_y + 0.75
    boxes = [
        Box(
            center=np.array([-0.15, floor_y + 0.375, -2.6]),
            half_extents=np.array([0.7, 0.375, 0.45]),
            color=(122, 84, 52),
            name="table",
        ),
        Box(
            center=np.array([-0.1, table_top_y + 0.14, -2.55]),
            half_extents=np.array([0.13, 0.14, 0.13]),
            color=(190, 96, 62),
            name="pot",
        ),
        Box(
            center=np.array([1.35, floor_y + 0.28, -3.4]),
            half_extents=np.array([0.22, 0.28, 0.22]),
            color=(96, 122, 150),
            name="crate",
        ),
    ]
    return planes, boxes


def render(
    planes: list[Plane],
    boxes: list[Box],
    width: int = 640,
    height: int = 480,
    hfov_deg: float = 60.0,
    shade: bool = True,
    noise_std: float = 0.0,
    seed: int = 0,
) -> SyntheticScene:
    """Ray-trace the scene into a depth map, colour image and masks.

    `noise_std` adds relative Gaussian noise to depth, so the geometry code
    can be tested against something less unfairly perfect than exact depth —
    a real monocular depth map is noisy and slightly warped, and thresholds
    tuned on noiseless input will not survive contact with one.
    """
    camera = PinholeCamera.from_fov(width, height, hfov_deg)
    origin = np.zeros(3)
    dirs = camera.ray_directions()  # (H, W, 3), unit length

    surfaces: list = list(planes) + list(boxes)
    ts = np.stack([s.intersect(origin, dirs) for s in surfaces], axis=0)
    nearest = np.argmin(ts, axis=0)
    t_hit = np.take_along_axis(ts, nearest[None], axis=0)[0]

    missed = ~np.isfinite(t_hit)
    t_hit = np.where(missed, 1e4, t_hit)

    # t is distance along the (unit) ray; the pipeline everywhere expects
    # depth along the optical axis, so project it back onto -Z.
    points = origin + dirs * t_hit[..., None]
    depth = (-points[..., 2]).astype(np.float32)

    image = np.zeros((height, width, 3), dtype=np.float64)
    masks: dict[str, np.ndarray] = {}
    for i, surface in enumerate(surfaces):
        m = (nearest == i) & ~missed
        masks[surface.name] = m
        image[m] = surface.color

    if shade:
        # Cheap Lambert-ish shading purely so the texture has structure for
        # a segmenter to latch onto; the geometry does not depend on it.
        normals = _surface_normals(surfaces, nearest, points, missed)
        light = np.array([0.35, 0.9, 0.25])
        light /= np.linalg.norm(light)
        lambert = np.clip(np.abs(normals @ light), 0.0, 1.0)
        image *= (0.55 + 0.45 * lambert)[..., None]

    image[missed] = (30, 30, 34)

    if noise_std > 0:
        rng = np.random.default_rng(seed)
        depth = depth * (1.0 + rng.normal(0.0, noise_std, depth.shape).astype(np.float32))

    return SyntheticScene(
        image=np.clip(image, 0, 255).astype(np.uint8),
        depth=depth,
        camera=camera,
        masks=masks,
        planes=planes,
        boxes=boxes,
    )


def _surface_normals(surfaces, nearest, points, missed) -> np.ndarray:
    normals = np.zeros(points.shape, dtype=np.float64)
    for i, surface in enumerate(surfaces):
        m = (nearest == i) & ~missed
        if not m.any():
            continue
        if isinstance(surface, Plane):
            normals[m] = surface.normal
        else:
            # Box face normal: the axis on which the hit point is closest to
            # a face, in units of that axis's half-extent.
            local = (points[m] - surface.center) / surface.half_extents
            axis = np.argmax(np.abs(local), axis=1)
            n = np.zeros_like(local)
            n[np.arange(len(local)), axis] = np.sign(
                local[np.arange(len(local)), axis]
            )
            normals[m] = n
    return normals


def default_scene(**kwargs) -> SyntheticScene:
    """The standard test room, rendered."""
    width = kwargs.pop("width", 640)
    height = kwargs.pop("height", 480)
    hfov_deg = kwargs.pop("hfov_deg", 60.0)
    include_ceiling = kwargs.pop("include_ceiling", True)
    planes, boxes = default_room(width, height, hfov_deg, include_ceiling)
    return render(planes, boxes, width=width, height=height, hfov_deg=hfov_deg, **kwargs)
