"""
Pinhole camera model and the image <-> 3D projection math.

Every stage of the pipeline (Tier 1 meshing, room-shell fitting, object
placement) has to agree on exactly one thing: given a pixel (u, v) and a
depth value d, where is that point in 3D? If depth.py and assembly.py
disagree by so much as a sign flip, objects land behind walls. So all of
that math lives here and nowhere else.

Coordinate convention (glTF / three.js, right-handed):

    +X right, +Y up, -Z forward (the camera looks down -Z)

Image coordinates are the usual (u right, v down) with the origin at the
top-left pixel, which is why the Y term picks up a minus sign.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

# Most phone/consumer cameras land somewhere near a 60-70 degree horizontal
# field of view. Without EXIF we have to assume something, and this is the
# least-wrong constant to assume. See PinholeCamera.from_fov.
DEFAULT_HFOV_DEG = 60.0


@dataclass(frozen=True)
class PinholeCamera:
    """Intrinsics for one image. Pixel units."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    # ---- constructors -------------------------------------------------

    @classmethod
    def from_fov(
        cls,
        width: int,
        height: int,
        hfov_deg: float = DEFAULT_HFOV_DEG,
    ) -> "PinholeCamera":
        """Build intrinsics from an assumed horizontal field of view.

        fx = (W / 2) / tan(hfov / 2) falls straight out of the pinhole
        similar-triangles relation: a point at the right edge of the image
        sits at angle hfov/2 off the optical axis.

        Square pixels are assumed (fy == fx), which is true of essentially
        every real camera, so the vertical FOV follows from the aspect
        ratio rather than being a free parameter.
        """
        if not 1.0 < hfov_deg < 179.0:
            raise ValueError(f"hfov_deg out of range: {hfov_deg}")
        f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(width, height, f, f, width / 2.0, height / 2.0)

    @classmethod
    def from_focal_35mm(
        cls, width: int, height: int, focal_35mm: float
    ) -> "PinholeCamera":
        """Build intrinsics from a 35mm-equivalent focal length (EXIF).

        35mm-equivalent means "the focal length that would give this same
        FOV on a 36mm-wide sensor", so the sensor width cancels out and
        fx in pixels is just focal / 36 * image_width.
        """
        if focal_35mm <= 0:
            raise ValueError("focal_35mm must be positive")
        f = focal_35mm / 36.0 * width
        return cls(width, height, f, f, width / 2.0, height / 2.0)

    # ---- derived quantities -------------------------------------------

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan((self.width / 2.0) / self.fx))

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan((self.height / 2.0) / self.fy))

    def scaled(self, width: int, height: int) -> "PinholeCamera":
        """Intrinsics for the same camera at a different image resolution.

        Needed because we run depth at one resolution, mesh at another, and
        crop objects at a third. Intrinsics scale linearly with resolution;
        forgetting to rescale them is the classic way to get objects that
        are the right shape but the wrong size.
        """
        sx = width / self.width
        sy = height / self.height
        return PinholeCamera(
            width=width,
            height=height,
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
        )

    def intrinsic_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    # ---- projection ----------------------------------------------------

    def unproject(self, u, v, depth) -> np.ndarray:
        """Pixel(s) + depth -> 3D point(s), shape (..., 3).

        `depth` is distance along the optical axis (Z), not ray length.
        Accepts scalars or broadcastable arrays.
        """
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        d = np.asarray(depth, dtype=np.float64)

        x = (u - self.cx) / self.fx * d
        y = -(v - self.cy) / self.fy * d  # image v grows downward, world Y up
        z = -d  # camera looks down -Z
        return np.stack([x, y, z], axis=-1)

    def unproject_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        """Whole depth map -> point grid of shape (H, W, 3).

        The depth map must already match this camera's resolution; use
        `scaled()` if it doesn't, rather than silently resizing here.
        """
        h, w = depth_map.shape[:2]
        if (w, h) != (self.width, self.height):
            raise ValueError(
                f"depth map is {w}x{h} but camera is {self.width}x{self.height}; "
                "call camera.scaled(w, h) first"
            )
        vs, us = np.mgrid[0:h, 0:w]
        return self.unproject(us, vs, depth_map)

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """3D point(s) -> (pixel coords (..., 2), depth (...,)).

        The inverse of `unproject`. The assembly solver needs this to score
        a candidate placement against the object's 2D mask.

        Points at or behind the camera plane (z >= 0) get depth <= 0 and
        garbage pixel coords; callers must mask on the returned depth.
        """
        pts = np.asarray(points, dtype=np.float64)
        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
        d = -z
        safe = np.where(np.abs(d) < 1e-9, 1e-9, d)
        u = x / safe * self.fx + self.cx
        v = -y / safe * self.fy + self.cy
        return np.stack([u, v], axis=-1), d

    def ray_directions(self) -> np.ndarray:
        """Unit ray direction per pixel, shape (H, W, 3).

        Useful for depth-along-ray vs depth-along-Z conversions and for the
        grazing-angle test in the Tier 1 mesher.
        """
        vs, us = np.mgrid[0 : self.height, 0 : self.width]
        dirs = self.unproject(us, vs, np.ones_like(us, dtype=np.float64))
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
        return dirs / np.maximum(norms, 1e-12)

    def __repr__(self) -> str:  # pragma: no cover - debug convenience
        return (
            f"PinholeCamera({self.width}x{self.height}, f={self.fx:.1f}px, "
            f"hfov={self.hfov_deg:.1f}deg)"
        )


def camera_from_image(
    width: int,
    height: int,
    exif_focal_35mm: float | None = None,
    hfov_deg: float = DEFAULT_HFOV_DEG,
) -> PinholeCamera:
    """Best available intrinsics for an image: EXIF if we have it, else FOV.

    Scale is ambiguous either way with monocular depth (see README), but a
    wrong focal length also warps *shape* — a too-narrow FOV flattens the
    scene, a too-wide one stretches it into a funnel. So it is worth using
    EXIF when the photo carries it.
    """
    if exif_focal_35mm:
        return PinholeCamera.from_focal_35mm(width, height, exif_focal_35mm)
    return PinholeCamera.from_fov(width, height, hfov_deg)


def read_exif_focal_35mm(image_path: str) -> float | None:
    """Pull FocalLengthIn35mmFilm out of a JPEG, or None if absent."""
    try:
        from PIL import Image, ExifTags
    except ImportError:  # pragma: no cover
        return None

    try:
        with Image.open(image_path) as im:
            exif = im.getexif()
        if not exif:
            return None
        tag_ids = {v: k for k, v in ExifTags.TAGS.items()}
        value = exif.get(tag_ids.get("FocalLengthIn35mmFilm"))
        if value:
            return float(value)
    except Exception:
        return None
    return None
