"""
Room shell: fit planes to the non-object part of the scene and turn them
into walls, a floor and a ceiling.

Input is the background point cloud — every pixel that no object mask
claimed, back-projected through the depth map. Output is a small textured
mesh approximating the room's structure, plus the fitted plane equations,
which the assembly solver needs (an object's base gets snapped to the floor
plane, so that plane has to be right).

The RANSAC here is written out rather than delegated to Open3D's
`segment_plane`, for two reasons beyond "the brief asks for own code":

  1. **Spatially-coherent sampling.** Textbook RANSAC picks three points
     uniformly at random. In a room-sized cloud, three uniformly random
     points are almost never on the same surface, so the odds of proposing
     a good plane are terrible and you need thousands of iterations. Seeding
     from one point and drawing its two companions from its local
     neighbourhood raises the hit rate by orders of magnitude. This is the
     single change that makes plane fitting on a 100k-point room cloud
     finish in under a second.

  2. **Normal priors.** Rooms are not arbitrary. Once you know which way is
     up, you can insist that a candidate floor be roughly horizontal, which
     stops RANSAC from confidently fitting a plane through a bed, a rug and
     half a sofa because that happens to have more inliers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import numpy as np
from PIL import Image

from .camera import PinholeCamera

UP = np.array([0.0, 1.0, 0.0])


@dataclass
class FittedPlane:
    """A plane fitted to part of the background cloud.

    Convention: `normal` always points toward the camera (into the room), so
    `signed_distance` is positive for points on the room side. Without this
    normalisation, "is the object above the floor" flips sign depending on
    which triplet RANSAC happened to sample.
    """

    normal: np.ndarray  # unit
    d: float  # dot(normal, p) + d == 0 on the plane
    inlier_points: np.ndarray  # (N, 3)
    inlier_index: np.ndarray  # indices into the cloud it was fitted to
    kind: str = "wall"  # floor | ceiling | wall | other
    rms_error: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def n_inliers(self) -> int:
        return len(self.inlier_points)

    @property
    def centroid(self) -> np.ndarray:
        return self.inlier_points.mean(axis=0)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float64) @ self.normal + self.d

    def project_onto(self, points: np.ndarray) -> np.ndarray:
        """Drop points perpendicularly onto the plane."""
        pts = np.asarray(points, dtype=np.float64)
        return pts - self.signed_distance(pts)[:, None] * self.normal

    def basis(self) -> tuple[np.ndarray, np.ndarray]:
        """Two orthonormal in-plane axes.

        The first axis is aligned with world-up where that is meaningful, so
        a wall's local +v really is "up the wall". That keeps the 2D hull
        below axis-aligned in a way that matches the room, which matters
        when the hull is later tightened to a rectangle.
        """
        n = self.normal
        ref = UP if abs(n @ UP) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(ref, n)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        return u, v


def plane_from_points(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray):
    """Plane through three points, or None if they are near-collinear."""
    n = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return None
    n = n / norm
    return n, float(-n @ p0)


def refit_plane_lstsq(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Total-least-squares plane through a point set.

    The RANSAC consensus set is found from a 3-point sample, which is a
    noisy estimate. Refitting to all inliers with SVD (the smallest singular
    vector of the centred points is the normal) is what actually makes the
    plane accurate; skipping this step is a common and costly shortcut.
    """
    centroid = points.mean(axis=0)
    centred = points - centroid
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    return normal, float(-normal @ centroid)


def _orient_toward_camera(normal: np.ndarray, d: float, centroid: np.ndarray):
    """Flip the plane so its normal points back at the camera (the origin)."""
    if normal @ (-centroid) < 0:
        return -normal, -d
    return normal, d


def estimate_point_normals(
    points: np.ndarray, k: int = 32, neighbours: np.ndarray | None = None
) -> np.ndarray | None:
    """Per-point surface normal by PCA over each point's k neighbours.

    Why this exists: plane RANSAC scored purely on inlier *count* has a
    failure mode that is easy to miss and ruins a room fit. A plane tilted
    to graze a large flat surface at a shallow angle sweeps a band across
    it and collects thousands of inliers — all of them genuinely within the
    distance threshold, none of them actually part of that plane. On the
    test room this produced a "floor" made entirely of back-wall pixels,
    tilted 20 degrees and 13cm out of position.

    Position alone cannot tell those apart, but orientation can: a point on
    a vertical wall has a horizontal normal, and no amount of grazing makes
    it agree with a floor. So each point gets a local normal (the smallest
    principal direction of its neighbourhood), and a point only counts as
    an inlier if its own normal agrees with the candidate plane's.

    Returns None if SciPy is unavailable, in which case the caller falls
    back to the distance-only test.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None

    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < k + 1:
        return None

    if neighbours is None:
        _, idx = cKDTree(pts).query(pts, k=k + 1, workers=-1)
    else:
        idx = np.concatenate([np.arange(len(pts))[:, None], neighbours[:, :k]], axis=1)

    nbrs = pts[idx]  # (N, k+1, 3)
    centred = nbrs - nbrs.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centred, centred) / centred.shape[1]
    # eigh returns eigenvalues ascending, so column 0 is the direction of
    # least variance — the surface normal.
    _, vecs = np.linalg.eigh(cov)
    normals = vecs[:, :, 0]
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


@dataclass
class RansacParams:
    # Inlier distance, in scene units (metres for the metric depth model).
    # 3cm is loose enough for monocular depth noise on a wall at 4m, tight
    # enough not to swallow a bookshelf standing against that wall.
    distance_threshold: float = 0.03
    iterations: int = 400
    min_inliers: int = 400
    min_inlier_fraction: float = 0.02
    # Draw the two companion points from the seed's k nearest neighbours.
    local_sampling: bool = True
    neighbourhood_k: int = 64
    # A candidate whose normal is more than this far from the prior is
    # rejected outright when a prior is supplied.
    normal_prior_tolerance_deg: float = 20.0
    # Reject inliers whose own local surface normal disagrees with the
    # candidate plane by more than this. Kills grazing fits; see
    # estimate_point_normals. Generous, because point normals from noisy
    # monocular depth are themselves noisy.
    use_point_normals: bool = True
    normal_agreement_deg: float = 35.0
    max_planes: int = 6
    seed: int = 0


def fit_plane_ransac(
    points: np.ndarray,
    params: RansacParams | None = None,
    normal_prior: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    point_normals: np.ndarray | None = None,
) -> FittedPlane | None:
    """Best single plane in `points`, or None if nothing fits well enough.

    `point_normals` may be supplied to avoid recomputing them for every
    plane in a sequential-RANSAC peel; they are computed here if omitted.
    """
    params = params or RansacParams()
    rng = rng or np.random.default_rng(params.seed)
    pts = np.asarray(points, dtype=np.float64)
    n_points = len(pts)
    if n_points < max(3, params.min_inliers // 4):
        return None

    neighbours = None
    if (params.local_sampling or params.use_point_normals) and n_points > 3 * params.neighbourhood_k:
        try:
            from scipy.spatial import cKDTree

            k = min(params.neighbourhood_k, n_points - 1)
            _, neighbours = cKDTree(pts).query(pts, k=k + 1, workers=-1)
            neighbours = neighbours[:, 1:]  # drop self
        except ImportError:
            neighbours = None

    if params.use_point_normals and point_normals is None:
        point_normals = estimate_point_normals(
            pts, k=min(32, max(8, params.neighbourhood_k // 2)), neighbours=neighbours
        )
    if not params.local_sampling:
        neighbours = None

    cos_agree = math.cos(math.radians(params.normal_agreement_deg))

    def inliers_of(normal: np.ndarray, d: float) -> np.ndarray:
        """Boolean inlier mask under the distance test and, when point
        normals are available, the orientation-agreement test."""
        ok = np.abs(pts @ normal + d) <= params.distance_threshold
        if point_normals is not None:
            ok &= np.abs(point_normals @ normal) >= cos_agree
        return ok

    cos_tol = math.cos(math.radians(params.normal_prior_tolerance_deg))
    best_count, best_plane = 0, None

    # Proposals are generated and scored in chunks: scoring all iterations
    # at once would need an (iterations x n_points) distance matrix, which
    # is gigabytes for a room-sized cloud.
    chunk = 32
    for start in range(0, params.iterations, chunk):
        n_iter = min(chunk, params.iterations - start)
        seeds = rng.integers(0, n_points, size=n_iter)

        if neighbours is not None:
            picks = rng.integers(0, neighbours.shape[1], size=(n_iter, 2))
            others = neighbours[seeds[:, None], picks]
        else:
            others = rng.integers(0, n_points, size=(n_iter, 2))

        p0 = pts[seeds]
        p1 = pts[others[:, 0]]
        p2 = pts[others[:, 1]]

        normals = np.cross(p1 - p0, p2 - p0)
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1e-9
        if not valid.any():
            continue
        normals = normals[valid] / lengths[valid, None]
        ds = -np.einsum("ij,ij->i", normals, p0[valid])

        if normal_prior is not None:
            aligned = np.abs(normals @ normal_prior) >= cos_tol
            if not aligned.any():
                continue
            normals, ds = normals[aligned], ds[aligned]

        # (candidates, n_points) support — bounded by the chunk size, since
        # scoring every proposal at once would need a matrix of gigabytes.
        support = np.abs(normals @ pts.T + ds[:, None]) <= params.distance_threshold
        if point_normals is not None:
            support &= np.abs(normals @ point_normals.T) >= cos_agree
        counts = support.sum(axis=1)
        best_i = int(np.argmax(counts))
        if counts[best_i] > best_count:
            best_count = int(counts[best_i])
            best_plane = (normals[best_i], float(ds[best_i]))

    if best_plane is None:
        return None
    min_needed = max(params.min_inliers, int(params.min_inlier_fraction * n_points))
    if best_count < min_needed:
        return None

    normal, d = best_plane
    inlier_index = np.nonzero(inliers_of(normal, d))[0]

    # Refit to the consensus set, then re-select inliers against the refined
    # plane — one round of this measurably tightens the fit.
    normal, d = refit_plane_lstsq(pts[inlier_index])
    inlier_index = np.nonzero(inliers_of(normal, d))[0]
    if len(inlier_index) < min_needed:
        return None
    normal, d = refit_plane_lstsq(pts[inlier_index])

    # The prior constrains proposals, but the refit can drift away from it;
    # re-check afterwards rather than trusting the proposal stage. A caller
    # asking for a floor should not be handed a plane 40 degrees off level.
    if normal_prior is not None and abs(float(normal @ normal_prior)) < cos_tol:
        return None

    inliers = pts[inlier_index]
    normal, d = _orient_toward_camera(normal, d, inliers.mean(axis=0))
    residual = inliers @ normal + d

    return FittedPlane(
        normal=normal,
        d=d,
        inlier_points=inliers,
        inlier_index=inlier_index,
        rms_error=float(np.sqrt((residual**2).mean())),
    )


def classify_plane(plane: FittedPlane, max_tilt_deg: float = 30.0) -> str:
    """Label a plane floor / ceiling / wall by its normal and position.

    World-up is taken as +Y, i.e. the photo is assumed roughly level. That
    assumption is stated rather than estimated because a level-ish photo is
    the normal case and getting gravity wrong from one image is its own
    research problem; `estimate_up_direction` below offers a correction when
    the assumption fails.
    """
    cos_up = float(plane.normal @ UP)
    tilt = math.degrees(math.acos(np.clip(abs(cos_up), 0.0, 1.0)))

    if tilt <= max_tilt_deg:
        # Normals point toward the camera, so a surface below the camera
        # with an upward normal is a floor; one above with a downward
        # normal is a ceiling.
        if cos_up > 0 and plane.centroid[1] < 0:
            return "floor"
        if cos_up < 0 and plane.centroid[1] > 0:
            return "ceiling"
        return "other"
    if tilt >= 90.0 - max_tilt_deg:
        return "wall"
    return "other"


def estimate_up_direction(planes: list[FittedPlane]) -> np.ndarray:
    """Refine world-up from the largest near-horizontal plane.

    If the photo was taken with the camera tilted down (very common — people
    photograph a table by pointing slightly downward), the true floor normal
    is not +Y, and every "snap the object to the floor" step inherits that
    tilt. Using the biggest horizontal-ish plane's own normal as up corrects
    for it. Falls back to +Y when no such plane exists.
    """
    horizontal = [
        p for p in planes if abs(float(p.normal @ UP)) > math.cos(math.radians(35))
    ]
    if not horizontal:
        return UP.copy()
    best = max(horizontal, key=lambda p: p.n_inliers)
    n = best.normal.copy()
    if n @ UP < 0:
        n = -n
    return n / np.linalg.norm(n)


def fit_floor_plane(
    points: np.ndarray,
    params: RansacParams | None = None,
    rng: np.random.Generator | None = None,
    max_attempts: int = 3,
    point_normals: np.ndarray | None = None,
) -> FittedPlane | None:
    """Find the floor specifically, rather than hoping it turns up.

    A plain horizontal normal prior is not enough on its own: a ceiling is
    exactly as horizontal as a floor, and in an indoor photo the ceiling is
    usually the *larger* unoccluded surface, so RANSAC picks it. That was a
    real failure — under depth noise this stage returned a ceiling, the
    floor-first pass rejected it, and the scene ended up with no floor at
    all, which in turn leaves the assembly step with nothing to snap objects
    to.

    Two fixes, both here:

      1. Only fit on points below the camera. A camera inside a room is
         always above its floor and below its ceiling, so this removes the
         ceiling from consideration entirely rather than trying to tell them
         apart after the fact.
      2. Retry. If a candidate still classifies as something other than a
         floor, drop its inliers and fit again — the floor is often the
         second or third horizontal surface (a table top is a plane too).
    """
    params = params or RansacParams()
    rng = rng or np.random.default_rng(params.seed)
    pts = np.asarray(points, dtype=np.float64)

    below = np.nonzero(pts[:, 1] < 0.0)[0]
    if len(below) < params.min_inliers:
        return None

    if point_normals is None and params.use_point_normals:
        point_normals = estimate_point_normals(pts)

    available = below
    for _ in range(max_attempts):
        if len(available) < params.min_inliers:
            return None
        # min_inlier_fraction is relative to the *whole* cloud, so relax it
        # here — the floor is a small slice of a room, especially once
        # furniture has occluded most of it.
        sub_params = replace(
            params,
            min_inlier_fraction=params.min_inlier_fraction * 0.25,
            min_inliers=max(100, params.min_inliers // 4),
        )
        candidate = fit_plane_ransac(
            pts[available],
            sub_params,
            normal_prior=UP,
            rng=rng,
            point_normals=None if point_normals is None else point_normals[available],
        )
        if candidate is None:
            return None

        candidate.inlier_index = available[candidate.inlier_index]
        if classify_plane(candidate) == "floor":
            candidate.kind = "floor"
            return candidate
        # Not a floor (a table top, a shelf) — remove it and look deeper.
        available = np.setdiff1d(available, candidate.inlier_index)

    return None


def extract_planes(
    points: np.ndarray,
    params: RansacParams | None = None,
    prefer_floor_first: bool = True,
) -> list[FittedPlane]:
    """Peel planes off the background cloud one at a time, largest first.

    Sequential RANSAC: fit the dominant plane, remove its inliers, repeat on
    what's left. Stops when a fit fails or too few points remain.

    `prefer_floor_first` runs one extra pass with a horizontal normal prior
    before the general search. The floor is the plane the assembly step
    depends on most (objects get snapped to it) but it is often *not* the
    largest background surface in an indoor photo — a wall usually is, and
    the floor is heavily occluded by the very furniture we're reconstructing.
    Fitting it deliberately, with a prior, rather than hoping it turns up in
    the top few, is what keeps object placement from drifting.
    """
    params = params or RansacParams()
    pts = np.asarray(points, dtype=np.float64)
    remaining = np.arange(len(pts))
    found: list[FittedPlane] = []
    rng = np.random.default_rng(params.seed)

    # Point normals are the expensive part (a kNN query plus a batched
    # eigendecomposition), so compute them once for the whole cloud and
    # slice them per peel rather than recomputing for every plane.
    point_normals = estimate_point_normals(pts) if params.use_point_normals else None

    if prefer_floor_first and len(pts) > params.min_inliers:
        floor = fit_floor_plane(pts, params, rng=rng, point_normals=point_normals)
        if floor is not None:
            found.append(floor)
            remaining = np.setdiff1d(remaining, floor.inlier_index)

    while len(found) < params.max_planes and len(remaining) >= params.min_inliers:
        subset = pts[remaining]
        plane = fit_plane_ransac(
            subset,
            params,
            rng=rng,
            point_normals=None if point_normals is None else point_normals[remaining],
        )
        if plane is None:
            break
        # Map indices back into the original cloud.
        plane.inlier_index = remaining[plane.inlier_index]
        plane.kind = classify_plane(plane)
        found.append(plane)
        remaining = np.setdiff1d(remaining, plane.inlier_index)

    found.sort(key=lambda p: p.n_inliers, reverse=True)
    return found


def plane_polygon(
    plane: FittedPlane,
    shrink: float = 0.0,
    use_bounding_rect: bool = False,
) -> np.ndarray | None:
    """Outline of a plane's extent, as an ordered (M, 3) 3D polygon.

    The plane equation is infinite; the room needs finite quads. The extent
    is taken as the convex hull of the plane's own inliers, projected into
    plane-local 2D and hulled there.

    `use_bounding_rect` replaces the hull with its axis-aligned box in plane
    coordinates. Walls really are rectangles, and the hull of a wall's
    inliers is a ragged polygon full of bites taken out of it by furniture,
    so for walls and floors the rectangle usually looks more like a room.
    The hull is the safer choice for anything irregular.
    """
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        return None

    u, v = plane.basis()
    origin = plane.centroid
    local = np.stack(
        [(plane.inlier_points - origin) @ u, (plane.inlier_points - origin) @ v],
        axis=1,
    )
    if len(local) < 3:
        return None

    if use_bounding_rect:
        # 2nd/98th percentile rather than min/max: a handful of stray
        # inliers should not inflate a wall by a metre.
        lo = np.percentile(local, 2, axis=0)
        hi = np.percentile(local, 98, axis=0)
        poly2d = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    else:
        try:
            hull = ConvexHull(local)
        except Exception:
            return None
        poly2d = local[hull.vertices]

    if shrink > 0:
        centre = poly2d.mean(axis=0)
        poly2d = centre + (poly2d - centre) * (1.0 - shrink)

    return origin + poly2d[:, 0:1] * u + poly2d[:, 1:2] * v


def _triangulate_fan(n_vertices: int, offset: int) -> np.ndarray:
    """Fan triangulation of a convex polygon."""
    return np.array(
        [[offset, offset + i, offset + i + 1] for i in range(1, n_vertices - 1)],
        dtype=np.int64,
    )


def build_shell_mesh(
    planes: list[FittedPlane],
    image: Image.Image,
    camera: PinholeCamera,
    use_bounding_rect_for_structure: bool = True,
    min_area: float = 0.15,
):
    """Fitted planes -> one textured room-shell mesh.

    Each plane becomes a polygon, textured by projecting its corners back
    into the source image. That reprojection is exact for the plane itself,
    which is why the walls come out looking like the photo's walls.

    Known artefact, worth stating plainly: where an object occluded the
    wall, the wall polygon still covers that region, so the *object's*
    pixels get painted onto the wall behind it. Tier 2 then puts a real 3D
    object in front of that smear. Inpainting the occluded background is a
    whole separate problem and is out of scope here.
    """
    import trimesh

    img = image.convert("RGB")
    w, h = img.size

    all_v: list[np.ndarray] = []
    all_f: list[np.ndarray] = []
    all_uv: list[np.ndarray] = []
    used: list[FittedPlane] = []
    offset = 0

    for plane in planes:
        rect = use_bounding_rect_for_structure and plane.kind in ("floor", "ceiling", "wall")
        poly = plane_polygon(plane, use_bounding_rect=rect)
        if poly is None or len(poly) < 3:
            continue

        # Skip slivers — a 15cm^2 patch is a mis-fit, not a wall.
        edges = np.linalg.norm(np.diff(np.vstack([poly, poly[:1]]), axis=0), axis=1)
        if edges.sum() < min_area:
            continue

        uv_px, depth = camera.project(poly)
        if not np.all(depth > 0):
            # Part of the polygon extends behind the camera; clamping its
            # UVs would smear the texture, so drop it.
            continue

        uv = np.stack(
            [uv_px[:, 0] / max(w - 1, 1), 1.0 - uv_px[:, 1] / max(h - 1, 1)], axis=1
        )
        faces = _triangulate_fan(len(poly), offset)
        if len(faces) == 0:
            continue

        all_v.append(poly)
        all_uv.append(np.clip(uv, 0.0, 1.0))
        all_f.append(faces)
        used.append(plane)
        offset += len(poly)

    if not all_v:
        return None, []

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=img,
        metallicFactor=0.0,
        roughnessFactor=1.0,
        doubleSided=True,  # the viewer will orbit outside the room
    )
    mesh = trimesh.Trimesh(
        vertices=np.vstack(all_v).astype(np.float32),
        faces=np.vstack(all_f),
        visual=trimesh.visual.TextureVisuals(
            uv=np.vstack(all_uv).astype(np.float32), material=material
        ),
        process=False,
    )
    return mesh, used


@dataclass
class RoomShell:
    mesh: object | None
    planes: list[FittedPlane]
    up: np.ndarray
    stats: dict = field(default_factory=dict)

    @property
    def floor(self) -> FittedPlane | None:
        floors = [p for p in self.planes if p.kind == "floor"]
        return max(floors, key=lambda p: p.n_inliers) if floors else None

    @property
    def walls(self) -> list[FittedPlane]:
        return [p for p in self.planes if p.kind == "wall"]


def fit_room_shell(
    background_points: np.ndarray,
    image: Image.Image,
    camera: PinholeCamera,
    params: RansacParams | None = None,
) -> RoomShell:
    """Background cloud -> fitted planes + shell mesh, with fit diagnostics.

    The stats returned are worth looking at rather than skipping past: an
    RMS error much above the inlier threshold, or a floor with very few
    inliers, means the shell is wrong and the object placement built on it
    will be wrong too.
    """
    params = params or RansacParams()
    planes = extract_planes(background_points, params)
    up = estimate_up_direction(planes)
    mesh, used = build_shell_mesh(planes, image, camera)

    stats = {
        "background_points": int(len(background_points)),
        "planes_found": len(planes),
        "planes_in_mesh": len(used),
        "up_direction": [round(float(x), 4) for x in up],
        "planes": [
            {
                "kind": p.kind,
                "inliers": p.n_inliers,
                "rms_error": round(p.rms_error, 4),
                "normal": [round(float(x), 3) for x in p.normal],
                "distance_from_camera": round(float(abs(p.d)), 3),
            }
            for p in planes
        ],
    }
    if mesh is not None:
        stats["shell_faces"] = int(len(mesh.faces))

    return RoomShell(mesh=mesh, planes=planes, up=up, stats=stats)
