"""
The placement solver: decide where each generated mesh actually goes.

This is the core of the project. Everything before it is pretrained models
doing perception and generation; this is the step that turns a pile of
unrelated canonical meshes into a scene that matches the photograph.

## The problem

TripoSR hands back each object centred at the origin, normalised to unit
size, in whatever pose its own latent space felt like. What we need is a
similarity transform per object — isotropic scale `s`, rotation `R`,
translation `t` — such that the transformed mesh sits where the photo says
the real object sits.

The evidence we have is the object's **target cloud**: its mask pixels
back-projected through the depth map. That cloud is the answer to "where is
this object in space, according to the photograph".

## Why this is not just ICP

The target cloud is only the object's **visible front surface**. The
generated mesh is a **complete closed object**. Registering one against the
other naively fails in a specific and instructive way: the mesh's back-side
vertices are perfectly happy to claim target points as their nearest
neighbours, which drags the mesh forward and shrinks it until its *back*
lies on the observed front surface. The object ends up half-buried in the
wall behind it.

So every iteration re-computes which mesh vertices would actually be
visible from the camera — a front-facing test plus a z-buffer occlusion
test — and only those participate in the matching. Once visibility is
handled, the objective can be symmetric (both "every observed point is
explained by the mesh" and "every visible part of the mesh is supported by
observations"), which is what pins down scale. One-sided matching alone
leaves scale badly under-determined when only a small patch of the object
is visible.

## The solve

Given correspondences, the transform is solved in **closed form**, not by
handing an objective to a generic optimizer:

- scale + translation with rotation fixed → an exact least-squares solution
- full similarity → Umeyama's method (SVD of the cross-covariance)
- yaw-only rotation → an exact `atan2` solution in the ground plane

ICP then alternates: associate, solve exactly, repeat.

## Solving order

Rotation is solved *second*, deliberately. From a single view under
occlusion the rotational signal is weak and noisy — a mostly-hidden object
supports many orientations almost equally well — so letting rotation move
early lets it absorb error that really belongs to scale, and the solve
walks away. Phase 1 freezes rotation and nails scale and position; phase 2
then refines rotation from a good starting point.

Because TripoSR's canonical orientation is not a documented contract, phase
2 is seeded from several candidate yaws and the best final fit wins, rather
than trusting one assumed convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .camera import PinholeCamera
from .meshing import backproject_mask
from .objects import measure_occlusion
from .room_shell import FittedPlane, RoomShell
from .segmentation import Instance

UP = np.array([0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# parameters and results
# ---------------------------------------------------------------------------


@dataclass
class PlacementParams:
    """Tunables for the solver. Defaults are for indoor room photos."""

    # --- ICP ---
    phase1_iterations: int = 12      # scale + translation, rotation frozen
    phase2_iterations: int = 20      # rotation refinement
    convergence_tol: float = 5e-4    # stop when trimmed RMS stops moving (m)

    # Trimmed ICP: discard the worst correspondences, which are usually
    # errors rather than signal (noisy depth, masks leaking onto the
    # background). The two directions are trimmed *differently*, and that
    # asymmetry is not cosmetic — trimming them equally biases every object
    # about 10% too small.
    #
    # Why: a target point is a real observation of this object, so the mesh
    # is obliged to explain it, and the peripheral points near the
    # silhouette are exactly the ones carrying the size signal. Trim them
    # and shrinking the mesh becomes free — the uncovered rim just gets
    # discarded. A *visible mesh vertex*, by contrast, may legitimately have
    # no matching observation (mask boundary, depth hole, generative shape
    # error), so that direction tolerates aggressive trimming.
    #
    # Measured on the ground-truth room: symmetric 0.80/0.80 put the
    # objective's minimum at 90% of true scale for every object; 0.95/0.80
    # moves it to 96-100%.
    trim_forward: float = 0.95    # target -> mesh: must be explained
    trim_backward: float = 0.80   # visible mesh -> target: may be unobserved

    # --- sampling ---
    max_mesh_points: int = 20000
    max_target_points: int = 6000

    # --- rotation ---
    # "upright" — try all 24 axis-aligned orientations as seeds, then refine
    #             yaw from the best. This is the default because TripoSR does
    #             NOT emit upright meshes: its output carries a fixed axis
    #             tilt (the v1 viewer shipped manual "straighten" buttons to
    #             work around it). Pure yaw refinement can never undo a tilt,
    #             so every object came out lying at whatever angle the
    #             generator chose. Seeding from the axis-aligned set lets the
    #             solver *discover* that convention from the data instead of
    #             hard-coding a rotation that may change with the model.
    # "yaw"     — rotate about the up axis only, no tilt correction. Correct
    #             when the mesh is already upright.
    # "full"    — unconstrained 3-DOF rotation (Umeyama). Most flexible,
    #             least stable from a single view.
    # "fixed"   — never rotate; use the initial guess.
    rotation_mode: str = "upright"
    yaw_starts: int = 12             # multi-start seeds around the up axis
    yaw_screen_iterations: int = 5   # cheap screening pass before committing
    yaw_finalists: int = 2

    # --- visibility ---
    # Set False only to demonstrate what visibility filtering is worth; the
    # solver is not usable without it (see scripts/selftest_assembly.py).
    use_visibility: bool = True
    visibility_grid: int = 80        # z-buffer resolution on the long side
    visibility_depth_tol: float = 0.03   # relative depth slack for "in front"
    front_facing_slack: float = 0.05     # cos tolerance on the normal test

    # --- sanity ---
    # Scale is bounded relative to the initializer (which comes from the
    # mask's angular size, and is hard to be badly wrong about). This stops
    # a diverging solve from producing a chair the size of the room.
    scale_bounds: tuple[float, float] = (0.35, 3.0)
    min_target_points: int = 60

    # Blend the fitted scale back toward the mask-derived initializer, by an
    # amount proportional to how much of the object is hidden.
    #
    # There are two independent estimates of an object's size and they fail
    # in different places. The 3D fit is the better one when most of the
    # object is visible. But when an object runs off the frame or is heavily
    # occluded, the visible patch stops constraining its extent — the fit
    # can shrink the object and simply not be penalised for it, because the
    # missing part was never observed. The mask's angular size does not have
    # that failure mode: a table cut off by the bottom of the photo still
    # shows its full width, and width times depth over focal length is its
    # real width.
    #
    # So the two are combined by reliability. Fully visible objects ignore
    # the prior entirely; the truncated table in the test scene goes from
    # 12% undersized to about 5%.
    scale_prior_strength: float = 1.0
    max_scale_prior: float = 0.45

    # --- support snapping ---
    # Force every object in a scene to share one frame-convention
    # correction, decided by a fit-quality-weighted vote. See place_objects.
    orientation_consensus: bool = True

    # Quality gate. Coverage is the fraction of the object's observed points
    # the placed mesh actually explains; below roughly half, the generated
    # mesh is not that object's shape and no transform will make it one.
    #
    # Showing such an object is strictly worse than showing nothing: the
    # viewer gets a blob AND loses the photo-accurate background pixels that
    # were cut away to make room for it. Dropping it puts the photograph
    # back. This is the cheapest large improvement available on real scenes.
    min_coverage: float = 0.6
    max_rms_error: float = 0.25   # metres; a fit this loose is meaningless

    # How much a confident semantic label relaxes the coverage bar.
    #
    # A hard coverage cutoff turned out to reject exactly the objects it was
    # meant to protect. On real photos, a museum jug behind reflective glass
    # was correctly detected and labelled "a vase" at 99.8% confidence, but
    # its placement only reached 0.578 coverage — just under the 0.6 bar —
    # because the glass reflections corrupt the depth around it. TripoSR's
    # mesh was probably fine; the *measurement* it was being judged against
    # was noisy. Low coverage is ambiguous between "bad mesh" and "hard
    # subject", and a blind threshold cannot tell them apart.
    #
    # A confident label is independent evidence that this is a real,
    # correctly-identified object, so it earns a lower bar rather than an
    # exemption — an object CLIP is unsure about still needs to prove itself
    # geometrically. effective_min_coverage = min_coverage - relief *
    # semantic_confidence, floored so nothing is ever fully exempt.
    # Widened from 0.35/0.20 after a real photo showed the gap: the actual
    # dish in a food photo was labelled correctly ("a bowl", 64% confidence)
    # but scored 0.334 coverage — its own reflective glaze and the steam/
    # shadow around it corrupt the depth the same way the museum jug's glass
    # case did. 0.35/0.20 still rejected it (threshold 0.375); 0.45/0.15
    # keeps it while still rejecting a synthetic low-confidence blob at
    # matching coverage — the floor stops relief from ever fully exempting
    # an object regardless of confidence.
    semantic_gate_relief: float = 0.45
    min_effective_coverage: float = 0.15

    # --- overlap resolution ---
    # Two objects sharing more than this fraction of the smaller one's
    # volume are treated as a solver error, not real furniture arrangement
    # (a pot resting on a table shares only a thin contact slice, not a
    # third of its own volume). The worse-fit one is dropped.
    max_overlap_fraction: float = 0.30

    snap_to_support: bool = True
    # Only snap when the correction is small. A big correction means the
    # solver and the support disagree substantially, and overriding a
    # confident solve with a guess makes things worse, not better.
    snap_tolerance: float = 0.15
    footprint_overlap: float = 0.15  # min IoU-ish overlap to count as support

    seed: int = 0


@dataclass
class Placement:
    """Where one object ended up, and how much to trust it."""

    instance_id: int
    scale: float = 1.0
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))

    rms_error: float = float("nan")   # trimmed RMS correspondence distance, m
    coverage: float = 0.0             # fraction of target points explained
    iterations: int = 0
    n_target_points: int = 0
    occlusion: float = 0.0
    support: str = "none"             # floor | object:N | none
    snap_offset: float = 0.0
    scale_prior_weight: float = 0.0   # how much the mask prior was trusted
    seed_index: int = -1              # which axis-aligned orientation won
    gate_threshold: float = float("nan")  # effective min_coverage actually applied
    status: str = "ok"                # ok | failed:<reason>

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def matrix(self) -> np.ndarray:
        """4x4 homogeneous transform, ready for trimesh / glTF."""
        m = np.eye(4)
        m[:3, :3] = self.scale * self.rotation
        m[:3, 3] = self.translation
        return m

    def apply(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float64) @ (self.scale * self.rotation).T + self.translation

    def yaw_degrees(self, up: np.ndarray = UP) -> float:
        """Rotation about the up axis, in degrees — the interpretable part."""
        u, v = _plane_basis(up)
        rotated = self.rotation @ u
        return math.degrees(math.atan2(float(rotated @ v), float(rotated @ u)))

    def summary(self) -> dict:
        """Compact, JSON-safe record for the API and the stats panel."""
        return {
            "instance_id": self.instance_id,
            "scale": round(float(self.scale), 4),
            "error": round(float(self.rms_error), 4) if np.isfinite(self.rms_error) else None,
            "coverage": round(float(self.coverage), 3),
            "yaw_deg": round(self.yaw_degrees(), 1),
            "translation": [round(float(x), 3) for x in self.translation],
            "support": self.support,
            "snap_offset": round(float(self.snap_offset), 4),
            "scale_prior_weight": round(float(self.scale_prior_weight), 3),
            "seed_index": int(self.seed_index),
            "gate_threshold": (round(float(self.gate_threshold), 3)
                              if np.isfinite(self.gate_threshold) else None),
            "iterations": self.iterations,
            "target_points": self.n_target_points,
            "occlusion": round(float(self.occlusion), 3),
            "status": self.status,
        }


# ---------------------------------------------------------------------------
# closed-form transform solves
# ---------------------------------------------------------------------------


def _plane_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal axes spanning the plane perpendicular to `up`."""
    n = np.asarray(up, dtype=np.float64)
    n = n / max(np.linalg.norm(n), 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    u = np.cross(ref, n)
    u /= max(np.linalg.norm(u), 1e-12)
    v = np.cross(n, u)
    return u, v


def solve_scale_translation(src: np.ndarray, dst: np.ndarray, rotation: np.ndarray):
    """Least-squares isotropic scale and translation with rotation fixed.

    Minimising Σ‖s·R·mᵢ + t − pᵢ‖² over (s, t) has an exact solution: centre
    both sets, and the optimal scale is the ratio of the cross-term to the
    source's own second moment. No iteration, no learning rate, no
    convergence to worry about.
    """
    src_r = src @ rotation.T
    src_mean = src_r.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    a = src_r - src_mean
    b = dst - dst_mean

    denom = float((a * a).sum())
    if denom < 1e-12:
        return 1.0, dst_mean - src_mean
    s = float((a * b).sum() / denom)
    if not np.isfinite(s) or s <= 1e-9:
        s = 1.0
    return s, dst_mean - s * src_mean


def solve_similarity_umeyama(src: np.ndarray, dst: np.ndarray):
    """Full similarity transform (s, R, t) by Umeyama's method.

    The rotation that best aligns two centred point sets is recovered from
    the SVD of their cross-covariance. The `det` correction is what keeps it
    a rotation rather than a reflection — without it, a noisy or nearly
    planar correspondence set will happily return a mirrored object, which
    looks subtly and unfixably wrong in the final scene.
    """
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    a = src - src_mean
    b = dst - dst_mean

    cov = (b.T @ a) / len(src)
    u_mat, sing, vt = np.linalg.svd(cov)

    d = np.ones(3)
    if np.linalg.det(u_mat) * np.linalg.det(vt) < 0:
        d[-1] = -1.0
    rotation = u_mat @ np.diag(d) @ vt

    var_src = float((a * a).sum() / len(src))
    if var_src < 1e-12:
        return 1.0, rotation, dst_mean - src_mean
    s = float((sing * d).sum() / var_src)
    if not np.isfinite(s) or s <= 1e-9:
        s = 1.0
    return s, rotation, dst_mean - s * (rotation @ src_mean)


def solve_similarity_yaw(src: np.ndarray, dst: np.ndarray, up: np.ndarray):
    """Similarity transform with rotation constrained to the up axis.

    Also exact. Projecting both point sets into the ground plane reduces the
    rotation to a single angle, and the least-squares angle is an `atan2` of
    the summed cross and dot terms — the 2D Procrustes solution.

    Worth constraining: an upright prior is correct for nearly everything in
    a room, and removing two rotational degrees of freedom removes exactly
    the two that a single viewpoint constrains worst.
    """
    axis = np.asarray(up, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    e0, e1 = _plane_basis(axis)

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    a = src - src_mean
    b = dst - dst_mean

    a0, a1, a2 = a @ e0, a @ e1, a @ axis
    b0, b1, b2 = b @ e0, b @ e1, b @ axis

    theta = math.atan2(
        float((b1 * a0 - b0 * a1).sum()),
        float((b0 * a0 + b1 * a1).sum()),
    )
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    # Rebuild the 3D rotation from the in-plane angle.
    basis = np.stack([e0, e1, axis])  # rows
    local = np.array([[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]])
    rotation = basis.T @ local @ basis

    # Optimal isotropic scale given that rotation.
    rotated_a = np.stack([cos_t * a0 - sin_t * a1, sin_t * a0 + cos_t * a1, a2], axis=1)
    b_local = np.stack([b0, b1, b2], axis=1)
    denom = float((a * a).sum())
    if denom < 1e-12:
        return 1.0, rotation, dst_mean - src_mean
    s = float((rotated_a * b_local).sum() / denom)
    if not np.isfinite(s) or s <= 1e-9:
        s = 1.0
    return s, rotation, dst_mean - s * (rotation @ src_mean)


def axis_aligned_rotations() -> list[np.ndarray]:
    """The 24 rotations that map the coordinate axes onto themselves.

    Signed permutation matrices with determinant +1 — the rotation group of
    a cube. Used as orientation seeds: whatever fixed frame convention a
    generative model uses, the rotation that undoes it is almost always one
    of these (models are trained on upright, axis-aligned assets), and
    checking all 24 is cheap compared to being wrong.

    Determinant +1 matters. The other 24 signed permutations are
    reflections, and seeding with one produces a mirrored object that then
    fits its own mirror image plausibly well.
    """
    import itertools

    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            m = np.zeros((3, 3))
            for row, col in enumerate(perm):
                m[row, col] = signs[row]
            if abs(np.linalg.det(m) - 1.0) < 1e-9:
                mats.append(m)
    return mats


def rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation matrix."""
    n = np.asarray(axis, dtype=np.float64)
    n = n / max(np.linalg.norm(n), 1e-12)
    k = np.array([[0.0, -n[2], n[1]], [n[2], 0.0, -n[0]], [-n[1], n[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


# ---------------------------------------------------------------------------
# visibility
# ---------------------------------------------------------------------------


def visible_vertex_mask(
    points: np.ndarray,
    normals: np.ndarray | None,
    camera: PinholeCamera,
    params: PlacementParams | None = None,
) -> np.ndarray:
    """Which of these world-space vertices would the camera actually see?

    Two tests, because neither is sufficient alone:

    1. **Front-facing.** A vertex whose normal points away from the camera is
       on the far side of the object. Cheap, and catches most of the back.
    2. **Z-buffer occlusion.** Front-facing is not enough for anything
       non-convex — the inside of a mug's handle faces the camera but is
       hidden behind the mug's body. Projecting every vertex into a coarse
       image grid and keeping only the nearest per cell handles it.

    3. **Inside the frame.** A vertex that projects outside the image is not
       observable either, and this one matters more than it looks. Objects
       routinely run off the bottom of a photo — the table in the test scene
       does. Without this test, the parts of the mesh hanging past the frame
       edge demand target points that could never exist, and the solver
       shrinks the object until its silhouette fits inside the picture. The
       table came out 24% too small until this was added.

    The grid is deliberately coarse (a fraction of image resolution) so that
    several vertices land in each cell and the depth comparison is
    meaningful; at full resolution most cells would hold a single vertex and
    the test would pass everything.
    """
    params = params or PlacementParams()
    pts = np.asarray(points, dtype=np.float64)
    n_pts = len(pts)
    if n_pts == 0:
        return np.zeros(0, dtype=bool)

    uv_all, depth = camera.project(pts)
    ok = depth > 1e-6
    ok &= (
        (uv_all[:, 0] >= 0)
        & (uv_all[:, 0] <= camera.width - 1)
        & (uv_all[:, 1] >= 0)
        & (uv_all[:, 1] <= camera.height - 1)
    )

    if normals is not None and params.front_facing_slack is not None:
        ranges = np.linalg.norm(pts, axis=1, keepdims=True)
        view_dirs = pts / np.maximum(ranges, 1e-12)  # camera at the origin
        facing = np.einsum("ij,ij->i", np.asarray(normals, dtype=np.float64), view_dirs)
        ok &= facing < params.front_facing_slack

    if not ok.any():
        return ok

    u, v, d = uv_all[ok, 0], uv_all[ok, 1], depth[ok]

    span = max(u.max() - u.min(), v.max() - v.min(), 1e-6)
    cell = span / max(params.visibility_grid, 4)
    iu = np.clip(((u - u.min()) / cell).astype(np.int64), 0, params.visibility_grid + 1)
    iv = np.clip(((v - v.min()) / cell).astype(np.int64), 0, params.visibility_grid + 1)
    stride = params.visibility_grid + 2
    key = iv * stride + iu

    zbuf = np.full(stride * stride, np.inf)
    np.minimum.at(zbuf, key, d)
    front = d <= zbuf[key] * (1.0 + params.visibility_depth_tol)

    out = np.zeros(n_pts, dtype=bool)
    out[np.nonzero(ok)[0][front]] = True
    return out


# ---------------------------------------------------------------------------
# initialisation
# ---------------------------------------------------------------------------


def initial_scale_from_mask(
    instance: Instance, depth: np.ndarray, camera: PinholeCamera
) -> float:
    """First guess at the object's real size, from its angular size.

    A mask `w` pixels wide, at depth `d`, spans `w · d / fx` metres. Because
    the canonical mesh is normalised to unit maximum extent, that width *is*
    the scale factor, near enough to start from.

    This is a much better initializer than anything derived from the point
    cloud's own spread, and it is what the scale bounds are anchored to: the
    angular size of a mask is one of the few things about a monocular
    reconstruction that is hard to get badly wrong.
    """
    d = depth[instance.mask]
    d = d[np.isfinite(d) & (d > 0)]
    if len(d) == 0:
        return 1.0
    median_depth = float(np.median(d))

    x0, y0, x1, y1 = instance.bbox
    width_m = (x1 - x0) * median_depth / camera.fx
    height_m = (y1 - y0) * median_depth / camera.fy
    return float(max(width_m, height_m, 1e-3))


def _sample(points: np.ndarray, limit: int, rng: np.random.Generator):
    if len(points) <= limit:
        return points, np.arange(len(points))
    idx = rng.choice(len(points), size=limit, replace=False)
    return points[idx], idx


# ---------------------------------------------------------------------------
# the ICP core
# ---------------------------------------------------------------------------


def _icp(
    canonical: np.ndarray,
    normals: np.ndarray | None,
    target: np.ndarray,
    camera: PinholeCamera,
    s: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    up: np.ndarray,
    mode: str,
    iterations: int,
    scale_range: tuple[float, float],
    params: PlacementParams,
):
    """Alternate correspondence and exact solve.

    Correspondences are built in **both** directions each iteration:

    - target → visible mesh: every observed point should be explained
    - visible mesh → target: every visible part of the mesh should be
      supported by observations

    The second direction is what determines scale. With only the first, a
    mesh twice too large can still pass through the observed patch and score
    well. It is only safe to include because the mesh side has already been
    filtered to visible vertices — run it over all vertices and the object's
    hidden back would demand observations that can never exist.
    """
    from scipy.spatial import cKDTree

    target_tree = cKDTree(target)
    rms = float("nan")
    coverage = 0.0
    previous = None
    used = 0

    for used in range(1, iterations + 1):
        world = canonical @ (s * rotation).T + translation
        world_normals = None if normals is None else normals @ rotation.T

        if params.use_visibility:
            visible = visible_vertex_mask(world, world_normals, camera, params)
        else:
            visible = np.ones(len(world), dtype=bool)
        if visible.sum() < 12:
            # Nothing visible usually means the transform has wandered
            # behind the camera; keep the last good estimate.
            break

        vis_world = world[visible]
        vis_canonical = canonical[visible]
        vis_tree = cKDTree(vis_world)

        d_fwd, i_fwd = vis_tree.query(target, workers=-1)
        d_bwd, i_bwd = target_tree.query(vis_world, workers=-1)

        # Trim each direction against its own distribution — see the note on
        # trim_forward / trim_backward for why they must differ.
        keep_fwd = d_fwd <= max(float(np.quantile(d_fwd, params.trim_forward)), 1e-9)
        keep_bwd = d_bwd <= max(float(np.quantile(d_bwd, params.trim_backward)), 1e-9)
        if keep_fwd.sum() + keep_bwd.sum() < 12:
            break

        src = np.vstack([vis_canonical[i_fwd][keep_fwd], vis_canonical[keep_bwd]])
        dst = np.vstack([target[keep_fwd], target[i_bwd][keep_bwd]])
        dist = np.concatenate([d_fwd[keep_fwd], d_bwd[keep_bwd]])

        if mode == "fixed":
            s_new, t_new = solve_scale_translation(src, dst, rotation)
            r_new = rotation
        elif mode == "yaw":
            s_new, r_new, t_new = solve_similarity_yaw(src, dst, up)
        else:
            s_new, r_new, t_new = solve_similarity_umeyama(src, dst)

        s = float(np.clip(s_new, scale_range[0], scale_range[1]))
        rotation, translation = r_new, t_new

        rms = float(np.sqrt((dist**2).mean()))
        # "Explained" is judged against the object's own size, not an
        # absolute distance — 2 cm is a good match for a table and a total
        # miss for a mug.
        tolerance = max(0.02, 0.08 * s)
        coverage = float((d_fwd <= tolerance).mean())

        if previous is not None and abs(previous - rms) < params.convergence_tol:
            break
        previous = rms

    return s, rotation, translation, rms, coverage, used


def solve_placement(
    mesh,
    target: np.ndarray,
    camera: PinholeCamera,
    initial_scale: float,
    up: np.ndarray = UP,
    params: PlacementParams | None = None,
    instance_id: int = 0,
    occlusion: float = 0.0,
    orientation_seed: np.ndarray | None = None,
) -> Placement:
    """Solve one object's placement against its target cloud.

    `orientation_seed` pins the frame-convention correction instead of
    searching for it, which is how the scene-level consensus pass forces a
    stubborn object into line with the rest.
    """
    params = params or PlacementParams()
    rng = np.random.default_rng(params.seed + instance_id)

    result = Placement(instance_id=instance_id, occlusion=occlusion,
                       n_target_points=len(target))

    if mesh is None or len(getattr(mesh, "vertices", [])) < 4:
        result.status = "failed:no-mesh"
        return result
    if len(target) < params.min_target_points:
        result.status = "failed:too-few-target-points"
        return result

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    try:
        vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    except Exception:
        vertex_normals = None

    canonical, idx = _sample(vertices, params.max_mesh_points, rng)
    normals = None if vertex_normals is None else vertex_normals[idx]
    target_pts, _ = _sample(np.asarray(target, dtype=np.float64),
                            params.max_target_points, rng)

    scale_range = (initial_scale * params.scale_bounds[0],
                   initial_scale * params.scale_bounds[1])

    # Start the mesh at the target's centroid, at the initial scale. The
    # centroid of a front surface is not the centroid of the object, but ICP
    # corrects that within a couple of iterations once visibility filtering
    # is comparing like with like.
    t0 = target_pts.mean(axis=0) - initial_scale * (canonical.mean(axis=0))

    # ---- phase 1: scale + translation, rotation frozen -----------------
    s1, r1, t1, rms1, cov1, it1 = _icp(
        canonical, normals, target_pts, camera,
        initial_scale, np.eye(3), t0, up,
        mode="fixed", iterations=params.phase1_iterations,
        scale_range=scale_range, params=params,
    )

    if params.rotation_mode == "fixed":
        result.scale, result.rotation, result.translation = s1, r1, t1
        result.rms_error, result.coverage, result.iterations = rms1, cov1, it1
        _apply_scale_prior(result, canonical, normals, camera, initial_scale, params)
        return result

    # ---- phase 2: rotation, multi-start --------------------------------
    # TripoSR's canonical orientation is not a documented contract, and a
    # single-view rotation objective is riddled with shallow local minima,
    # so several yaw seeds are screened cheaply and the best few run to
    # convergence. Committing to one seed is how this stage silently returns
    # objects facing the wrong way.
    if params.rotation_mode == "upright":
        # Each seed is a candidate fix for the generator's frame convention;
        # yaw is then solved *within* that frame, so the object stays
        # upright relative to whichever seed wins.
        seeds = ([np.asarray(orientation_seed, dtype=np.float64)]
                 if orientation_seed is not None else axis_aligned_rotations())
        inner_mode = "yaw"
    else:
        seeds = [np.eye(3)]
        if params.rotation_mode in ("yaw", "full") and params.yaw_starts > 1:
            seeds += [
                rotation_about_axis(up, 2 * math.pi * k / params.yaw_starts)
                for k in range(1, params.yaw_starts)
            ]
        inner_mode = params.rotation_mode

    def run(seed, start_s, start_r, start_t, iterations):
        """ICP in the frame defined by `seed`.

        Rotating the canonical points by the seed up front means the inner
        solve never has to know about seeds at all — the composed rotation
        is simply (solved @ seed).
        """
        seeded = canonical @ seed.T
        seeded_normals = None if normals is None else normals @ seed.T
        out = _icp(
            seeded, seeded_normals, target_pts, camera,
            start_s, start_r, start_t, up,
            mode=inner_mode, iterations=iterations,
            scale_range=scale_range, params=params,
        )
        return out

    screened = []
    for seed in seeds:
        cand = run(seed, s1, np.eye(3), t1, params.yaw_screen_iterations)
        screened.append((cand, seed))

    screened.sort(key=lambda cs: _score(cs[0][3], cs[0][4]))
    best, best_seed = (s1, np.eye(3), t1, rms1, cov1, it1), r1

    for cand, seed in screened[: max(1, params.yaw_finalists)]:
        refined = run(seed, cand[0], cand[1], cand[2], params.phase2_iterations)
        if _score(refined[3], refined[4]) < _score(best[3], best[4]):
            best, best_seed = refined, seed

    result.scale = best[0]
    result.rotation = best[1] @ best_seed   # undo the frame, then orient
    result.translation = best[2]
    result.seed_index = next(
        (i for i, sd in enumerate(seeds) if np.allclose(sd, best_seed)), -1
    )
    result.rms_error, result.coverage = best[3], best[4]
    result.iterations = it1 + best[5]

    _apply_scale_prior(result, canonical, normals, camera, initial_scale, params)

    if not np.isfinite(result.rms_error):
        result.status = "failed:no-convergence"
    return result


def _apply_scale_prior(
    placement: Placement,
    canonical: np.ndarray,
    normals: np.ndarray | None,
    camera: PinholeCamera,
    initial_scale: float,
    params: PlacementParams,
) -> None:
    """Pull the fitted scale toward the mask prior, in place.

    Weighted by occlusion, so a fully visible object is untouched. The
    translation is corrected at the same time to hold the object's visible
    centroid still — rescaling alone would slide it toward or away from the
    camera, undoing the alignment the fit just achieved.
    """
    if params.scale_prior_strength <= 0 or placement.occlusion <= 0:
        return

    weight = min(placement.occlusion * params.scale_prior_strength,
                 params.max_scale_prior)
    if weight <= 0:
        return

    old_scale = placement.scale
    new_scale = (1.0 - weight) * old_scale + weight * initial_scale
    if not np.isfinite(new_scale) or new_scale <= 1e-6:
        return

    world = canonical @ (old_scale * placement.rotation).T + placement.translation
    world_normals = None if normals is None else normals @ placement.rotation.T
    visible = visible_vertex_mask(world, world_normals, camera, params)
    anchor = canonical[visible].mean(axis=0) if visible.sum() >= 12 else canonical.mean(axis=0)

    placement.translation = placement.translation + (old_scale - new_scale) * (
        placement.rotation @ anchor
    )
    placement.scale = float(new_scale)
    placement.scale_prior_weight = float(weight)


def _score(rms: float, coverage: float) -> float:
    """Rank candidate fits: low error, but not by abandoning the object.

    Plain RMS is gameable — a solve that explains a small easy corner of the
    object scores beautifully. Dividing by coverage makes a fit that
    accounts for more of the observation win over a tighter fit that
    accounts for less.
    """
    if not np.isfinite(rms):
        return float("inf")
    return rms / max(coverage, 0.05)


# ---------------------------------------------------------------------------
# support snapping
# ---------------------------------------------------------------------------


def _footprint(points: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned extent of a point set projected into the ground plane."""
    e0, e1 = _plane_basis(up)
    local = np.stack([points @ e0, points @ e1], axis=1)
    return local.min(axis=0), local.max(axis=0)


def _overlap_fraction(a: tuple, b: tuple) -> float:
    """Overlap of two 2D boxes, relative to the smaller one."""
    lo = np.maximum(a[0], b[0])
    hi = np.minimum(a[1], b[1])
    inter = np.prod(np.maximum(hi - lo, 0.0))
    if inter <= 0:
        return 0.0
    area_a = np.prod(np.maximum(a[1] - a[0], 1e-9))
    area_b = np.prod(np.maximum(b[1] - b[0], 1e-9))
    return float(inter / min(area_a, area_b))


def snap_placements_to_supports(
    placements: list[Placement],
    meshes: dict[int, object],
    floor: FittedPlane | None,
    up: np.ndarray = UP,
    params: PlacementParams | None = None,
) -> None:
    """Rest each object on whatever it is actually standing on, in place.

    The brief calls for snapping objects to the floor plane, which fixes the
    vertical drift that monocular depth always leaves behind. But snapping
    everything to the *floor* is wrong for the brief's own example scene — a
    flower pot on a table belongs on the table, and dropping it to the floor
    would be a worse error than the drift being corrected.

    So the support is chosen rather than assumed: objects are settled from
    the bottom up, and each one rests on the highest already-placed surface
    that lies beneath it and whose footprint it overlaps, falling back to the
    floor. Bottom-up ordering matters — a table has to be settled before the
    pot can be told to stand on it.

    Snapping is skipped when the required correction is large, because that
    means the solver and the support strongly disagree, and a confident
    solve is better evidence than a guessed support.
    """
    params = params or PlacementParams()
    if not params.snap_to_support:
        return

    axis = np.asarray(up, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-12)

    def height(points: np.ndarray) -> np.ndarray:
        if floor is not None:
            return floor.signed_distance(points)
        return points @ axis

    settled: list[tuple[Placement, tuple, float]] = []  # placement, footprint, top

    candidates = []
    for placement in placements:
        mesh = meshes.get(placement.instance_id)
        if not placement.ok or mesh is None:
            continue
        world = placement.apply(np.asarray(mesh.vertices, dtype=np.float64))
        h = height(world)
        candidates.append((placement, world, h))

    # Settle lowest-based objects first so supports exist before they're used.
    candidates.sort(key=lambda c: float(c[2].min()))

    for placement, world, h in candidates:
        base = float(h.min())
        footprint = _footprint(world, axis)

        support_height = 0.0 if floor is not None else None
        support_name = "floor" if floor is not None else "none"

        for other, other_footprint, other_top in settled:
            if _overlap_fraction(footprint, other_footprint) < params.footprint_overlap:
                continue
            # Only surfaces below this object can hold it up.
            if other_top <= base + params.snap_tolerance:
                if support_height is None or other_top > support_height:
                    support_height = other_top
                    support_name = f"object:{other.instance_id}"

        if support_height is None:
            settled.append((placement, footprint, float(h.max())))
            continue

        offset = support_height - base
        if abs(offset) <= params.snap_tolerance:
            placement.translation = placement.translation + offset * axis
            placement.snap_offset = float(offset)
            placement.support = support_name
            h = h + offset
        else:
            placement.support = "none:correction-too-large"

        settled.append((placement, footprint, float(h.max())))


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------


def place_objects(
    generated: list,
    instances: list[Instance],
    depth: np.ndarray,
    camera: PinholeCamera,
    shell: RoomShell | None = None,
    params: PlacementParams | None = None,
) -> list[Placement]:
    """Solve placement for every generated object, then settle them.

    `generated` is a list of `objects.GeneratedObject`; entries whose
    generation failed (`mesh is None`) are carried through as failed
    placements rather than dropped, so the scene stats still account for
    every detected object.
    """
    params = params or PlacementParams()
    by_id = {inst.id: inst for inst in instances}
    up = shell.up if shell is not None else UP
    floor = shell.floor if shell is not None else None

    placements: list[Placement] = []
    meshes: dict[int, object] = {}

    for item in generated:
        instance = by_id.get(item.instance_id)
        if instance is None:
            placements.append(
                Placement(instance_id=item.instance_id, status="failed:no-instance")
            )
            continue

        if item.mesh is None:
            placements.append(
                Placement(instance_id=item.instance_id, status="failed:no-mesh",
                          occlusion=getattr(item.crop, "occlusion", 0.0))
            )
            continue

        target = backproject_mask(
            instance.mask, depth, camera, max_points=params.max_target_points * 3
        )
        # Occlusion is recomputed from the mask rather than read off the
        # crop. It drives the scale prior, so it has to be right even when
        # placement is being run without the cropping stage (tests, or a
        # pipeline fed externally supplied meshes).
        placement = solve_placement(
            item.mesh,
            target,
            camera,
            initial_scale=initial_scale_from_mask(instance, depth, camera),
            up=up,
            params=params,
            instance_id=item.instance_id,
            occlusion=measure_occlusion(instance, depth),
        )
        placements.append(placement)
        meshes[item.instance_id] = item.mesh

    # --- scene-level orientation consensus -----------------------------
    # The generator's frame convention is a property of the *model*, not of
    # any one object, so every object in a scene shares the same tilt. But
    # solving it per object lets ambiguous ones disagree: a table cut off by
    # the frame edge fits its own 90-degree-rotated self about as well as
    # the correct pose, and comes out lying on its side next to correctly
    # oriented neighbours.
    #
    # So take a vote. Objects that fitted confidently (high coverage, low
    # error) carry more weight, and anything that disagrees with the
    # consensus is re-solved with the winning orientation pinned. One
    # well-observed object is enough to rescue several ambiguous ones.
    if params.rotation_mode == "upright" and params.orientation_consensus:
        seeds = axis_aligned_rotations()
        votes: dict[int, float] = {}
        for p in placements:
            if not p.ok or p.seed_index < 0:
                continue
            weight = p.coverage / max(p.rms_error, 1e-3)
            votes[p.seed_index] = votes.get(p.seed_index, 0.0) + weight

        if votes:
            winner = max(votes, key=votes.get)
            for p in placements:
                if not p.ok or p.seed_index == winner or p.seed_index < 0:
                    continue
                instance = by_id.get(p.instance_id)
                mesh = meshes.get(p.instance_id)
                if instance is None or mesh is None:
                    continue
                target = backproject_mask(
                    instance.mask, depth, camera,
                    max_points=params.max_target_points * 3,
                )
                redone = solve_placement(
                    mesh, target, camera,
                    initial_scale=initial_scale_from_mask(instance, depth, camera),
                    up=up, params=params, instance_id=p.instance_id,
                    occlusion=measure_occlusion(instance, depth),
                    orientation_seed=seeds[winner],
                )
                if redone.ok:
                    # Report the consensus seed, not index 0 of the
                    # single-element forced list.
                    redone.seed_index = winner
                    placements[placements.index(p)] = redone

    # --- quality gate ---------------------------------------------------
    for p in placements:
        if not p.ok:
            continue

        threshold = params.min_coverage
        instance = by_id.get(p.instance_id)
        if instance is not None and instance.meta.get("semantic_category") == "object":
            confidence = instance.meta.get("semantic_confidence", 0.0) or 0.0
            threshold = max(
                params.min_effective_coverage,
                params.min_coverage - params.semantic_gate_relief * confidence,
            )
        p.gate_threshold = threshold

        if threshold > 0 and p.coverage < threshold:
            p.status = f"rejected:low-coverage {p.coverage:.2f} < {threshold:.2f}"
        elif np.isfinite(p.rms_error) and p.rms_error > params.max_rms_error:
            p.status = f"rejected:poor-fit {p.rms_error:.2f}m"

    snap_placements_to_supports(placements, meshes, floor, up, params)
    resolve_overlaps(placements, meshes, params)
    return placements


def _world_aabb(mesh, placement: "Placement") -> tuple[np.ndarray, np.ndarray]:
    world = placement.apply(np.asarray(mesh.vertices, dtype=np.float64))
    return world.min(axis=0), world.max(axis=0)


def _aabb_overlap_fraction(
    a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
) -> float:
    """Intersection volume over the smaller box's own volume, in [0, 1].

    Relative to the smaller object rather than the union: a small object
    fully swallowed by a large one scores 1.0 either way round, which is
    what should trigger a rejection regardless of which one happens to be
    "a" versus "b".
    """
    lo = np.maximum(a[0], b[0])
    hi = np.minimum(a[1], b[1])
    inter = np.prod(np.maximum(hi - lo, 0.0))
    if inter <= 0:
        return 0.0
    vol_a = np.prod(np.maximum(a[1] - a[0], 1e-9))
    vol_b = np.prod(np.maximum(b[1] - b[0], 1e-9))
    return float(inter / max(min(vol_a, vol_b), 1e-9))


def resolve_overlaps(
    placements: list["Placement"], meshes: dict[int, object], params: PlacementParams
) -> None:
    """Drop the worse-fit object out of any pair that clips through each other.

    Each object is placed independently against its own target cloud, so
    nothing before this point stops two of them from occupying the same
    space — a solver error on one object can happily land it halfway inside
    its neighbour. This is deliberately the crude version of the joint
    reasoning a system like Picasso does properly (see project notes): no
    physics, no repositioning, just "if these two occupy mostly the same
    volume, keep the one with better evidence and drop the other" — but
    dropping is enough to stop a scene from reading as visibly broken, which
    is the actual, immediate problem.

    Volume overlap, not 2D footprint overlap, is what is tested, so a pot
    legitimately resting on a table is unaffected: their AABBs touch only
    at a thin contact slice near the table's top, which is a tiny fraction
    of the pot's own volume, not a large one. Two chairs placed on top of
    each other by a bad fit, by contrast, share most of their volume.
    """
    ok = [p for p in placements if p.ok]
    boxes = {
        p.instance_id: _world_aabb(meshes[p.instance_id], p)
        for p in ok
        if meshes.get(p.instance_id) is not None
    }

    dropped: set[int] = set()
    for i in range(len(ok)):
        for j in range(i + 1, len(ok)):
            a, b = ok[i], ok[j]
            if a.instance_id in dropped or b.instance_id in dropped:
                continue
            if a.instance_id not in boxes or b.instance_id not in boxes:
                continue
            frac = _aabb_overlap_fraction(boxes[a.instance_id], boxes[b.instance_id])
            if frac < params.max_overlap_fraction:
                continue

            # Keep the better-supported placement; combine coverage and fit
            # error the same way _score() ranks candidates during solving,
            # so this is judging by the same evidence the solver itself
            # trusted, not a separate ad hoc rule.
            score_a = _score(a.rms_error, a.coverage)
            score_b = _score(b.rms_error, b.coverage)
            loser = b if score_a <= score_b else a
            loser.status = (
                f"rejected:overlaps-object:{a.instance_id if loser is b else b.instance_id} "
                f"({frac:.2f} of its volume)"
            )
            dropped.add(loser.instance_id)


def kept_instance_ids(placements: list[Placement]) -> set[int]:
    """Instances that survived placement and the quality gate.

    Only these should have their pixels cut out of the background mesh —
    a rejected object leaves no hole, so the photograph shows through where
    it would have been.
    """
    return {p.instance_id for p in placements if p.ok}
