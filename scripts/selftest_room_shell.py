"""
Ground-truth checks for the room-shell stage.

Run: python scripts/selftest_room_shell.py

The synthetic room's planes are known exactly, so RANSAC's output can be
compared against the truth in the only two ways that matter: is the normal
pointing the right way, and is the plane at the right distance. A floor that
is 8cm too low puts every object in the scene 8cm underground.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import synthetic
from pipeline.meshing import backproject_mask
from dataclasses import replace

from pipeline.room_shell import (
    RansacParams,
    classify_plane,
    estimate_up_direction,
    fit_plane_ransac,
    fit_room_shell,
    plane_polygon,
    refit_plane_lstsq,
)
from pipeline.segmentation import background_mask, instances_from_masks

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Degrees between two directions, ignoring sign."""
    c = abs(float(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(np.clip(c, 0.0, 1.0)))


def scene_background_cloud(scene, noise_std: float = 0.0):
    """The cloud the room-shell stage actually receives: background pixels
    only, back-projected, with object regions removed as segmentation would."""
    objects = instances_from_masks(list(scene.object_masks.values()))
    bg = background_mask(objects, scene.depth.shape)
    return backproject_mask(bg, scene.depth, scene.camera, max_points=60000,
                            depth_percentile_trim=None)


def test_single_plane_fit() -> None:
    print("\nransac: single plane against exact ground truth")
    scene = synthetic.default_scene()
    truth = scene.ground_truth_floor()

    pts = backproject_mask(scene.masks["floor"], scene.depth, scene.camera,
                           max_points=None, depth_percentile_trim=None)
    plane = fit_plane_ransac(pts, RansacParams(distance_threshold=0.01))

    check("fits a plane to clean floor points", plane is not None)
    if plane is None:
        return
    check("normal matches the true floor normal",
          angle_between(plane.normal, truth.normal) < 0.5,
          f"{angle_between(plane.normal, truth.normal):.3f} deg off")
    check("plane offset matches truth", abs(abs(plane.d) - abs(truth.d)) < 0.01,
          f"|d|={abs(plane.d):.4f} vs {abs(truth.d):.4f}")
    check("normal is oriented toward the camera", plane.normal @ -plane.centroid > 0)
    check("rms error is sub-millimetre", plane.rms_error < 1e-3,
          f"rms={plane.rms_error:.2e} m")
    check("captures nearly all the floor points",
          plane.n_inliers > 0.98 * len(pts), f"{plane.n_inliers}/{len(pts)}")


def test_lstsq_refit_beats_sample() -> None:
    print("\nransac: SVD refit improves on the 3-point sample")
    rng = np.random.default_rng(3)
    true_n = np.array([0.0, 1.0, 0.0])
    pts = np.stack(
        [rng.uniform(-2, 2, 4000), np.full(4000, -1.4), rng.uniform(-4, -1, 4000)], 1
    )
    pts += rng.normal(0, 0.02, pts.shape)

    fitted_n, fitted_d = refit_plane_lstsq(pts)
    check("refit recovers the true normal within 0.5deg",
          angle_between(fitted_n, true_n) < 0.5,
          f"{angle_between(fitted_n, true_n):.3f} deg")
    check("refit recovers the true offset", abs(abs(fitted_d) - 1.4) < 0.01,
          f"|d|={abs(fitted_d):.4f} vs 1.4")


def test_local_sampling_helps() -> None:
    print("\nransac: local sampling vs uniform sampling")
    scene = synthetic.default_scene()
    cloud = scene_background_cloud(scene)

    # Deliberately starved iteration budget — with a generous one both
    # strategies converge, and the test proves nothing. The claim being
    # made is that local sampling needs *fewer* proposals, so measure it
    # where the budget actually binds, averaged over seeds.
    local_scores, uniform_scores = [], []
    for seed in range(6):
        base = RansacParams(iterations=4, distance_threshold=0.02, seed=seed)
        loc = fit_plane_ransac(cloud, base)
        uni = fit_plane_ransac(cloud, replace(base, local_sampling=False))
        local_scores.append(loc.n_inliers if loc else 0)
        uniform_scores.append(uni.n_inliers if uni else 0)

    mean_local = float(np.mean(local_scores))
    mean_uniform = float(np.mean(uniform_scores))
    check("local sampling finds a plane on every seed at 4 iterations",
          all(s > 0 for s in local_scores), f"{local_scores}")
    check("local sampling beats uniform on a starved budget",
          mean_local > mean_uniform,
          f"mean inliers {mean_local:.0f} vs {mean_uniform:.0f}")


def test_full_shell_on_clean_scene() -> None:
    print("\nshell: full fit on the clean synthetic room")
    scene = synthetic.default_scene()
    cloud = scene_background_cloud(scene)
    shell = fit_room_shell(cloud, scene.pil_image(), scene.camera,
                           RansacParams(distance_threshold=0.02, max_planes=6))

    kinds = [p.kind for p in shell.planes]
    print("        found:", ", ".join(f"{p.kind}({p.n_inliers})" for p in shell.planes))

    check("finds a floor", shell.floor is not None)
    check("finds at least two walls", len(shell.walls) >= 2, f"{kinds}")
    check("no plane is misclassified as a ceiling below the camera",
          all(p.centroid[1] > 0 for p in shell.planes if p.kind == "ceiling"))

    if shell.floor is not None:
        truth = scene.ground_truth_floor()
        check("floor normal within 1deg of truth",
              angle_between(shell.floor.normal, truth.normal) < 1.0,
              f"{angle_between(shell.floor.normal, truth.normal):.3f} deg")
        check("floor height within 2cm of truth",
              abs(abs(shell.floor.d) - abs(truth.d)) < 0.02,
              f"|d|={abs(shell.floor.d):.3f} vs {abs(truth.d):.3f}")

    wall_truth = next(p for p in scene.planes if p.name == "back_wall")
    matched = [w for w in shell.walls if angle_between(w.normal, wall_truth.normal) < 5]
    check("recovers the back wall", len(matched) > 0)
    if matched:
        best = max(matched, key=lambda p: p.n_inliers)
        check("back wall distance within 3cm",
              abs(abs(best.d) - abs(wall_truth.d)) < 0.03,
              f"|d|={abs(best.d):.3f} vs {abs(wall_truth.d):.3f}")

    check("builds a shell mesh", shell.mesh is not None)
    if shell.mesh is not None:
        check("shell mesh has faces and UVs",
              len(shell.mesh.faces) > 0 and shell.mesh.visual.uv is not None,
              f"{len(shell.mesh.faces)} faces")

    check("up direction is close to +Y",
          angle_between(shell.up, np.array([0.0, 1.0, 0.0])) < 2.0,
          f"{angle_between(shell.up, np.array([0.0, 1.0, 0.0])):.2f} deg")


def test_shell_under_noise() -> None:
    print("\nshell: fit under 1% depth noise (realistic monocular case)")
    scene = synthetic.default_scene(noise_std=0.01)
    cloud = scene_background_cloud(scene)
    # Noise of 1% at 4m is ~4cm, so the inlier threshold has to open up.
    shell = fit_room_shell(cloud, scene.pil_image(), scene.camera,
                           RansacParams(distance_threshold=0.08, max_planes=6))

    print("        found:", ", ".join(f"{p.kind}({p.n_inliers})" for p in shell.planes))
    check("still finds a floor", shell.floor is not None)
    if shell.floor is not None:
        truth = scene.ground_truth_floor()
        check("floor normal still within 3deg",
              angle_between(shell.floor.normal, truth.normal) < 3.0,
              f"{angle_between(shell.floor.normal, truth.normal):.2f} deg")
        check("floor height still within 8cm",
              abs(abs(shell.floor.d) - abs(truth.d)) < 0.08,
              f"|d|={abs(shell.floor.d):.3f} vs {abs(truth.d):.3f}")
    check("still finds walls", len(shell.walls) >= 1)


def test_objects_are_not_swallowed() -> None:
    print("\nshell: object points must not be absorbed into the shell")
    scene = synthetic.default_scene()
    cloud = scene_background_cloud(scene)
    shell = fit_room_shell(cloud, scene.pil_image(), scene.camera,
                           RansacParams(distance_threshold=0.02))

    if shell.floor is None:
        check("floor available for the occupancy test", False)
        return

    table = next(b for b in scene.boxes if b.name == "table")
    table_pts = backproject_mask(scene.masks["table"], scene.depth, scene.camera,
                                 max_points=None, depth_percentile_trim=None)
    # Every visible table point should sit clearly above the fitted floor.
    heights = shell.floor.signed_distance(table_pts)
    check("table sits above the fitted floor, not in it", heights.min() > 0.02,
          f"min height {heights.min():.3f} m")
    check("table top height is about right",
          abs(heights.max() - 2 * table.half_extents[1]) < 0.05,
          f"{heights.max():.3f} vs {2 * table.half_extents[1]:.3f} m")


def test_plane_polygon() -> None:
    print("\nshell: plane extent polygons")
    scene = synthetic.default_scene()
    pts = backproject_mask(scene.masks["back_wall"], scene.depth, scene.camera,
                           max_points=None, depth_percentile_trim=None)
    plane = fit_plane_ransac(pts, RansacParams(distance_threshold=0.01))
    check("plane fitted for polygon test", plane is not None)
    if plane is None:
        return

    hull = plane_polygon(plane)
    rect = plane_polygon(plane, use_bounding_rect=True)
    check("hull polygon is non-degenerate", hull is not None and len(hull) >= 3,
          f"{len(hull) if hull is not None else 0} verts")
    check("bounding rect is a quad", rect is not None and len(rect) == 4)
    if rect is not None:
        residual = np.abs(plane.signed_distance(rect))
        check("polygon vertices lie on the plane", residual.max() < 1e-6,
              f"max {residual.max():.2e}")


def test_tilted_camera_up_estimate() -> None:
    print("\nshell: up-direction recovery with a tilted camera")
    # Rotate the whole room about X, which is what pointing the camera
    # downward does to the observed geometry.
    tilt = math.radians(12.0)
    rot = np.array([
        [1, 0, 0],
        [0, math.cos(tilt), -math.sin(tilt)],
        [0, math.sin(tilt), math.cos(tilt)],
    ])
    planes, boxes = synthetic.default_room()
    for p in planes:
        p.normal = rot @ p.normal
        p.bounds = None  # rotated bounds are no longer axis-aligned
    for b in boxes:
        b.center = rot @ b.center

    scene = synthetic.render(planes, boxes)
    cloud = scene_background_cloud(scene)
    shell = fit_room_shell(cloud, scene.pil_image(), scene.camera,
                           RansacParams(distance_threshold=0.02))

    expected_up = rot @ np.array([0.0, 1.0, 0.0])
    err = angle_between(shell.up, expected_up)
    check("recovers the tilted up direction within 3deg", err < 3.0,
          f"{err:.2f} deg off (naive +Y would be 12 deg off)")


def main() -> int:
    print("=" * 68)
    print("room-shell self-test  (synthetic ground-truth room, no GPU needed)")
    print("=" * 68)

    test_single_plane_fit()
    test_lstsq_refit_beats_sample()
    test_local_sampling_helps()
    test_full_shell_on_clean_scene()
    test_shell_under_noise()
    test_objects_are_not_swallowed()
    test_plane_polygon()
    test_tilted_camera_up_estimate()

    print("\n" + "=" * 68)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: " + ", ".join(FAILURES))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
