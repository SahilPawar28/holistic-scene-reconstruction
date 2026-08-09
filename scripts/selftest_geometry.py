"""
Ground-truth checks for the geometry layer. No GPU, no pretrained models.

Run: python scripts/selftest_geometry.py

Everything here is checked against a number that is known exactly, because
the synthetic room in pipeline/synthetic.py is ray-traced from analytic
planes and boxes. If these pass, any error in the full pipeline is coming
from the neural components or from the stages above, not from the projection
math underneath.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.camera import PinholeCamera
from pipeline.depth import disparity_to_depth
from pipeline.meshing import MeshingParams, backproject_mask, depth_to_mesh
from pipeline import synthetic

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_camera_roundtrip() -> None:
    print("\ncamera: project/unproject round-trip")
    cam = PinholeCamera.from_fov(640, 480, 60.0)

    check("fx from 60deg hfov", abs(cam.fx - 554.256) < 0.01, f"fx={cam.fx:.3f}")
    check("hfov round-trips", abs(cam.hfov_deg - 60.0) < 1e-9)
    check("vfov < hfov for 4:3", cam.vfov_deg < cam.hfov_deg,
          f"vfov={cam.vfov_deg:.2f}")

    rng = np.random.default_rng(0)
    u = rng.uniform(0, 640, 500)
    v = rng.uniform(0, 480, 500)
    d = rng.uniform(0.5, 10.0, 500)
    pts = cam.unproject(u, v, d)
    uv2, d2 = cam.project(pts)

    check("unproject -> project recovers pixels",
          np.allclose(uv2[:, 0], u) and np.allclose(uv2[:, 1], v),
          f"max err {np.abs(uv2 - np.stack([u, v], 1)).max():.2e} px")
    check("unproject -> project recovers depth", np.allclose(d2, d),
          f"max err {np.abs(d2 - d).max():.2e}")

    # Sign conventions: get one of these backwards and the whole scene is
    # mirrored or inside out, which is easy to miss visually.
    right = cam.unproject(cam.cx + 100, cam.cy, 2.0)
    up = cam.unproject(cam.cx, cam.cy - 100, 2.0)
    check("pixel right of centre -> +X", right[0] > 0, f"x={right[0]:.3f}")
    check("pixel above centre -> +Y", up[1] > 0, f"y={up[1]:.3f}")
    check("scene is in front of camera (-Z)", right[2] < 0, f"z={right[2]:.3f}")

    # Intrinsics must scale with resolution or objects come out mis-sized.
    half = cam.scaled(320, 240)
    p_full = cam.unproject(640 - 1, 0, 3.0)
    p_half = half.unproject((640 - 1) / 2.0, 0.0, 3.0)
    check("scaled() preserves rays", np.allclose(p_full, p_half, atol=1e-6),
          f"delta {np.abs(p_full - p_half).max():.2e}")


def test_disparity_conversion() -> None:
    print("\ndepth: relative disparity -> depth")
    # Ground truth: depths spanning 1..8m, turned into disparity the way a
    # relative depth model would emit it (proportional to 1/depth).
    true_depth = np.linspace(1.0, 8.0, 400).astype(np.float32)
    disparity = 1.0 / true_depth

    recovered = disparity_to_depth(disparity, near=1.0, far=8.0,
                                   percentile_clip=(0.0, 100.0))
    rel_err = np.abs(recovered - true_depth) / true_depth
    check("recovers depth from ideal disparity", rel_err.max() < 0.01,
          f"max rel err {rel_err.max():.2%}")

    # The ordering property is the one that must hold even when the affine
    # scale is wrong: nearest pixel stays nearest.
    noisy = disparity * 3.7 + 0.9  # arbitrary affine, as models actually emit
    rec2 = disparity_to_depth(noisy, near=1.0, far=8.0, percentile_clip=(0.0, 100.0))
    check("affine-invariant (scale+shift on disparity)",
          np.allclose(rec2, recovered, rtol=1e-4),
          f"max rel diff {np.abs(rec2 / recovered - 1).max():.2e}")
    check("monotonic: higher disparity -> nearer", np.all(np.diff(rec2) > 0))


def test_backprojection_against_truth() -> None:
    print("\nmeshing: back-projection lands on the true surfaces")
    scene = synthetic.default_scene()
    cam = scene.camera

    floor = scene.ground_truth_floor()
    pts = backproject_mask(scene.masks["floor"], scene.depth, cam,
                           max_points=None, depth_percentile_trim=None)
    residual = np.abs(pts @ floor.normal + floor.d)
    check("floor pixels land on the floor plane", residual.max() < 1e-3,
          f"max |residual| {residual.max():.2e} m over {len(pts)} pts")

    wall = next(p for p in scene.planes if p.name == "back_wall")
    wpts = backproject_mask(scene.masks["back_wall"], scene.depth, cam,
                            max_points=None, depth_percentile_trim=None)
    wres = np.abs(wpts @ wall.normal + wall.d)
    check("back-wall pixels land on the wall plane", wres.max() < 1e-3,
          f"max |residual| {wres.max():.2e} m")

    # An object's back-projected cloud is what the assembly solver aligns
    # against, so its extent has to match the real object.
    pot = next(b for b in scene.boxes if b.name == "pot")
    ppts = backproject_mask(scene.masks["pot"], scene.depth, cam,
                            max_points=None, depth_percentile_trim=None)
    lo, hi = ppts.min(axis=0), ppts.max(axis=0)
    # Only the visible faces are seen, so width/height should match but the
    # depth extent is at most the true one.
    check("pot cloud width matches truth",
          abs((hi[0] - lo[0]) - 2 * pot.half_extents[0]) < 0.03,
          f"got {hi[0] - lo[0]:.3f} vs {2 * pot.half_extents[0]:.3f} m")
    check("pot cloud height matches truth",
          abs((hi[1] - lo[1]) - 2 * pot.half_extents[1]) < 0.03,
          f"got {hi[1] - lo[1]:.3f} vs {2 * pot.half_extents[1]:.3f} m")
    check("pot cloud sits at the true distance",
          abs(np.median(ppts[:, 2]) - (pot.center[2] + pot.half_extents[2])) < 0.05,
          f"median z {np.median(ppts[:, 2]):.3f}")

    # Depth trimming must not eat the object.
    trimmed = backproject_mask(scene.masks["pot"], scene.depth, cam)
    check("percentile trim keeps most of the cloud",
          len(trimmed) > 0.9 * len(ppts),
          f"{len(trimmed)}/{len(ppts)} kept")


def test_tier1_mesh() -> None:
    print("\nmeshing: Tier 1 mesh construction and edge-aware culling")
    scene = synthetic.default_scene()
    img = scene.pil_image()

    loose = depth_to_mesh(img, scene.depth, scene.camera,
                          MeshingParams(max_relative_depth_jump=1e9,
                                        max_grazing_angle_deg=89.999,
                                        smooth_depth=False))
    tight = depth_to_mesh(img, scene.depth, scene.camera,
                          MeshingParams(smooth_depth=False))

    check("mesh has geometry", len(tight.mesh.faces) > 10000,
          f"{len(tight.mesh.faces)} faces")
    check("culling removes the stretched triangles",
          len(tight.mesh.faces) < len(loose.mesh.faces),
          f"{len(loose.mesh.faces)} -> {len(tight.mesh.faces)} faces")
    check("culling is not overzealous", tight.stats["kept_fraction"] > 0.85,
          f"kept {tight.stats['kept_fraction']:.1%}")
    check("uv coords present and in range",
          tight.mesh.visual.uv is not None
          and tight.mesh.visual.uv.min() >= 0
          and tight.mesh.visual.uv.max() <= 1)
    check("no unreferenced vertices",
          len(np.unique(tight.mesh.faces)) == len(tight.mesh.vertices),
          f"{len(tight.mesh.vertices)} verts")
    check("mesh is entirely in front of the camera",
          tight.mesh.vertices[:, 2].max() < 0,
          f"max z {tight.mesh.vertices[:, 2].max():.3f}")

    # The decisive test: without culling, triangles bridge the gap between
    # the pot and the wall 2m behind it. Measure the longest edge.
    def longest_edge(mesh):
        v = mesh.vertices[mesh.faces]
        e = np.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 1], v[:, 0] - v[:, 2]], 1)
        return np.linalg.norm(e, axis=2).max()

    check("culling shortens the worst stretched triangle",
          longest_edge(tight.mesh) < 0.5 * longest_edge(loose.mesh),
          f"{longest_edge(loose.mesh):.3f} m -> {longest_edge(tight.mesh):.3f} m")

    # Winding: front faces must point back toward the camera.
    normals = tight.mesh.face_normals
    centroids = tight.mesh.triangles.mean(axis=1)
    facing = np.einsum("ij,ij->i", normals, -centroids)
    check("faces are wound toward the camera", (facing > 0).mean() > 0.98,
          f"{(facing > 0).mean():.1%} front-facing")


def test_tier1_survives_noise() -> None:
    print("\nmeshing: Tier 1 on noisy depth (realistic monocular case)")
    scene = synthetic.default_scene(noise_std=0.01)
    result = depth_to_mesh(scene.pil_image(), scene.depth, scene.camera,
                           MeshingParams())
    check("still produces a usable mesh under 1% depth noise",
          result.stats["kept_fraction"] > 0.6,
          f"kept {result.stats['kept_fraction']:.1%}, "
          f"{result.stats['faces']} faces")


def test_inpaint_leaves_no_holes() -> None:
    print("\ninpaint: filled regions must not be re-culled into black patches")
    from pipeline.inpaint import inpaint_depth
    from pipeline.segmentation import instances_from_masks, occupancy_mask

    scene = synthetic.default_scene()
    instances = instances_from_masks(list(scene.object_masks.values()), scene.depth)
    occupied = occupancy_mask(instances, scene.depth.shape)

    for label, smooth in (("nearest-neighbour", False), ("harmonic", True)):
        filled = inpaint_depth(scene.depth.astype(np.float32), occupied, smooth=smooth)
        step = np.abs(np.diff(filled, axis=1))
        interior = occupied[:, :-1] & occupied[:, 1:]
        relative = step[interior] / np.maximum(filled[:, :-1][interior], 1e-6)
        result = depth_to_mesh(scene.pil_image(), filled, scene.camera, MeshingParams())
        print(f"        {label:18s} max interior step {relative.max():.4f}  "
              f"culled_jump {result.stats['culled_depth_jump']:,}")
        if smooth:
            # The whole point: a filled region with interior steps above the
            # cull threshold gets punched back out into a star-shaped hole,
            # which is what the black patches were.
            check("harmonic fill leaves no step above the cull threshold",
                  relative.max() < MeshingParams().max_relative_depth_jump,
                  f"max step {relative.max():.4f} vs threshold "
                  f"{MeshingParams().max_relative_depth_jump}")
            check("harmonic fill causes no depth-jump culling",
                  result.stats["culled_depth_jump"] == 0,
                  f"{result.stats['culled_depth_jump']} triangles culled")


def test_glb_export() -> None:
    print("\nmeshing: .glb export")
    import tempfile

    scene = synthetic.default_scene(width=320, height=240)
    result = depth_to_mesh(scene.pil_image(), scene.depth, scene.camera)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tier1.glb")
        result.mesh.export(path)
        size = os.path.getsize(path)
        check("exports a non-trivial .glb", size > 20_000, f"{size / 1024:.0f} KB")

        import trimesh

        reloaded = trimesh.load(path)
        geoms = (list(reloaded.geometry.values())
                 if isinstance(reloaded, trimesh.Scene) else [reloaded])
        check("reloads with the same face count",
              sum(len(g.faces) for g in geoms) == len(result.mesh.faces))
        check("texture survives the round-trip",
              any(getattr(g.visual, "uv", None) is not None for g in geoms))


def main() -> int:
    print("=" * 68)
    print("geometry self-test  (synthetic ground-truth room, no GPU needed)")
    print("=" * 68)

    test_camera_roundtrip()
    test_disparity_conversion()
    test_backprojection_against_truth()
    test_tier1_mesh()
    test_tier1_survives_noise()
    test_inpaint_leaves_no_holes()
    test_glb_export()

    print("\n" + "=" * 68)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: " + ", ".join(FAILURES))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
