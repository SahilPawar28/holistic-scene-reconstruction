"""
Ground-truth checks for the placement solver.

Run: python scripts/selftest_assembly.py

The synthetic room's objects are exact axis-aligned boxes at known
positions, so "did the solver put this object in the right place" is a
number, not an opinion. Every check here compares against that truth.

Method: stand in for TripoSR with a densified box mesh of the object's true
shape, canonicalised exactly as `objects.canonicalize_mesh` would
canonicalise a real generated mesh. That isolates the solver — if placement
is wrong here, it is the solver's fault and not the generator's. A perturbed
variant (vertices displaced along their normals) then re-runs the same
checks to confirm the solver tolerates a generated mesh that is the wrong
shape, which every real one is.

Placement error is measured as distance from the placed mesh's vertices to
the *surface* of the true box, which is symmetry-agnostic: a cube rotated
90 degrees about its vertical axis is genuinely the same object, and a test
that compared rotation matrices directly would fail it wrongly.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataclasses import replace

from pipeline import synthetic
from pipeline.assembly import (
    TRIPOSR_TO_SCENE,
    Placement,
    PlacementParams,
    initial_scale_from_mask,
    place_objects,
    rotation_about_axis,
    solve_placement,
    solve_scale_translation,
    solve_similarity_umeyama,
    solve_similarity_yaw,
    visible_vertex_mask,
)
from pipeline.meshing import backproject_mask
from pipeline.objects import canonicalize_mesh, measure_occlusion
from pipeline.room_shell import RansacParams, fit_room_shell
from pipeline.segmentation import background_mask, instances_from_masks

FAILURES: list[str] = []
UP = np.array([0.0, 1.0, 0.0])


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def box_surface_distance(points: np.ndarray, box) -> np.ndarray:
    """|distance| from each point to the surface of an axis-aligned box."""
    q = np.abs(points - box.center) - box.half_extents
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(q.max(axis=1), 0.0)
    return np.abs(outside + inside)


def generated_mesh_for(box, perturb: float = 0.0, seed: int = 0,
                       in_triposr_frame: bool = True):
    """A dense stand-in for what TripoSR would return for this object.

    Canonicalised the same way a real generated mesh is: centred on the
    origin, longest side normalised to 1.

    `in_triposr_frame` (default True) additionally pushes the mesh into
    TripoSR's own output frame — X backward, Y right, Z up — because that
    is what a real generated mesh actually arrives in, and the solver's
    default calibrated mode is built to undo exactly that. Feeding an
    already-world-oriented mesh, as this helper originally did, tests a
    situation that never occurs in the real pipeline. Pass False for the
    tests that specifically exercise the rotation *search* machinery
    (yaw recovery, phase ordering), which needs an unrotated starting point
    to have a known answer.
    """
    import trimesh

    mesh = trimesh.creation.box(extents=2 * box.half_extents)
    canonical, _, _ = canonicalize_mesh(mesh)
    # Densify *after* canonicalising, so the edge length is in canonical
    # units and every object gets the same vertex density regardless of its
    # real size. Subdividing first made the small objects' meshes two orders
    # of magnitude coarser than a real TripoSR output, which is not a fair
    # stand-in: a 200-vertex mesh cannot be registered to anything.
    canonical = canonical.subdivide_to_size(max_edge=0.025)

    if in_triposr_frame:
        canonical.vertices = (
            np.asarray(canonical.vertices, dtype=np.float64) @ TRIPOSR_TO_SCENE
        )

    if perturb > 0:
        rng = np.random.default_rng(seed)
        normals = np.asarray(canonical.vertex_normals, dtype=np.float64)
        offset = rng.normal(0.0, perturb, len(canonical.vertices))[:, None]
        canonical.vertices = (
            np.asarray(canonical.vertices, dtype=np.float64) + normals * offset
        ).astype(np.float32)

    return canonical


def truth_for(box) -> tuple[float, np.ndarray]:
    return float(2 * box.half_extents.max()), np.asarray(box.center, dtype=np.float64)


def scene_setup(noise_std: float = 0.0):
    scene = synthetic.default_scene(noise_std=noise_std)
    names = list(scene.object_masks.keys())
    instances = instances_from_masks(list(scene.object_masks.values()), scene.depth)
    for inst, name in zip(instances, names):
        inst.label = name
    return scene, instances, {inst.label: inst for inst in instances}


def solve_one(scene, instance, box, params=None, perturb=0.0):
    mesh = generated_mesh_for(box, perturb=perturb)
    target = backproject_mask(instance.mask, scene.depth, scene.camera, max_points=20000)
    placement = solve_placement(
        mesh,
        target,
        scene.camera,
        initial_scale=initial_scale_from_mask(instance, scene.depth, scene.camera),
        up=UP,
        params=params or PlacementParams(),
        instance_id=instance.id,
        occlusion=measure_occlusion(instance, scene.depth),
    )
    return mesh, placement


def report(name, box, mesh, placement):
    s_true, t_true = truth_for(box)
    placed = placement.apply(np.asarray(mesh.vertices, dtype=np.float64))
    surf = box_surface_distance(placed, box)
    centre_err = float(np.linalg.norm(placed.mean(axis=0) - t_true))
    scale_err = abs(placement.scale - s_true) / s_true
    p90 = float(np.percentile(surf, 90))
    print(
        f"        {name:6s} scale {placement.scale:.3f} (true {s_true:.3f}, "
        f"{scale_err*100:4.1f}% off)  centre {centre_err*100:5.1f}cm  "
        f"surface p90 {p90*100:4.1f}cm = {p90/s_true*100:4.1f}% of size  "
        f"cov {placement.coverage:.2f}  rms {placement.rms_error*100:.1f}cm"
    )
    return scale_err, centre_err, surf


# --------------------------------------------------------------------------
# 1. closed-form solvers, against known transforms
# --------------------------------------------------------------------------


def test_closed_form_solvers() -> None:
    print("\nsolvers: closed-form transforms recover known ground truth")
    rng = np.random.default_rng(7)
    src = rng.normal(0, 1, (600, 3))

    s_true = 2.37
    r_true = rotation_about_axis(np.array([0.3, 0.9, -0.2]), 0.8)
    t_true = np.array([1.5, -0.4, 3.2])
    dst = src @ (s_true * r_true).T + t_true

    s, r, t = solve_similarity_umeyama(src, dst)
    check("umeyama recovers scale", abs(s - s_true) < 1e-9, f"{s:.6f} vs {s_true}")
    check("umeyama recovers rotation", np.allclose(r, r_true, atol=1e-9),
          f"max err {np.abs(r - r_true).max():.2e}")
    check("umeyama recovers translation", np.allclose(t, t_true, atol=1e-9),
          f"max err {np.abs(t - t_true).max():.2e}")

    # No reflections, ever — a mirrored object is subtly and unfixably wrong.
    flipped = src.copy()
    flipped[:, 0] *= -1
    _, r_flip, _ = solve_similarity_umeyama(src, flipped)
    check("umeyama never returns a reflection", np.linalg.det(r_flip) > 0,
          f"det={np.linalg.det(r_flip):.4f}")

    # Yaw-only solver against a pure yaw transform.
    yaw = 0.7
    r_yaw = rotation_about_axis(UP, yaw)
    dst_yaw = src @ (s_true * r_yaw).T + t_true
    s2, r2, t2 = solve_similarity_yaw(src, dst_yaw, UP)
    check("yaw solver recovers scale", abs(s2 - s_true) < 1e-9, f"{s2:.6f}")
    check("yaw solver recovers rotation", np.allclose(r2, r_yaw, atol=1e-9),
          f"max err {np.abs(r2 - r_yaw).max():.2e}")
    check("yaw solver recovers translation", np.allclose(t2, t_true, atol=1e-9))

    # Yaw solver must stay in the yaw family even given out-of-plane data.
    r_tilt = rotation_about_axis(np.array([1.0, 0.0, 0.0]), 0.5)
    dst_tilt = src @ (s_true * r_tilt).T + t_true
    _, r3, _ = solve_similarity_yaw(src, dst_tilt, UP)
    check("yaw solver keeps up axis fixed", np.allclose(r3 @ UP, UP, atol=1e-9),
          f"up maps to {np.round(r3 @ UP, 4)}")

    # Scale+translation with rotation frozen.
    s4, t4 = solve_scale_translation(src, dst, r_true)
    check("fixed-rotation solver recovers scale", abs(s4 - s_true) < 1e-9, f"{s4:.6f}")
    check("fixed-rotation solver recovers translation", np.allclose(t4, t_true, atol=1e-9))


# --------------------------------------------------------------------------
# 2. visibility
# --------------------------------------------------------------------------


def test_visibility() -> None:
    print("\nvisibility: only the camera-facing, unoccluded, in-frame side")
    import trimesh

    scene, instances, by_name = scene_setup()
    cam = scene.camera

    sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.4)
    sphere.apply_translation([0.0, 0.0, -3.0])
    verts = np.asarray(sphere.vertices, dtype=np.float64)
    normals = np.asarray(sphere.vertex_normals, dtype=np.float64)

    vis = visible_vertex_mask(verts, normals, cam, PlacementParams())
    frac = vis.mean()
    check("about half a sphere is visible", 0.35 < frac < 0.62, f"{frac:.1%}")
    check("visible vertices are the near ones",
          verts[vis][:, 2].mean() > verts[~vis][:, 2].mean(),
          f"near z {verts[vis][:, 2].mean():.3f} vs far {verts[~vis][:, 2].mean():.3f}")

    # Out-of-frame vertices must be excluded — this was a real bug: without
    # it the solver shrinks any object that runs off the edge of the photo.
    off = verts + np.array([6.0, 0.0, 0.0])
    vis_off = visible_vertex_mask(off, normals, cam, PlacementParams())
    check("vertices outside the frame are not visible", vis_off.sum() == 0,
          f"{vis_off.sum()} marked visible")

    behind = verts + np.array([0.0, 0.0, 4.0])  # now behind the camera
    check("vertices behind the camera are not visible",
          visible_vertex_mask(behind, normals, cam, PlacementParams()).sum() == 0)


# --------------------------------------------------------------------------
# 3. placement against ground truth
# --------------------------------------------------------------------------


def test_placement_exact_mesh() -> None:
    print("\nplacement: exact object shape, perfect depth")
    scene, instances, by_name = scene_setup()
    boxes = {b.name: b for b in scene.boxes}

    for name in ("table", "pot", "crate"):
        mesh, placement = solve_one(scene, by_name[name], boxes[name])
        scale_err, centre_err, surf = report(name, boxes[name], mesh, placement)

        check(f"{name}: placement succeeded", placement.ok, placement.status)
        check(f"{name}: scale within 15%", scale_err < 0.15, f"{scale_err*100:.1f}%")
        check(f"{name}: centre within 10cm", centre_err < 0.10, f"{centre_err*100:.1f}cm")
        # Relative to the object's own size, not an absolute distance: 4cm
        # is a close fit for a table and a total miss for a mug, and a flat
        # centimetre threshold silently tests the big objects hardest.
        s_true, _ = truth_for(boxes[name])
        relative = float(np.percentile(surf, 90)) / s_true
        check(f"{name}: mesh lies on the true surface (p90 < 15% of size)",
              relative < 0.15, f"p90 = {relative*100:.1f}% of size")


def test_placement_perturbed_mesh() -> None:
    print("\nplacement: imperfect generated shape (simulating TripoSR error)")
    scene, instances, by_name = scene_setup()
    boxes = {b.name: b for b in scene.boxes}

    for name in ("table", "pot", "crate"):
        mesh, placement = solve_one(scene, by_name[name], boxes[name], perturb=0.02)
        scale_err, centre_err, surf = report(name, boxes[name], mesh, placement)
        check(f"{name}: scale still within 20%", scale_err < 0.20, f"{scale_err*100:.1f}%")
        check(f"{name}: centre still within 12cm", centre_err < 0.12,
              f"{centre_err*100:.1f}cm")


def test_placement_noisy_depth() -> None:
    print("\nplacement: 1% depth noise (realistic monocular case)")
    scene, instances, by_name = scene_setup(noise_std=0.01)
    boxes = {b.name: b for b in scene.boxes}

    for name in ("table", "pot", "crate"):
        mesh, placement = solve_one(scene, by_name[name], boxes[name], perturb=0.015)
        scale_err, centre_err, surf = report(name, boxes[name], mesh, placement)
        check(f"{name}: scale within 25% under noise", scale_err < 0.25,
              f"{scale_err*100:.1f}%")
        check(f"{name}: centre within 15cm under noise", centre_err < 0.15,
              f"{centre_err*100:.1f}cm")


# --------------------------------------------------------------------------
# 4. the design decisions, tested as ablations
# --------------------------------------------------------------------------


def test_visibility_ablation() -> None:
    print("\nablation: what visibility filtering is actually worth")
    scene, instances, by_name = scene_setup()
    boxes = {b.name: b for b in scene.boxes}

    worse = 0
    for name in ("table", "pot", "crate"):
        mesh, with_vis = solve_one(scene, by_name[name], boxes[name])
        _, without = solve_one(scene, by_name[name], boxes[name],
                               params=PlacementParams(use_visibility=False))
        s_true, _ = truth_for(boxes[name])
        e_with = abs(with_vis.scale - s_true) / s_true
        e_without = abs(without.scale - s_true) / s_true
        placed_without = without.apply(np.asarray(mesh.vertices, dtype=np.float64))
        d_without = np.percentile(box_surface_distance(placed_without, boxes[name]), 90)
        placed_with = with_vis.apply(np.asarray(mesh.vertices, dtype=np.float64))
        d_with = np.percentile(box_surface_distance(placed_with, boxes[name]), 90)
        print(f"        {name:6s} scale err {e_with*100:5.1f}% -> {e_without*100:6.1f}% "
              f"| surface p90 {d_with*100:4.1f}cm -> {d_without*100:6.1f}cm  "
              f"(with -> without visibility)")
        if e_without > e_with or d_without > d_with:
            worse += 1

    check("disabling visibility filtering degrades every object", worse == 3,
          f"{worse}/3 got worse")


def test_phase_order() -> None:
    print("\nablation: solving rotation first vs the recommended order")
    scene, instances, by_name = scene_setup()
    boxes = {b.name: b for b in scene.boxes}

    # Judged across every object, not one. A single object's scale error is
    # noisy enough that either ordering can win on it by a point or two,
    # which says nothing about the ordering itself — the claim being tested
    # ("freezing rotation first is more stable") is a claim about the
    # method, so it has to be measured over the whole set.
    #
    # Both arms use the search mode ("upright") rather than the calibrated
    # default, which has no phase-2 rotation solve whose ordering could
    # differ.
    staged_errs, first_errs = [], []
    for name in ("table", "crate", "pot"):
        box = boxes[name]
        s_true, _ = truth_for(box)
        _, staged = solve_one(scene, by_name[name], box,
                              params=PlacementParams(rotation_mode="upright"))
        _, rotation_first = solve_one(
            scene, by_name[name], box,
            params=PlacementParams(rotation_mode="upright",
                                   phase1_iterations=0, phase2_iterations=32),
        )
        e_s = abs(staged.scale - s_true) / s_true
        e_f = abs(rotation_first.scale - s_true) / s_true
        staged_errs.append(e_s)
        first_errs.append(e_f)
        print(f"        {name:6s} staged {e_s*100:5.1f}%   rotation-first {e_f*100:5.1f}%")

    mean_staged = float(np.mean(staged_errs))
    mean_first = float(np.mean(first_errs))
    print(f"        mean:  staged {mean_staged*100:.1f}%   "
          f"rotation-first {mean_first*100:.1f}%")
    check("staged order is no worse on average than rotating from the start",
          mean_staged <= mean_first + 0.02,
          f"{mean_staged*100:.1f}% vs {mean_first*100:.1f}%")


def test_yaw_recovery() -> None:
    print("\nplacement: recovering a known yaw offset")
    scene, instances, by_name = scene_setup()
    box = next(b for b in scene.boxes if b.name == "table")  # non-square footprint

    import trimesh

    for degrees in (25.0, -40.0):
        # Search-machinery test: needs a world-frame mesh so the injected
        # yaw is the only rotation present and the answer is known.
        mesh = generated_mesh_for(box, in_triposr_frame=False)
        pre = rotation_about_axis(UP, math.radians(degrees))
        spun = mesh.copy()
        spun.vertices = np.asarray(mesh.vertices, dtype=np.float64) @ pre.T

        target = backproject_mask(by_name["table"].mask, scene.depth, scene.camera,
                                  max_points=20000)
        placement = solve_placement(
            spun, target, scene.camera,
            initial_scale=initial_scale_from_mask(by_name["table"], scene.depth, scene.camera),
            up=UP, params=PlacementParams(rotation_mode="upright"), instance_id=0,
        )
        # The solver should undo the pre-rotation: R_solved @ pre ~ identity,
        # up to the box's 180-degree symmetry.
        composed = placement.rotation @ pre
        angle = math.degrees(math.acos(np.clip((np.trace(composed) - 1) / 2, -1, 1)))
        angle = min(angle, abs(180 - angle))
        placed = placement.apply(np.asarray(spun.vertices, dtype=np.float64))
        surf_p90 = float(np.percentile(box_surface_distance(placed, box), 90))
        print(f"        pre-rotated {degrees:+.0f}deg -> residual {angle:5.1f}deg, "
              f"surface p90 {surf_p90*100:.1f}cm")
        check(f"yaw {degrees:+.0f}deg recovered within 15deg", angle < 15.0,
              f"{angle:.1f}deg residual")


# --------------------------------------------------------------------------
# 5. support snapping and full composition
# --------------------------------------------------------------------------


def test_support_snapping() -> None:
    print("\nsnapping: objects rest on what they actually stand on")
    scene, instances, by_name = scene_setup()
    boxes = {b.name: b for b in scene.boxes}

    bg = background_mask(instances, scene.depth.shape)
    cloud = backproject_mask(bg, scene.depth, scene.camera, max_points=60000,
                             depth_percentile_trim=None)
    shell = fit_room_shell(cloud, scene.pil_image(), scene.camera,
                           RansacParams(distance_threshold=0.02))
    check("shell has a floor to snap against", shell.floor is not None)

    generated = []
    for inst in instances:
        mesh = generated_mesh_for(boxes[inst.label])
        generated.append(
            type("G", (), {"instance_id": inst.id, "mesh": mesh,
                           "crop": type("C", (), {"occlusion": 0.0})()})()
        )

    placements = place_objects(generated, instances, scene.depth, scene.camera, shell)
    by_id = {p.instance_id: p for p in placements}
    labels = {inst.id: inst.label for inst in instances}

    for p in placements:
        print(f"        {labels[p.instance_id]:6s} support={p.support:12s} "
              f"snap={p.snap_offset*100:+5.1f}cm  scale={p.scale:.3f}  "
              f"status={p.status}")

    pot_id = next(i.id for i in instances if i.label == "pot")
    table_id = next(i.id for i in instances if i.label == "table")
    crate_id = next(i.id for i in instances if i.label == "crate")

    check("every object placed", all(p.ok for p in placements),
          str([p.status for p in placements]))
    check("crate stands on the floor", by_id[crate_id].support == "floor",
          by_id[crate_id].support)
    check("table stands on the floor", by_id[table_id].support == "floor",
          by_id[table_id].support)
    check("pot stands on the table, not the floor",
          by_id[pot_id].support == f"object:{table_id}",
          f"got '{by_id[pot_id].support}'")

    # Ground-truth heights after snapping.
    for name, ident in (("table", table_id), ("crate", crate_id), ("pot", pot_id)):
        box = boxes[name]
        mesh = generated_mesh_for(box)
        placed = by_id[ident].apply(np.asarray(mesh.vertices, dtype=np.float64))
        base = float(shell.floor.signed_distance(placed).min())
        true_base = float(box.center[1] - box.half_extents[1] + 1.4)  # floor at y=-1.4
        check(f"{name}: base height within 8cm of truth", abs(base - true_base) < 0.08,
              f"{base:.3f}m vs {true_base:.3f}m")


def test_compose_and_export() -> None:
    print("\ncompose: scene graph and .glb export")
    import tempfile

    import trimesh

    from pipeline.scene_compose import compose_scene, export_glb, scene_statistics

    scene, instances, by_name = scene_setup()
    boxes = {b.name: b for b in scene.boxes}

    bg = background_mask(instances, scene.depth.shape)
    cloud = backproject_mask(bg, scene.depth, scene.camera, max_points=60000,
                             depth_percentile_trim=None)
    shell = fit_room_shell(cloud, scene.pil_image(), scene.camera,
                           RansacParams(distance_threshold=0.02))

    generated = []
    for inst in instances:
        mesh = generated_mesh_for(boxes[inst.label])
        generated.append(
            type("G", (), {"instance_id": inst.id, "mesh": mesh,
                           "crop": type("C", (), {"occlusion": 0.0})()})()
        )
    placements = place_objects(generated, instances, scene.depth, scene.camera, shell)

    composed = compose_scene(shell, generated, placements)
    stats = scene_statistics(composed, placements)
    print(f"        nodes={stats['nodes']} faces={stats['faces']:,} "
          f"extents={stats['extents_m']}")
    print(f"        names: {stats['names']}")

    check("scene has the shell plus every object", stats["nodes"] == 1 + len(instances),
          f"{stats['nodes']} nodes")
    check("objects keep their own identity in the graph",
          all(f"object_{i.id}" in stats["names"] for i in instances))
    check("room shell is present", "room_shell" in stats["names"])
    check("scene sits in front of the camera", stats["bounds_max"][2] < 0,
          f"max z {stats['bounds_max'][2]}")
    check("scene is room-sized", 2.0 < stats["extents_m"][0] < 8.0,
          f"width {stats['extents_m'][0]}m")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "scene.glb")
        export_glb(composed, path)
        size = os.path.getsize(path)
        check("exports a non-trivial .glb", size > 50_000, f"{size/1024:.0f} KB")

        reloaded = trimesh.load(path)
        check("reloads as a multi-geometry scene",
              isinstance(reloaded, trimesh.Scene) and len(reloaded.geometry) == stats["nodes"],
              f"{len(reloaded.geometry) if isinstance(reloaded, trimesh.Scene) else 1} geometries")

        # Transforms must survive the round-trip, or objects reload at the
        # origin — i.e. inside the camera.
        rt_bounds = reloaded.bounds
        check("transforms survive the glb round-trip",
              np.allclose(rt_bounds, composed.bounds, atol=1e-3),
              f"max delta {np.abs(rt_bounds - composed.bounds).max():.4f}")


def test_failure_paths() -> None:
    print("\nrobustness: degenerate inputs are handled, not crashed on")
    scene, instances, by_name = scene_setup()
    boxes = {b.name: b for b in scene.boxes}

    p = solve_placement(None, np.zeros((100, 3)), scene.camera, 1.0,
                        params=PlacementParams(), instance_id=0)
    check("missing mesh -> failed placement, no exception", p.status == "failed:no-mesh")

    mesh = generated_mesh_for(boxes["pot"])
    p2 = solve_placement(mesh, np.zeros((5, 3)), scene.camera, 1.0,
                         params=PlacementParams(), instance_id=1)
    check("too few target points -> failed placement",
          p2.status == "failed:too-few-target-points")

    # A placement that failed must not be exported into the scene, where it
    # would sit at the origin — which is the camera position.
    from pipeline.scene_compose import compose_scene

    generated = [type("G", (), {"instance_id": 0, "mesh": mesh,
                                "crop": type("C", (), {"occlusion": 0.0})()})()]
    failed = [Placement(instance_id=0, status="failed:no-convergence")]
    try:
        compose_scene(None, generated, failed)
        check("composing only failed placements raises rather than exporting junk", False)
    except RuntimeError:
        check("composing only failed placements raises rather than exporting junk", True)


def main() -> int:
    print("=" * 74)
    print("assembly self-test  (synthetic ground-truth room, no GPU needed)")
    print("=" * 74)

    test_closed_form_solvers()
    test_visibility()
    test_placement_exact_mesh()
    test_placement_perturbed_mesh()
    test_placement_noisy_depth()
    test_visibility_ablation()
    test_phase_order()
    test_yaw_recovery()
    test_support_snapping()
    test_compose_and_export()
    test_failure_paths()

    print("\n" + "=" * 74)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
