"""
Assemble the room shell and the placed objects into one exportable scene.

Deliberately a *scene graph* rather than one merged mesh. Baking every
object's transform into its vertices and concatenating would produce a
smaller file, but it would also throw away the thing that makes Tier 2
different from Tier 1: the fact that this output knows it contains a floor,
three walls and four distinct objects, each with its own solved transform.
Keeping them as named nodes means the .glb can be opened in Blender and
picked apart, individual objects can be hidden or moved, and the placement
solver's output is inspectable rather than fused into an anonymous blob.
"""

from __future__ import annotations

import numpy as np

from .assembly import Placement
from .room_shell import RoomShell


def compose_scene(
    shell: RoomShell | None,
    generated: list,
    placements: list[Placement],
    include_shell: bool = True,
    include_failed: bool = False,
):
    """Room shell + placed object meshes -> one `trimesh.Scene`.

    Objects whose generation or placement failed are skipped by default —
    an unplaced mesh would land at the origin, which is the camera position,
    and a mesh wrapped around the camera makes the whole scene unviewable.
    """
    import trimesh

    scene = trimesh.Scene()
    meshes = {g.instance_id: g.mesh for g in generated}
    placed = 0

    if include_shell and shell is not None and shell.mesh is not None:
        scene.add_geometry(shell.mesh, node_name="room_shell", geom_name="room_shell")

    for placement in placements:
        mesh = meshes.get(placement.instance_id)
        if mesh is None:
            continue
        if not placement.ok and not include_failed:
            continue

        name = f"object_{placement.instance_id}"
        scene.add_geometry(
            mesh, node_name=name, geom_name=name, transform=placement.matrix
        )
        placed += 1

    if placed == 0 and (shell is None or shell.mesh is None):
        raise RuntimeError(
            "nothing to compose: no room shell and no successfully placed objects"
        )

    return scene


def scene_statistics(scene, placements: list[Placement] | None = None) -> dict:
    """Summary of what actually made it into the export."""
    geometries = list(scene.geometry.items())
    total_faces = int(sum(len(g.faces) for _, g in geometries))
    bounds = scene.bounds

    stats = {
        "nodes": len(geometries),
        "faces": total_faces,
        "names": [name for name, _ in geometries],
    }
    if bounds is not None:
        extents = bounds[1] - bounds[0]
        stats["extents_m"] = [round(float(v), 3) for v in extents]
        stats["bounds_min"] = [round(float(v), 3) for v in bounds[0]]
        stats["bounds_max"] = [round(float(v), 3) for v in bounds[1]]

    if placements is not None:
        stats["placed"] = sum(1 for p in placements if p.ok)
        stats["failed"] = sum(1 for p in placements if not p.ok)
    return stats


def export_glb(scene, path: str | None = None):
    """Export to .glb, returning bytes when no path is given."""
    data = scene.export(file_type="glb")
    if path is None:
        return data
    with open(path, "wb") as handle:
        handle.write(data)
    return path


def placement_debug_scene(
    shell: RoomShell | None,
    generated: list,
    placements: list[Placement],
    target_clouds: dict[int, np.ndarray] | None = None,
):
    """A diagnostic variant that also shows what each object was fitted to.

    Adds every object's target point cloud alongside its placed mesh. When a
    placement looks wrong in the viewer, the immediate question is whether
    the solver missed the target or the target itself was wrong (a leaking
    mask, a bad depth patch), and seeing both together answers it in one
    look instead of a debugging session.
    """
    import trimesh

    scene = compose_scene(shell, generated, placements, include_failed=True)
    if not target_clouds:
        return scene

    palette = [
        [255, 90, 90], [90, 200, 255], [140, 255, 140], [255, 210, 90],
        [220, 130, 255], [90, 255, 220], [255, 150, 200], [190, 190, 255],
    ]
    for i, (instance_id, cloud) in enumerate(sorted(target_clouds.items())):
        if cloud is None or len(cloud) == 0:
            continue
        colour = palette[i % len(palette)] + [255]
        points = trimesh.PointCloud(
            np.asarray(cloud, dtype=np.float64),
            colors=np.tile(colour, (len(cloud), 1)),
        )
        scene.add_geometry(points, node_name=f"target_{instance_id}")

    return scene
