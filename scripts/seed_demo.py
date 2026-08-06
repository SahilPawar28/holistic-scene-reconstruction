"""
Run the Tier 1 pipeline on the synthetic room and write the result straight
into the backend's history, with no GPU and no Colab session involved.

    python scripts/seed_demo.py

This exists because the frontend and backend otherwise cannot be worked on
at all without a live Colab tunnel — and the tunnel dies every time the
notebook idles out. Seeding a real record (real .glb, real depth preview,
real stats) means the viewer can be developed and demoed offline, and it
doubles as an end-to-end smoke test of the local half of the stack.

Pass --image PATH to run against a real photo instead. That needs a depth
map, so it also needs transformers + torch installed locally; the synthetic
path needs neither.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline import synthetic
from pipeline.camera import camera_from_image, read_exif_focal_35mm
from pipeline.meshing import MeshingParams, backproject_mask, depth_to_mesh
from pipeline.room_shell import RansacParams, fit_room_shell
from pipeline.segmentation import background_mask, instances_from_masks

HISTORY_DIR = os.path.join(ROOT, "backend", "history")


def ensure_dirs() -> None:
    for sub in ("images", "models", "depth"):
        os.makedirs(os.path.join(HISTORY_DIR, sub), exist_ok=True)


def depth_preview(depth: np.ndarray) -> Image.Image:
    finite = np.isfinite(depth)
    lo, hi = np.percentile(depth[finite], [2, 98])
    norm = np.clip((depth - lo) / max(hi - lo, 1e-9), 0, 1)
    return Image.fromarray(((1.0 - norm) * 255).astype(np.uint8))


def mask_overlay(image: Image.Image, instances) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    palette = np.array(
        [[255, 96, 96], [96, 200, 255], [140, 255, 140], [255, 210, 90],
         [220, 130, 255], [90, 255, 220]], dtype=np.float32
    )
    for i, inst in enumerate(instances):
        base[inst.mask] = 0.5 * base[inst.mask] + 0.5 * palette[i % len(palette)]
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def append_record(record: dict) -> None:
    path = os.path.join(HISTORY_DIR, "history.json")
    history = []
    if os.path.exists(path):
        try:
            with open(path) as f:
                history = json.load(f)
        except (json.JSONDecodeError, OSError):
            history = []
    history = [r for r in history if r["id"] != record["id"]]
    history.append(record)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def build_synthetic(noise_std: float):
    scene = synthetic.default_scene(noise_std=noise_std)
    image = scene.pil_image()
    instances = instances_from_masks(list(scene.object_masks.values()), scene.depth)
    for inst, name in zip(instances, scene.object_masks.keys()):
        inst.label = name
    return image, scene.depth, scene.camera, instances


def build_from_photo(path: str, hfov_deg: float):
    from pipeline.depth import DepthAnythingV2

    image = Image.open(path).convert("RGB")
    cam = camera_from_image(
        image.width, image.height,
        exif_focal_35mm=read_exif_focal_35mm(path), hfov_deg=hfov_deg,
    )
    print(f"  camera: {cam}")
    print("  running Depth Anything V2 (CPU is slow — a minute or two is normal)…")
    result = DepthAnythingV2().predict(image, max_side=518)
    return image, result.depth, cam, []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="real photo instead of the synthetic room")
    ap.add_argument("--hfov", type=float, default=60.0)
    ap.add_argument("--noise", type=float, default=0.004,
                    help="synthetic depth noise (0 = perfect depth)")
    ap.add_argument("--grid", type=int, default=480)
    args = ap.parse_args()

    ensure_dirs()
    record_id = str(uuid.uuid4())

    if args.image:
        print(f"Building from {args.image}")
        image, depth, cam, instances = build_from_photo(args.image, args.hfov)
        source_name = os.path.basename(args.image)
    else:
        print("Building from the synthetic ground-truth room")
        image, depth, cam, instances = build_synthetic(args.noise)
        source_name = "synthetic_room.png"

    print("  meshing (Tier 1)…")
    tier1 = depth_to_mesh(image, depth, cam, MeshingParams(max_grid_side=args.grid))
    stats = dict(tier1.stats)
    stats["mode"] = "tier1"
    stats["source"] = "synthetic" if not args.image else "photo"
    for key, value in stats.items():
        if key in ("faces", "vertices", "kept_fraction", "culled_depth_jump", "culled_grazing"):
            print(f"    {key}: {value}")

    mode = "tier1"
    if instances:
        # With ground-truth masks available we can also exercise the room
        # shell, which is the Tier 2 stage that does not need a GPU.
        print("  fitting room shell…")
        bg = background_mask(instances, depth.shape)
        cloud = backproject_mask(bg, depth, cam, max_points=60000, depth_percentile_trim=None)
        shell = fit_room_shell(cloud, image, cam, RansacParams(distance_threshold=0.03))
        stats["segmentation"] = {
            "objects": len(instances),
            "instances": [
                {"id": i.id, "area": i.area, "bbox": list(i.bbox),
                 "depth_median": round(float(i.depth_median), 3)}
                for i in instances
            ],
        }
        stats["room_shell"] = shell.stats
        stats["placement_error"] = (
            "assembly stage not implemented yet — this record shows the Tier 1 "
            "mesh with the room shell fitted but not yet composed."
        )
        for p in shell.stats["planes"]:
            print(f"    plane {p['kind']:8s} {p['inliers']:6d} inliers  rms {p['rms_error']}")
        overlay = mask_overlay(image, instances)
        overlay.save(os.path.join(HISTORY_DIR, "depth", f"{record_id}_overlay.png"))

    image_name = f"{record_id}.png"
    image.save(os.path.join(HISTORY_DIR, "images", image_name))
    depth_preview(depth).save(os.path.join(HISTORY_DIR, "depth", f"{record_id}_depth.png"))

    model_path = os.path.join(HISTORY_DIR, "models", f"{record_id}.glb")
    tier1.mesh.export(model_path)
    print(f"  exported {os.path.getsize(model_path) / 1024:.0f} KB glb")

    record = {
        "id": record_id,
        "mode": mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": source_name,
        "image_url": f"/files/images/{image_name}",
        "model_url": f"/files/models/{record_id}.glb",
        "depth_url": f"/files/depth/{record_id}_depth.png",
        "stats": stats,
    }
    if instances:
        record["overlay_url"] = f"/files/depth/{record_id}_overlay.png"
    append_record(record)

    print(f"\nSeeded record {record_id}")
    print("Start the backend and open frontend/index.html:")
    print("  uvicorn main:app --reload --port 8000   (from backend/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
