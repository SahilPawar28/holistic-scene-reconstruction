"""
Where do the holes in a Tier 1 mesh come from, and how big are they really?

    python scripts/analyze_holes.py --image photo.jpg

Orbit a Tier 1 reconstruction even slightly and black gaps open up around
every foreground object. This script answers whether those gaps are
over-culling (a tuning problem, fixable) or occlusion (a property of
single-image geometry, not fixable at Tier 1).

It does three things:

  1. Sweeps the depth-jump threshold and reports how many triangles each
     value keeps — if the holes were over-culling, loosening the threshold
     would close them.
  2. Writes a hole map: which pixels of the original image lost their
     geometry, so the culled region can be compared against the photo.
  3. Measures how wide those gaps become as the camera rotates away from
     the original viewpoint, which is the number that actually explains
     why 4% of triangles looks like a third of the frame.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.camera import camera_from_image, read_exif_focal_35mm
from pipeline.meshing import (
    MeshingParams,
    build_face_grid,
    depth_jump_mask,
    depth_to_mesh,
    grazing_mask,
    _resize_to_grid,
)


def load_depth(path: str, hfov: float):
    from pipeline.depth import DepthAnythingV2

    image = Image.open(path).convert("RGB")
    cam = camera_from_image(
        image.width, image.height,
        exif_focal_35mm=read_exif_focal_35mm(path), hfov_deg=hfov,
    )
    print(f"  {cam}")
    result = DepthAnythingV2().predict(image, max_side=518)
    return image, result.depth, cam


def hole_map(image: Image.Image, depth: np.ndarray, cam, params: MeshingParams):
    """Per-pixel: does this pixel still have geometry attached to it?"""
    from pipeline.depth import smooth_depth_edge_preserving

    d_full = smooth_depth_edge_preserving(depth) if params.smooth_depth else depth
    img, grid_depth = _resize_to_grid(image, d_full, params.max_grid_side)
    w, h = img.size
    grid_cam = cam.scaled(w, h)

    d = grid_depth.astype(np.float64)
    bad = ~np.isfinite(d) | (d < params.min_depth) | (d > params.max_depth)
    d = np.where(bad, np.nan, d)

    pts = grid_cam.unproject_depth_map(d).reshape(-1, 3)
    dflat = d.reshape(-1)
    faces = build_face_grid(h, w)

    keep = np.isfinite(dflat)[faces].all(axis=1)
    keep_jump = keep & depth_jump_mask(dflat, faces, params.max_relative_depth_jump)
    keep_all = keep_jump & grazing_mask(pts, faces, params.max_grazing_angle_deg)

    covered = np.zeros(h * w, dtype=bool)
    covered[faces[keep_all].ravel()] = True
    return covered.reshape(h, w), img, (h, w), int(keep_all.sum()), len(faces)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--hfov", type=float, default=60.0)
    ap.add_argument("--out", default="assets/diagnostics")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0][:24]

    print(f"Analyzing {args.image}")
    image, depth, cam = load_depth(args.image, args.hfov)

    # --- 1. threshold sweep -------------------------------------------
    print("\n  depth-jump threshold sweep (default 0.06):")
    print(f"    {'threshold':>10}  {'faces kept':>12}  {'kept %':>8}")
    for thr in (0.02, 0.04, 0.06, 0.10, 0.20, 0.50, 1e9):
        r = depth_to_mesh(image, depth, cam,
                          MeshingParams(max_relative_depth_jump=thr))
        label = "no cull" if thr > 1e8 else f"{thr:.2f}"
        print(f"    {label:>10}  {r.stats['faces']:>12,}  {r.stats['kept_fraction']*100:>7.1f}%")

    # --- 2. hole map ---------------------------------------------------
    params = MeshingParams()
    covered, grid_img, (h, w), kept, total = hole_map(image, depth, cam, params)
    hole_frac = 1.0 - covered.mean()
    print(f"\n  pixels with no geometry: {hole_frac*100:.1f}% "
          f"({(~covered).sum():,} of {covered.size:,})")

    base = np.asarray(grid_img, dtype=np.float32)
    tinted = base.copy()
    tinted[~covered] = [255, 40, 40]
    Image.fromarray(np.clip(tinted, 0, 255).astype(np.uint8)).save(
        os.path.join(args.out, f"{stem}_holes.png")
    )
    print(f"  wrote {args.out}/{stem}_holes.png  (red = culled)")

    # --- 3. how wide do those gaps get when you orbit? ------------------
    # A culled triangle spans a depth jump over ~1 pixel. Seen head-on it is
    # a sliver; rotate by theta and it opens into a gap of roughly
    # (depth difference) * sin(theta) in world units. That ratio is why a
    # small culled fraction reads as a large black region.
    dsm = depth.astype(np.float64)
    dx = np.abs(np.diff(dsm, axis=1))
    jumps = dx[dx > 0.05 * dsm[:, :-1]]
    if len(jumps):
        med_jump = float(np.median(jumps))
        p90 = float(np.percentile(jumps, 90))
        print(f"\n  depth discontinuities: median {med_jump:.2f} m, p90 {p90:.2f} m")
        for angle in (5, 15, 30, 45):
            gap = med_jump * np.sin(np.radians(angle))
            print(f"    orbit {angle:>2}deg -> typical gap opens to ~{gap:.2f} m wide")

    print("\n  Reading: if 'no cull' barely differs from 0.06, the black")
    print("  regions are occlusion, not over-culling — there is no surface")
    print("  there to reconstruct, because the photo never saw it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
