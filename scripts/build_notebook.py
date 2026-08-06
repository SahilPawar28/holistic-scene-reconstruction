"""
Generates colab/scene_pipeline.ipynb.

The notebook is kept as a build script rather than hand-edited JSON: the
cells are long, they change together (an endpoint added in the server cell
usually needs a helper in the pipeline cell), and reviewing a diff of
notebook JSON is miserable. Run this after changing anything here:

    python scripts/build_notebook.py
"""

from __future__ import annotations

import json
import os

CELLS: list[tuple[str, str]] = []


def md(source: str) -> None:
    CELLS.append(("markdown", source.strip("\n")))


def code(source: str) -> None:
    CELLS.append(("code", source.strip("\n")))


md("""
# Scene pipeline server (Colab, free T4 GPU)

Hosts every GPU stage of the holistic scene reconstruction pipeline behind
one API for `backend/main.py` to call:

| endpoint | what it does |
|---|---|
| `GET  /health`  | liveness + which models are loaded |
| `POST /depth`   | depth map only (debugging) |
| `POST /segment` | instance masks only (debugging) |
| `POST /object`  | one pre-cropped object -> mesh (TripoSR) |
| `POST /tier1`   | whole image -> one continuous scene mesh |
| `POST /scene`   | Tier 2: shell + placed objects, one .glb |

**One notebook, not two.** Free Colab gives you a single GPU session, so
Depth Anything V2, SAM2 and TripoSR all live here together. That also means
each model is loaded once and reused across every object in a scene, instead
of paying the load cost per call.

### Running it

1. `Runtime` -> `Change runtime type` -> **T4 GPU**
2. Run cell 1 (installs). Then **`Runtime` -> `Restart session`** — required
   exactly once. TripoSR pins `numpy<2` while Colab ships packages built
   against numpy 2, so the environment is inconsistent until a restart.
3. Run cells 2 onward in order.
4. The last cell prints an ngrok URL. Paste it into the server field in the
   frontend (or into `backend/config.json`). Leave that cell running.

Free ngrok token: https://dashboard.ngrok.com/get-started/your-authtoken
""")

code("""
# --- Cell 1: install. Run this, then Runtime -> Restart session, then
# --- continue from cell 2. You only ever run this cell once per session.

# TripoSR first, because its requirements pin numpy<2 and we want to be the
# ones who decide the final numpy version, not whatever installs last.
!git clone -q https://github.com/VAST-AI-Research/TripoSR /content/TripoSR
!pip install -q -r /content/TripoSR/requirements.txt

# SAM2 (instance segmentation) and Depth Anything V2 (via transformers).
!pip install -q "git+https://github.com/facebookresearch/sam2.git"
!pip install -q transformers accelerate

# Geometry + serving.
!pip install -q trimesh scipy opencv-python-headless
!pip install -q fastapi "uvicorn[standard]" python-multipart pyngrok nest-asyncio

# Same ABI fix as the v1 notebook: TripoSR drags numpy below 2.0, which
# breaks Colab's preinstalled cupy and scipy (both built against numpy>=2).
# cupy is unused here so it goes; scipy is needed, so it gets rebuilt to
# match rather than removed.
!pip uninstall -y -q cupy-cuda12x
!pip install -q --force-reinstall --no-cache-dir "numpy==1.26.4" scipy

print("Install finished.")
print("NOW: Runtime -> Restart session, then run cell 2 onward.")
""")

code('''
# --- Cell 2: get the pipeline code onto the machine.
#
# The geometry lives in this project's pipeline/ package rather than being
# pasted into the notebook, so the same code that the ground-truth self-tests
# exercise locally is the code that runs on the GPU here.

REPO_URL = ""   # e.g. "https://github.com/<you>/holistic-scene-reconstruction"
PROJECT_DIR = "/content/holistic-scene-reconstruction"

import os, sys, subprocess

if not os.path.isdir(PROJECT_DIR):
    if REPO_URL:
        subprocess.run(["git", "clone", "-q", REPO_URL, PROJECT_DIR], check=True)
    else:
        raise SystemExit(
            "Set REPO_URL above, or upload the project folder to "
            f"{PROJECT_DIR} (Files pane -> upload, or mount Drive)."
        )
else:
    # Already present — pull so a re-run picks up local edits.
    subprocess.run(["git", "-C", PROJECT_DIR, "pull", "-q"], check=False)

for path in (PROJECT_DIR, "/content/TripoSR"):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy, torch
print("numpy", numpy.__version__, "| torch", torch.__version__,
      "| cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("WARNING: no GPU. Runtime -> Change runtime type -> T4 GPU.")

from pipeline import (camera, depth as depth_mod, meshing, segmentation,
                      room_shell, objects, assembly, scene_compose)
print("pipeline package loaded")
''')

code('''
# --- Cell 3: load the three models. ~4 GB of VRAM total on a 15 GB T4,
# --- which leaves plenty of headroom for TripoSR's marching-cubes pass.

import torch, gc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Depth Anything V2, metric indoor checkpoint. Metric rather than
#    relative because it removes the scale ambiguity outright, which the
#    room-shell and placement stages both benefit from.
depth_model = depth_mod.DepthAnythingV2(
    checkpoint="depth-anything/Depth-Anything-V2-metric-indoor-small-hf",
    device=DEVICE,
).load()
print("depth model ready:", depth_model.checkpoint)

# 2) SAM2 for instance masks.
segmenter = segmentation.SAM2Segmenter(
    checkpoint="facebook/sam2.1-hiera-small", device=DEVICE
).load()
print("segmenter ready:", segmenter.checkpoint)

# 3) TripoSR for per-object generation — the same model, same settings as
#    the v1 pipeline, just invoked once per object instead of once per image.
from tsr.system import TSR

triposr = TSR.from_pretrained(
    "stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt"
)
triposr.renderer.set_chunk_size(8192)
triposr.to(DEVICE)
print("TripoSR ready")

gc.collect(); torch.cuda.empty_cache()
print("VRAM used: %.2f GB" % (torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0))
''')

code('''
# --- Cell 4: the pipeline stages, wired together.

import base64, io, json, time
import numpy as np
from PIL import Image
import trimesh

from pipeline.assembly import PlacementParams, place_objects
from pipeline.camera import camera_from_image
from pipeline.meshing import MeshingParams, backproject_mask, depth_to_mesh
from pipeline.scene_compose import compose_scene, scene_statistics
from pipeline.room_shell import RansacParams, fit_room_shell
from pipeline.segmentation import background_mask
from pipeline.objects import canonicalize_mesh, prepare_crops


def to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def png_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return to_b64(buf.getvalue())


def glb_b64(mesh_or_scene) -> str:
    return to_b64(mesh_or_scene.export(file_type="glb"))


def run_depth(image: Image.Image, hfov_deg: float = 60.0):
    """Depth + intrinsics for one image — the input to every later stage."""
    cam = camera_from_image(image.width, image.height, hfov_deg=hfov_deg)
    result = depth_model.predict(image, max_side=700)
    return result, cam


def run_tier1(image: Image.Image, hfov_deg=60.0, max_grid_side=480,
              max_relative_depth_jump=0.06, max_grazing_angle_deg=87.0):
    """Whole image -> one continuous textured mesh."""
    t0 = time.time()
    depth_result, cam = run_depth(image, hfov_deg)
    t_depth = time.time() - t0

    t0 = time.time()
    tier1 = depth_to_mesh(
        image, depth_result.depth, cam,
        MeshingParams(
            max_grid_side=int(max_grid_side),
            max_relative_depth_jump=float(max_relative_depth_jump),
            max_grazing_angle_deg=float(max_grazing_angle_deg),
        ),
    )
    stats = dict(tier1.stats)
    stats.update({
        "mode": "tier1",
        "depth_checkpoint": depth_result.checkpoint,
        "depth_is_metric": depth_result.is_metric,
        "seconds_depth": round(t_depth, 2),
        "seconds_mesh": round(time.time() - t0, 2),
    })
    depth_png = Image.fromarray(depth_result.normalized_for_preview())
    return tier1.mesh, stats, depth_png, depth_result, cam


def run_segmentation(image: Image.Image, depth_map, max_objects=6):
    segmenter.params.max_instances = int(max_objects)
    return segmenter.segment(image, depth=depth_map)


def mask_overlay(image: Image.Image, instances) -> Image.Image:
    """Tint each instance mask over the photo — the quickest way to see
    whether segmentation, and not the geometry, is what went wrong."""
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    palette = np.array([
        [255, 96, 96], [96, 200, 255], [140, 255, 140], [255, 210, 90],
        [220, 130, 255], [90, 255, 220], [255, 150, 200], [190, 190, 255],
    ], dtype=np.float32)
    for i, inst in enumerate(instances):
        colour = palette[i % len(palette)]
        base[inst.mask] = 0.5 * base[inst.mask] + 0.5 * colour
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def generate_object_meshes(image, instances, depth_map, resolution=256):
    """TripoSR once per object, in this one session.

    Batching matters here: the model stays resident, so a six-object scene
    costs six forward passes rather than six model loads.
    """
    crops = prepare_crops(image, instances, depth_map, output_size=512)
    generated = []
    for crop in crops:
        t0 = time.time()
        try:
            # The crop is already background-flattened using the instance
            # mask, so TripoSR's own rembg step is skipped — our mask is
            # strictly better information than its guess would be.
            arr = np.asarray(crop.image).astype(np.float32) / 255.0
            processed = Image.fromarray((arr * 255).astype(np.uint8))
            with torch.no_grad():
                codes = triposr([processed], device=DEVICE)
            mesh = triposr.extract_mesh(codes, has_vertex_color=True,
                                        resolution=int(resolution))[0]
            torch.cuda.empty_cache()
            if mesh is None or len(mesh.faces) == 0:
                raise RuntimeError("TripoSR returned an empty mesh")
            canonical, scale, centre = canonicalize_mesh(mesh)
            generated.append(objects.GeneratedObject(
                instance_id=crop.instance_id, mesh=canonical,
                canonical_scale=scale, canonical_center=centre, crop=crop,
                meta={"faces": int(len(canonical.faces)),
                      "seconds": round(time.time() - t0, 2),
                      "occlusion": round(crop.occlusion, 3)},
            ))
        except Exception as exc:
            generated.append(objects.GeneratedObject(
                instance_id=crop.instance_id, mesh=None, canonical_scale=1.0,
                canonical_center=np.zeros(3), crop=crop,
                meta={"error": f"{type(exc).__name__}: {exc}"},
            ))
            torch.cuda.empty_cache()
    return generated


print("pipeline stages defined")
''')

code('''
# --- Cell 5: the API server.

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uuid, os, io, traceback

app = FastAPI(title="Scene pipeline server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

OUT_DIR = "/content/outputs"
os.makedirs(OUT_DIR, exist_ok=True)


async def read_image(file: UploadFile) -> Image.Image:
    return Image.open(io.BytesIO(await file.read())).convert("RGB")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "models": {
            "depth": depth_model.checkpoint,
            "segmentation": segmenter.checkpoint,
            "generation": "stabilityai/TripoSR",
        },
        "endpoints": ["/depth", "/segment", "/object", "/tier1", "/scene"],
    }


@app.post("/depth")
async def depth_endpoint(file: UploadFile = File(...), hfov_deg: float = Form(60.0)):
    image = await read_image(file)
    result, cam = run_depth(image, hfov_deg)
    return {
        "depth_png_base64": png_b64(Image.fromarray(result.normalized_for_preview())),
        "stats": {
            "checkpoint": result.checkpoint,
            "is_metric": result.is_metric,
            "depth_min": float(np.nanmin(result.depth)),
            "depth_max": float(np.nanmax(result.depth)),
            "hfov_deg": round(cam.hfov_deg, 2),
            "focal_px": round(cam.fx, 1),
        },
    }


@app.post("/segment")
async def segment_endpoint(file: UploadFile = File(...), hfov_deg: float = Form(60.0),
                           max_objects: int = Form(6)):
    image = await read_image(file)
    result, cam = run_depth(image, hfov_deg)
    instances = run_segmentation(image, result.depth, max_objects)
    return {
        "overlay_png_base64": png_b64(mask_overlay(image, instances)),
        "stats": {
            "objects": len(instances),
            "instances": [
                {"id": i.id, "area": i.area, "area_fraction": round(i.area_fraction, 4),
                 "bbox": list(i.bbox), "score": round(i.score, 3),
                 "depth_median": round(float(i.depth_median), 3)}
                for i in instances
            ],
        },
    }


@app.post("/object")
async def object_endpoint(file: UploadFile = File(...), resolution: int = Form(256),
                          skip_background_removal: str = Form("1")):
    """One already-cropped object -> one .glb. Used by TripoSRClient when
    the pipeline is driven from outside the notebook."""
    image = await read_image(file)
    if skip_background_removal not in ("1", "true", "True"):
        import rembg
        from tsr.utils import remove_background, resize_foreground
        image = remove_background(image, rembg.new_session())
        image = resize_foreground(image, 0.85)
        arr = np.array(image).astype(np.float32) / 255.0
        arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
        image = Image.fromarray((arr * 255).astype(np.uint8))

    with torch.no_grad():
        codes = triposr([image], device=DEVICE)
    mesh = triposr.extract_mesh(codes, has_vertex_color=True, resolution=int(resolution))[0]
    torch.cuda.empty_cache()

    path = f"{OUT_DIR}/{uuid.uuid4()}.glb"
    mesh.export(path)
    return FileResponse(path, media_type="model/gltf-binary", filename="object.glb")


@app.post("/tier1")
async def tier1_endpoint(
    file: UploadFile = File(...),
    hfov_deg: float = Form(60.0),
    max_grid_side: int = Form(480),
    max_relative_depth_jump: float = Form(0.06),
    max_grazing_angle_deg: float = Form(87.0),
):
    image = await read_image(file)
    mesh, stats, depth_png, _, _ = run_tier1(
        image, hfov_deg, max_grid_side, max_relative_depth_jump, max_grazing_angle_deg
    )
    return {"glb_base64": glb_b64(mesh), "depth_png_base64": png_b64(depth_png),
            "stats": stats}


@app.post("/scene")
async def scene_endpoint(
    file: UploadFile = File(...),
    hfov_deg: float = Form(60.0),
    max_objects: int = Form(6),
    plane_threshold: float = Form(0.03),
    include_tier1_fallback: str = Form("1"),
):
    """Tier 2: room shell + per-object meshes, composed into one scene.

    Stages run in order and each one records its own timing and diagnostics,
    so a bad result can be attributed to a stage rather than guessed at.
    """
    image = await read_image(file)
    stats = {"mode": "tier2"}

    tier1_mesh, tier1_stats, depth_png, depth_result, cam = run_tier1(
        image, hfov_deg=hfov_deg
    )
    stats["tier1"] = tier1_stats

    instances = run_segmentation(image, depth_result.depth, max_objects)
    stats["segmentation"] = {
        "objects": len(instances),
        "instances": [
            {"id": i.id, "area": i.area, "bbox": list(i.bbox),
             "depth_median": round(float(i.depth_median), 3)}
            for i in instances
        ],
    }
    overlay = mask_overlay(image, instances)

    bg = background_mask(instances, depth_result.depth.shape)
    bg_cloud = backproject_mask(bg, depth_result.depth, cam, max_points=60000,
                                depth_percentile_trim=None)
    shell = fit_room_shell(bg_cloud, image, cam,
                           RansacParams(distance_threshold=float(plane_threshold)))
    stats["room_shell"] = shell.stats

    generated = generate_object_meshes(image, instances, depth_result.depth)
    stats["generation"] = [
        {"instance_id": g.instance_id, **g.meta} for g in generated
    ]

    # --- placement + composition -------------------------------------
    # The core of the project: solve each object's similarity transform
    # against its own back-projected target cloud, settle it onto whatever
    # supports it, and assemble everything into one scene graph.
    try:
        t0 = time.time()
        placements = place_objects(generated, instances, depth_result.depth, cam, shell)
        stats["placement"] = [p.summary() for p in placements]
        stats["placement_seconds"] = round(time.time() - t0, 2)

        scene = compose_scene(shell, generated, placements)
        stats["scene"] = scene_statistics(scene, placements)
        glb = glb_b64(scene)
    except Exception:
        stats["placement_error"] = traceback.format_exc(limit=4)
        if include_tier1_fallback not in ("1", "true", "True"):
            return JSONResponse(status_code=500, content={"detail": stats["placement_error"]})
        glb = glb_b64(tier1_mesh)

    return {"glb_base64": glb, "depth_png_base64": png_b64(depth_png),
            "overlay_png_base64": png_b64(overlay), "stats": stats}


print("server defined —", len(app.routes), "routes")
''')

code('''
# --- Cell 6: expose it. Paste your ngrok token, run, copy the printed URL
# --- into the frontend's server field. Leave this cell running.

NGROK_AUTHTOKEN = "PASTE_YOUR_TOKEN_HERE"

import nest_asyncio, uvicorn
from pyngrok import ngrok

nest_asyncio.apply()
ngrok.set_auth_token(NGROK_AUTHTOKEN)
ngrok.kill()  # drop any tunnel left over from a previous run of this cell
public_url = ngrok.connect(8000).public_url

print("\\n" + "=" * 60)
print("SERVER URL — paste this into the frontend:")
print("   ", public_url)
print("=" * 60)
print("Leave this cell running. Stopping it stops the server.\\n")

uvicorn.run(app, port=8000, log_level="warning")
''')


def build() -> dict:
    return {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": source.splitlines(keepends=True),
                **({"execution_count": None, "outputs": []} if kind == "code" else {}),
            }
            for kind, source in CELLS
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "colab", "scene_pipeline.ipynb")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(build(), f, indent=1)
        f.write("\n")
    print(f"wrote {out}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
