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

code('''
# --- Cell 1: install EVERYTHING. Run this once, then Runtime -> Restart
# --- session, then run cell 2 onward. Never re-run this cell afterwards.
#
# Three pins in here are load-bearing. Each one was a cryptic failure the
# first time this notebook was run on a fresh Colab machine, so they are
# documented rather than left as bare version numbers:
#
#   transformers <5   TripoSR's checkpoint stores its DINO image tokenizer
#                     with the transformers 4.x ViT layer names
#                     ("encoder.layer.0.attention.attention.query"). v5
#                     renamed every one of them ("layers.0.attention.q_proj"),
#                     so loading the checkpoint dies with a wall of
#                     "Missing key(s) in state_dict" / "Unexpected key(s)".
#   transformers >=4.45  Depth Anything V2 needs it; older raises
#                     "KeyError: 'depth_anything'".
#   numpy ==1.26.4    TripoSR needs numpy<2, and Colab's numba refuses
#                     anything above 2.0 ("Numba needs NumPy 2.0 or less").
#
# ORDER MATTERS. The numpy pin must be the LAST install in this cell —
# almost every other package will happily pull numpy 2.x back in, and
# whichever install runs last is the one that wins.

!git clone -q https://github.com/VAST-AI-Research/TripoSR /content/TripoSR
!pip install -q -r /content/TripoSR/requirements.txt

# rembg sits in tsr.system's import chain and needs onnxruntime, which
# TripoSR's own requirements.txt does not list.
!pip install -q onnxruntime

!pip install -q "git+https://github.com/facebookresearch/sam2.git"
!pip install -q "transformers>=4.45,<5" accelerate

!pip install -q trimesh scipy opencv-python-headless
!pip install -q fastapi "uvicorn[standard]" python-multipart pyngrok nest-asyncio

# cupy ships with Colab, is built against numpy 2.x, and is unused here.
!pip uninstall -y -q cupy-cuda12x

# LAST. See the ORDER MATTERS note above.
!pip install -q --force-reinstall --no-cache-dir "numpy==1.26.4" scipy

print("\\nInstall finished. Verifying the pins that actually matter…\\n")
!python -c "import numpy, transformers; print(f'  numpy        {numpy.__version__}  (need 1.26.x)'); print(f'  transformers {transformers.__version__}  (need >=4.45, <5)')"
print("\\nIf either line above is wrong, re-run THIS cell only.")
print("NOW: Runtime -> Restart session, then run cell 2 onward.")
print("Do NOT run this cell again after restarting.")
''')

code('''
# --- Cell 2: get the pipeline code onto the machine.
#
# The geometry lives in this project's pipeline/ package rather than being
# pasted into the notebook, so the same code that the ground-truth self-tests
# exercise locally is the code that runs on the GPU here.

# Pre-filled so this notebook needs no editing — open it straight from
# GitHub and run. Change it only if you fork the repo.
REPO_URL = "https://github.com/SahilPawar28/holistic-scene-reconstruction"
PROJECT_DIR = "/content/holistic-scene-reconstruction"

import os, sys, shutil, subprocess

# Getting stale code onto the GPU box is the single most expensive mistake
# available here: everything runs, nothing errors, and you spend an evening
# debugging results produced by a version you already fixed. So this cell
# refuses to be quiet about which code it actually loaded.
if os.path.isdir(PROJECT_DIR) and not os.path.isdir(os.path.join(PROJECT_DIR, ".git")):
    # Uploaded as a zip rather than cloned. `git pull` would fail silently
    # here and leave the old files in place, which is exactly what happened.
    if REPO_URL:
        print(f"{PROJECT_DIR} is not a git checkout — replacing it with a fresh clone.")
        shutil.rmtree(PROJECT_DIR)
    else:
        print("WARNING: not a git checkout and no REPO_URL set — this code can")
        print("         only be updated by re-uploading the zip. If you have")
        print("         changed anything since the upload, re-upload it now.")

if not os.path.isdir(PROJECT_DIR):
    if not REPO_URL:
        raise SystemExit(
            "Set REPO_URL above, or upload the project folder to "
            f"{PROJECT_DIR} (Files pane -> upload, or mount Drive)."
        )
    subprocess.run(["git", "clone", "-q", REPO_URL, PROJECT_DIR], check=True)
elif os.path.isdir(os.path.join(PROJECT_DIR, ".git")):
    subprocess.run(["git", "-C", PROJECT_DIR, "fetch", "-q", "origin"], check=False)
    subprocess.run(["git", "-C", PROJECT_DIR, "reset", "--hard", "-q",
                    "origin/HEAD"], check=False)
    subprocess.run(["git", "-C", PROJECT_DIR, "pull", "-q"], check=False)

for path in (PROJECT_DIR, "/content/TripoSR"):
    if path not in sys.path:
        sys.path.insert(0, path)

# Drop any already-imported copy, so re-running this cell after a pull
# actually picks the new files up instead of the cached modules.
for name in [m for m in sys.modules if m == "pipeline" or m.startswith("pipeline.")]:
    del sys.modules[name]

import numpy, torch, transformers

# --- preflight ------------------------------------------------------
# Every version problem in this stack surfaces late and cryptically: a
# KeyError from a config mapping, a wall of state_dict key mismatches, an
# ImportError from numba. All of them are decidable right here, so check
# them up front and say exactly what to do instead.
def _version_tuple(text):
    parts = []
    for chunk in text.split(".")[:2]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)

problems = []
if _version_tuple(numpy.__version__) >= (2, 0):
    problems.append(
        f"numpy is {numpy.__version__}, needs 1.26.x — TripoSR and numba both "
        f"reject numpy 2.x"
    )
tf = _version_tuple(transformers.__version__)
if tf >= (5, 0):
    problems.append(
        f"transformers is {transformers.__version__}, needs <5 — v5 renamed the "
        f"ViT layers, so TripoSR's checkpoint will fail to load with "
        f"'Missing key(s) in state_dict'"
    )
elif tf < (4, 45):
    problems.append(
        f"transformers is {transformers.__version__}, needs >=4.45 — older "
        f"versions raise KeyError: 'depth_anything'"
    )

if problems:
    print("ENVIRONMENT IS WRONG:\\n")
    for p in problems:
        print("  *", p)
    print("\\nFix: run this, then Runtime -> Restart session, then re-run")
    print("this cell (cell 2). Do not re-run cell 1.\\n")
    print('  !pip install -q "transformers>=4.45,<5"')
    print('  !pip install -q --force-reinstall --no-cache-dir "numpy==1.26.4" scipy')
    raise SystemExit("environment check failed — see above")

print("numpy", numpy.__version__, "| transformers", transformers.__version__,
      "| torch", torch.__version__, "| cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("WARNING: no GPU. Runtime -> Change runtime type -> T4 GPU.")

from pipeline import (camera, depth as depth_mod, meshing, segmentation,
                      room_shell, objects, assembly, scene_compose)


def _has_module(name):
    import importlib
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False

# Version marker. Checking for a symbol that only exists in the current
# code is the only reliable way to know the pull worked — a stale copy
# imports perfectly happily and just behaves like the old version.
_missing = [
    name for name, present in [
        ("segmentation.looks_like_object", hasattr(segmentation, "looks_like_object")),
        ("meshing.depth_to_mesh(exclude_mask=)",
         "exclude_mask" in meshing.depth_to_mesh.__code__.co_varnames),
        ("assembly.place_objects", hasattr(assembly, "place_objects")),
        ("assembly semantic-aware gate",
         hasattr(assembly, "PlacementParams")
         and hasattr(assembly.PlacementParams(), "semantic_gate_relief")),
        ("pipeline.inpaint", _has_module("pipeline.inpaint")),
        ("pipeline.detection", _has_module("pipeline.detection")),
        ("segmentation.merge_instances", hasattr(segmentation, "merge_instances")),
        ("assembly.resolve_overlaps", hasattr(assembly, "resolve_overlaps")),
        ("assembly.TRIPOSR_TO_SCENE (calibrated orientation)",
         hasattr(assembly, "TRIPOSR_TO_SCENE")),
        ("segmentation.merge_instances (stepwise dependency)",
         hasattr(segmentation, "merge_instances")),
        ("assembly gate v2 (0.45 relief)",
         hasattr(assembly, "PlacementParams")
         and assembly.PlacementParams().semantic_gate_relief >= 0.45),
    ] if not present
]
if _missing:
    print("STALE PIPELINE CODE — missing:", ", ".join(_missing))
    print("The copy at", PROJECT_DIR, "is out of date. Set REPO_URL above and")
    print("re-run this cell, or delete that folder and re-upload the zip.")
    raise SystemExit("stale pipeline code — see above")

_rev = subprocess.run(["git", "-C", PROJECT_DIR, "log", "-1", "--format=%h %ad %s",
                       "--date=short"], capture_output=True, text=True)
print("pipeline loaded:", (_rev.stdout or "(not a git checkout)").strip()[:90])
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

# 4) CLIP for semantic labelling of segments. ~350MB, and it replaces a
#    stack of geometric heuristics that could only ever approximate
#    "is this region an object or part of the room".
from pipeline.semantic import SemanticLabeler

labeler = SemanticLabeler(device=DEVICE).load()
print("semantic labeller ready:", labeler.params.checkpoint)

# 5) GroundingDINO for text-prompted detection, recovering objects the
#    automatic mask generator's point grid misses (occluded, low-contrast,
#    or simply unlucky with where the grid landed).
from pipeline.detection import GroundingDinoDetector

detector = GroundingDinoDetector(device=DEVICE).load()
print("detector ready:", detector.params.checkpoint)

gc.collect(); torch.cuda.empty_cache()
print("VRAM used: %.2f GB" % (torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0))
''')

code('''
# --- Cell 4: the pipeline stages, wired together.

import base64, io, json, time
import numpy as np
from PIL import Image
import trimesh

from pipeline.assembly import (PlacementParams, kept_instance_ids,
                               place_objects)
from pipeline.camera import camera_from_image
from pipeline.meshing import MeshingParams, backproject_mask, depth_to_mesh
from pipeline.scene_compose import compose_scene, scene_statistics
from pipeline.serialization import json_safe
from pipeline.room_shell import RansacParams, fit_room_shell
from pipeline.segmentation import background_mask, occupancy_mask
from pipeline.inpaint import inpaint_background
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


def run_segmentation_detailed(image: Image.Image, depth_map, max_objects=6,
                              use_semantics=True, use_detection=True):
    """SAM2 (automatic) + GroundingDINO (text-prompted) -> merge -> CLIP
    labelling for anything detection did not already label -> filtering.

    Returns every intermediate, not just the final list — this is what
    powers the Stepwise viewer. `run_segmentation` below is the thin
    wrapper for callers that only want the end result.

    Two independent proposal mechanisms feed one instance list: the
    automatic pass samples a point grid and knows nothing about what it is
    looking at; the detection pass asks directly for named objects and
    recovers ones the grid missed (a chair blending into a similarly-toned
    floor was the case that motivated this — the automatic pass never
    proposed a mask for it at all, so no downstream filter could have kept
    it either).
    """
    segmenter.params.max_instances = int(max_objects)
    raw = segmenter.raw_masks(image)
    auto_instances = []
    for i, m in enumerate(raw):
        mask = segmentation.clean_mask(np.asarray(m["segmentation"], dtype=bool))
        area = int(np.count_nonzero(mask))
        if area == 0:
            continue
        auto_instances.append(segmentation.Instance(
            id=i, mask=mask, bbox=segmentation.mask_to_bbox(mask), area=area,
            score=float(m.get("predicted_iou", 1.0))))

    detections, detected_instances = [], []
    if use_detection:
        try:
            detections = detector.detect(image)
            detected_instances = segmentation.instances_from_detections(
                image, detections, segmenter.predictor, start_id=len(auto_instances)
            )
        except Exception as exc:
            print(f"  detection pass failed, continuing with automatic masks only: {exc}")

    merged = (segmentation.merge_instances(auto_instances, detected_instances)
             if use_detection else list(auto_instances))

    if use_semantics and merged:
        # Detection-seeded instances already carry a trusted label; only
        # label what came from the automatic pass and was not merged into
        # one of them.
        unlabelled = [
            i for i in merged
            if i.meta.get("source") != "detection" or not i.meta.get("semantic_trusted")
        ]
        if unlabelled:
            labeler.annotate(image, unlabelled)

    segmentation.attach_depth_stats(merged, depth_map)
    rejections = []
    filtered = segmentation.filter_instances(
        merged, depth=depth_map, params=segmenter.params, rejections=rejections)

    return {
        "auto_instances": auto_instances,
        "detections": detections,
        "detected_instances": detected_instances,
        "merged_instances": merged,
        "filtered_instances": filtered,
        "rejections": rejections,
    }


def run_segmentation(image: Image.Image, depth_map, max_objects=6,
                     rejections=None, use_semantics=True, use_detection=True):
    """Thin wrapper over run_segmentation_detailed for callers that only
    want the final object list (the /segment and /scene endpoints)."""
    result = run_segmentation_detailed(image, depth_map, max_objects,
                                       use_semantics, use_detection)
    if rejections is not None:
        rejections.extend(result["rejections"])
    return result["filtered_instances"]


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


def resize_max(image, max_side=720):
    """Cap an image's longest side, for base64 payloads that don't need
    full resolution — the Stepwise viewer's overlays are for inspection,
    not pixel-level scrutiny."""
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    scale = max_side / max(w, h)
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


def indexed_mask_overlay(image, instances, highlight_ids=None):
    """Like mask_overlay, but instances outside `highlight_ids` are dimmed.

    Used for the "objects vs. structure" stepwise stage: one image showing
    every candidate that survived merging, with the ones the filter actually
    kept as objects standing out and the rejected ones faded — so the
    filtering decision is visible in a single glance rather than a list of
    ids to cross-reference.
    """
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    palette = np.array([
        [255, 96, 96], [96, 200, 255], [140, 255, 140], [255, 210, 90],
        [220, 130, 255], [90, 255, 220], [255, 150, 200], [190, 190, 255],
    ], dtype=np.float32)
    out = base.copy()
    for i, inst in enumerate(instances):
        colour = palette[i % len(palette)]
        weight = 0.55 if (highlight_ids is None or inst.id in highlight_ids) else 0.22
        out[inst.mask] = (1 - weight) * base[inst.mask] + weight * colour
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def draw_detection_boxes(image, detections):
    """Boxes + label + score overlay for the detection stepwise stage."""
    from PIL import ImageDraw, ImageFont

    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    palette = [(255,96,96),(96,200,255),(140,255,140),(255,210,90),
              (220,130,255),(90,255,220),(255,150,200),(190,190,255)]
    for i, det in enumerate(detections):
        colour = palette[i % len(palette)]
        x0, y0, x1, y1 = [int(round(v)) for v in det.box]
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=3)
        label = f"{det.label[:26]} {det.score:.2f}"
        ty = max(0, y0 - 14)
        draw.rectangle([x0, ty, x0 + 7 * len(label) + 6, ty + 13], fill=colour)
        draw.text((x0 + 3, ty), label, fill=(15, 15, 18), font=font)
    return img


def generate_object_meshes(image, instances, depth_map, resolution=256, crops=None):
    """TripoSR once per object, in this one session.

    Batching matters here: the model stays resident, so a six-object scene
    costs six forward passes rather than six model loads.

    `crops` lets a caller that already built the crops (the Stepwise
    endpoint, which shows them as their own stage) hand them in rather than
    having them built a second time.
    """
    if crops is None:
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
        "endpoints": ["/depth", "/segment", "/object", "/tier1", "/scene", "/scene/stepwise"],
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
        "stats": json_safe({
            "objects": len(instances),
            "instances": [
                {"id": i.id, "area": i.area, "area_fraction": round(i.area_fraction, 4),
                 "bbox": list(i.bbox), "score": round(i.score, 3),
                 "depth_median": round(float(i.depth_median), 3)}
                for i in instances
            ],
        }),
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
            "stats": json_safe(stats)}


@app.post("/scene")
async def scene_endpoint(
    file: UploadFile = File(...),
    hfov_deg: float = Form(60.0),
    max_objects: int = Form(6),
    plane_threshold: float = Form(0.03),
    include_tier1_fallback: str = Form("1"),
    background_mode: str = Form("depth"),
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

    rejected = []
    instances = run_segmentation(image, depth_result.depth, max_objects,
                                 rejections=rejected)
    stats["segmentation"] = {
        "objects": len(instances),
        "instances": [
            {"id": i.id, "area": i.area, "bbox": list(i.bbox),
             "depth_median": round(float(i.depth_median), 3),
             "relief": round(float(i.meta.get("relative_relief", float("nan"))), 3),
             "label": i.meta.get("semantic_label"),
             "category": i.meta.get("semantic_category"),
             "confidence": i.meta.get("semantic_confidence")}
            for i in instances
        ],
        # Why candidates were thrown away. Without this, "the scene came out
        # empty" or "the scene is full of junk" are both undebuggable.
        "rejected": [
            {"area": i.area, "bbox": list(i.bbox), "reason": why}
            for i, why in rejected[:12]
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
        {"instance_id": g.instance_id,
         "label": g.crop.meta.get("semantic_label"),
         "crop_quality": g.crop.meta.get("crop_quality"),
         "crop_warnings": g.crop.meta.get("crop_warnings"),
         **g.meta}
        for g in generated
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

        # Background geometry is built AFTER the quality gate, and only
        # objects that survived it get their pixels cut out. A rejected
        # object leaves no hole, so the photograph shows through where it
        # would have been — strictly better than a hole with a blob in it.
        kept_ids = kept_instance_ids(placements)
        kept_instances = [i for i in instances if i.id in kept_ids]
        stats["quality_gate"] = {
            "kept": sorted(kept_ids),
            "rejected": [
                {"instance_id": p.instance_id, "reason": p.status,
                 "coverage": round(float(p.coverage), 3)}
                for p in placements if not p.ok
            ],
        }

        background_geom = None
        if background_mode in ("depth", "auto"):
            # Inpaint each kept object's footprint (colour via Telea, depth
            # via nearest-neighbour extrapolation) rather than cutting a
            # hole. A cut hole only looks right from the exact camera
            # position the photo was taken from — orbit at all and there is
            # nothing behind the object. An inpainted, continuous mesh has
            # a plausible (if low-detail) wall/table there instead.
            # Every DETECTED object gets its footprint inpainted, not just
            # the ones that ended up with a placed 3D mesh. A rejected
            # object still caused a real depth discontinuity in the original
            # photo, and Tier 1's own edge-aware culling removes the
            # bridging triangles there regardless of whether we placed
            # anything — so a rejected object left a black gap even though
            # no hole was ever deliberately cut. Inpainting the full
            # candidate set closes that; only kept_instances get an actual
            # mesh placed on top afterwards.
            occupied = occupancy_mask(instances, depth_result.depth.shape)
            inpainted_image, inpainted_depth = inpaint_background(
                image, depth_result.depth, occupied
            )
            background_geom = depth_to_mesh(
                inpainted_image, inpainted_depth, cam,
                MeshingParams(max_grid_side=480),
            ).mesh
            stats["background"] = {"mode": "depth-mesh-inpainted",
                                   "faces": int(len(background_geom.faces)),
                                   "inpainted_for": sorted(i.id for i in instances),
                                   "mesh_placed_for": sorted(kept_ids)}
        else:
            stats["background"] = {"mode": "fitted-planes"}

        scene = compose_scene(shell, generated, placements,
                              background_mesh=background_geom)
        stats["scene"] = scene_statistics(scene, placements)
        glb = glb_b64(scene)
    except Exception:
        stats["placement_error"] = traceback.format_exc(limit=4)
        if include_tier1_fallback not in ("1", "true", "True"):
            return JSONResponse(status_code=500, content={"detail": stats["placement_error"]})
        glb = glb_b64(tier1_mesh)

    return {"glb_base64": glb, "depth_png_base64": png_b64(depth_png),
            "overlay_png_base64": png_b64(overlay), "stats": json_safe(stats)}


@app.post("/scene/stepwise")
async def scene_stepwise_endpoint(
    file: UploadFile = File(...),
    hfov_deg: float = Form(60.0),
    max_objects: int = Form(6),
    plane_threshold: float = Form(0.03),
    background_mode: str = Form("depth"),
):
    """Every stage of Tier 2, captured as its own artifact.

    Runs the IDENTICAL pipeline /scene does -- same functions, same order,
    same objects.py/assembly.py/room_shell.py code -- it just keeps what
    each stage produced instead of only keeping what the next stage
    consumed. Nothing here is a separate, lighter, or approximate version
    of the real pipeline; it is the real pipeline with checkpoints.
    """
    image = await read_image(file)
    stages = []
    stats = {"mode": "stepwise"}

    def add_stage(sid, title, caption, **kw):
        stage = {"id": sid, "title": title, "caption": caption}
        stage.update(kw)
        stages.append(stage)

    add_stage("photo", "1. Photo", "The input photograph.",
              type="image", image_base64=png_b64(resize_max(image)))

    tier1_mesh, tier1_stats, depth_png, depth_result, cam = run_tier1(
        image, hfov_deg=hfov_deg
    )
    stats["tier1"] = tier1_stats
    dr = tier1_stats.get("depth_range", [0, 0])
    add_stage("depth", "2. Depth estimation",
              f"Depth Anything V2. Range {dr[0]:.2f}-{dr[1]:.2f}m. "
              "Every 3D point in every later stage comes from this map.",
              type="image", image_base64=png_b64(resize_max(depth_png)))

    seg = run_segmentation_detailed(image, depth_result.depth, max_objects)
    add_stage("sam2", "3. SAM2 automatic masks",
              f"{len(seg['auto_instances'])} raw regions proposed by a point-grid pass "
              "that knows nothing about what it is looking at.",
              type="image",
              image_base64=png_b64(resize_max(mask_overlay(image, seg["auto_instances"]))))

    det_source = (draw_detection_boxes(image, seg["detections"])
                 if seg["detections"] else image)
    add_stage("detection", "4. Text-prompted detection",
              f"{len(seg['detections'])} boxes from GroundingDINO, asking directly for "
              "named objects -- this is what recovers things the automatic pass misses.",
              type="image", image_base64=png_b64(resize_max(det_source)))

    kept_ids_seg = {i.id for i in seg["filtered_instances"]}
    add_stage("filtered", "5. Objects vs. structure",
              f"{len(seg['merged_instances'])} candidates after merging -> "
              f"{len(seg['filtered_instances'])} kept as real, distinct foreground objects. "
              "Dim regions were rejected as structure, duplicates, or noise.",
              type="image",
              image_base64=png_b64(resize_max(indexed_mask_overlay(
                  image, seg["merged_instances"], highlight_ids=kept_ids_seg))))
    stats["segmentation"] = {
        "auto_masks": len(seg["auto_instances"]),
        "detections": len(seg["detections"]),
        "merged": len(seg["merged_instances"]),
        "kept": len(seg["filtered_instances"]),
        "rejected": [{"area": i.area, "bbox": list(i.bbox), "reason": why}
                    for i, why in seg["rejections"][:20]],
    }

    instances = seg["filtered_instances"]
    crops = prepare_crops(image, instances, depth_result.depth, output_size=512)
    add_stage("crops", "6. Per-object crops",
              "Each object cropped to its own SAM2 mask, background flattened to grey, "
              "contrast-enhanced -- exactly what TripoSR receives, one at a time.",
              type="gallery",
              items=[{
                  "id": c.instance_id,
                  "label": c.meta.get("semantic_label") or f"object {c.instance_id}",
                  "image_base64": png_b64(c.image.resize((256, 256), Image.LANCZOS)),
                  "warnings": c.meta.get("crop_warnings", []),
              } for c in crops])

    generated = generate_object_meshes(image, instances, depth_result.depth, crops=crops)
    stats["generation"] = [
        {"instance_id": g.instance_id,
         "label": g.crop.meta.get("semantic_label"),
         "crop_warnings": g.crop.meta.get("crop_warnings"), **g.meta}
        for g in generated
    ]
    add_stage("models", "7. Generated 3D models",
              "TripoSR's output per object, each in its own canonical frame -- not yet "
              "scaled, rotated or positioned into the scene. Pick one to inspect it.",
              type="gallery3d",
              items=[{
                  "id": g.instance_id,
                  "label": g.crop.meta.get("semantic_label") or f"object {g.instance_id}",
                  "glb_base64": glb_b64(g.mesh) if g.mesh is not None else None,
                  "error": g.meta.get("error"),
                  "faces": g.meta.get("faces"),
              } for g in generated])

    bg = background_mask(instances, depth_result.depth.shape)
    bg_cloud = backproject_mask(bg, depth_result.depth, cam, max_points=60000,
                                depth_percentile_trim=None)
    shell = fit_room_shell(bg_cloud, image, cam,
                           RansacParams(distance_threshold=float(plane_threshold)))
    stats["room_shell"] = shell.stats

    placements = place_objects(generated, instances, depth_result.depth, cam, shell)
    stats["placement"] = [p.summary() for p in placements]
    kept_ids = kept_instance_ids(placements)
    kept_instances = [i for i in instances if i.id in kept_ids]
    stats["quality_gate"] = {
        "kept": sorted(kept_ids),
        "rejected": [{"instance_id": p.instance_id, "reason": p.status,
                     "coverage": round(float(p.coverage), 3)}
                    for p in placements if not p.ok],
    }

    occupied = occupancy_mask(instances, depth_result.depth.shape)
    inpainted_image, inpainted_depth = inpaint_background(
        image, depth_result.depth, occupied
    )
    add_stage("background", "8. Background fill",
              "Every detected object's footprint is filled -- colour via texture "
              "propagation, depth via a smooth harmonic solve -- before meshing. A "
              "rejected object leaves a plausible wall behind it, not a hole.",
              type="compare",
              before_base64=png_b64(resize_max(image)),
              after_base64=png_b64(resize_max(inpainted_image)))

    background_geom = depth_to_mesh(
        inpainted_image, inpainted_depth, cam, MeshingParams(max_grid_side=480)
    ).mesh
    stats["background"] = {"faces": int(len(background_geom.faces)),
                           "mesh_placed_for": sorted(kept_ids)}

    scene = compose_scene(shell, generated, placements, background_mesh=background_geom)
    stats["scene"] = scene_statistics(scene, placements)
    add_stage("final", "9. Final assembled scene",
              f"{len(kept_ids)} of {len(instances)} detected objects placed and composed "
              "with the inpainted background into one scene graph.",
              type="glb", glb_base64=glb_b64(scene))

    return {"stages": stages, "stats": json_safe(stats)}


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

# NOT uvicorn.run() — it calls asyncio.run() internally, which raises
# "asyncio.run() cannot be called from a running event loop" because the
# notebook kernel already has one. Driving the Server object directly and
# awaiting it uses the kernel's existing loop instead. Colab supports
# top-level await, so this works in a cell.
config = uvicorn.Config(app, port=8000, log_level="warning")
server = uvicorn.Server(config)
await server.serve()
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
