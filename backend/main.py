"""
FastAPI backend for the holistic scene reconstruction pipeline.

Same role as the v1 image-to-3d backend: a thin proxy in front of the Colab
GPU server, plus the persistence layer that keeps every reconstruction
around after a restart. The GPU work all happens in colab/scene_pipeline.ipynb.

Three reconstruction modes are exposed, in increasing ambition:

    POST /convert       v1 behaviour — one object, one mesh (TripoSR only)
    POST /scene/tier1   whole image -> one continuous textured scene mesh
    POST /scene/tier2   compositional — room shell + per-object meshes

Tier 1 is the fallback that always produces something; Tier 2 is the real
target. Both are kept because the honest comparison between them is part of
the deliverable.
"""

import base64
import json
import os
import shutil
import uuid
from datetime import datetime, timezone

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Holistic Scene Reconstruction")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_DIR = os.path.join(BASE_DIR, "history")
IMAGES_DIR = os.path.join(HISTORY_DIR, "images")
MODELS_DIR = os.path.join(HISTORY_DIR, "models")
DEPTH_DIR = os.path.join(HISTORY_DIR, "depth")
STAGES_DIR = os.path.join(HISTORY_DIR, "stages")
HISTORY_FILE = os.path.join(HISTORY_DIR, "history.json")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

for d in (IMAGES_DIR, MODELS_DIR, DEPTH_DIR, STAGES_DIR):
    os.makedirs(d, exist_ok=True)

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)

app.mount("/files", StaticFiles(directory=HISTORY_DIR), name="files")

# The ngrok URL changes every time the Colab notebook restarts. In v1 that
# meant editing this file and restarting uvicorn every session, which was a
# genuine papercut during testing — so it now lives in config.json and can
# be set from the frontend via POST /config.
DEFAULT_COLAB_URL = os.environ.get("COLAB_SERVER_URL", "")

# Tier 2 runs depth, segmentation and one TripoSR pass per object, so a
# multi-object scene legitimately takes several minutes on a free T4.
# Stepwise runs the identical pipeline plus encodes every intermediate
# artifact to base64, which costs real time on a slow tunnel — budgeted
# generously rather than have a demo run die two minutes from the end.
TIMEOUTS = {"convert": 300, "tier1": 420, "tier2": 1200, "stepwise": 1500}


def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"colab_server_url": DEFAULT_COLAB_URL}


def save_config(cfg: dict) -> None:
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def colab_url() -> str:
    url = load_config().get("colab_server_url", "").strip().rstrip("/")
    if not url:
        raise HTTPException(
            status_code=503,
            detail=(
                "No Colab server URL configured. Run the last cell of "
                "colab/scene_pipeline.ipynb and paste the printed ngrok URL "
                "into the server field in the UI (or POST it to /config)."
            ),
        )
    return url


def load_history() -> list:
    with open(HISTORY_FILE) as f:
        return json.load(f)


def save_history(records: list) -> None:
    with open(HISTORY_FILE, "w") as f:
        json.dump(records, f, indent=2)


class ConfigUpdate(BaseModel):
    colab_server_url: str


@app.get("/config")
def get_config():
    cfg = load_config()
    url = cfg.get("colab_server_url", "")
    reachable, detail = False, "not configured"
    if url:
        try:
            r = requests.get(f"{url.rstrip('/')}/health", timeout=8)
            reachable = r.status_code == 200
            detail = r.text[:200] if reachable else f"HTTP {r.status_code}"
        except requests.exceptions.RequestException as exc:
            detail = f"{type(exc).__name__}"
    return {"colab_server_url": url, "reachable": reachable, "detail": detail}


@app.post("/config")
def set_config(update: ConfigUpdate):
    cfg = load_config()
    cfg["colab_server_url"] = update.colab_server_url.strip().rstrip("/")
    save_config(cfg)
    return get_config()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/history")
def get_history():
    return load_history()


def _save_upload(file: UploadFile) -> tuple[str, str, str]:
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")
    record_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or "")[1] or ".png"
    name = f"{record_id}{ext}"
    path = os.path.join(IMAGES_DIR, name)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return record_id, name, path


def _post_to_colab(endpoint: str, image_path: str, data: dict, timeout: int):
    """POST an image to the Colab server, with the failure modes spelled out.

    Colab sessions die (idle timeout, GPU quota) far more often than a normal
    backend dependency does, so each failure gets a message that says what to
    actually do about it rather than a bare 502.
    """
    url = f"{colab_url()}{endpoint}"
    try:
        with open(image_path, "rb") as fh:
            resp = requests.post(
                url,
                files={"file": (os.path.basename(image_path), fh, "image/png")},
                data=data,
                timeout=timeout,
            )
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=502,
            detail=(
                "Could not reach the Colab server. Check the notebook's last "
                "cell is still running, and that the ngrok URL here matches "
                "the one it printed (it changes on every restart)."
            ),
        )
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail=(
                f"Colab did not respond within {timeout}s. Tier 2 on a busy "
                "scene can exceed this — try fewer objects, or a lower mesh "
                "resolution."
            ),
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Colab server error ({resp.status_code}): {resp.text[:400]}",
        )
    return resp


def _decode_scene_response(resp, record_id: str) -> dict:
    """Unpack the Colab response into saved files plus stats.

    The scene endpoints return JSON rather than raw bytes, because a scene
    reconstruction is not just a .glb — the depth preview and the per-stage
    statistics are what make the result explainable, and carrying them in
    one round trip beats three separate calls to a flaky tunnel.
    """
    try:
        payload = resp.json()
    except ValueError:
        raise HTTPException(
            status_code=502, detail="Colab returned a non-JSON response"
        )

    if "glb_base64" not in payload:
        raise HTTPException(
            status_code=502,
            detail=f"Colab response missing 'glb_base64': {str(payload)[:300]}",
        )

    model_name = f"{record_id}.glb"
    with open(os.path.join(MODELS_DIR, model_name), "wb") as f:
        f.write(base64.b64decode(payload["glb_base64"]))

    out = {"model_url": f"/files/models/{model_name}", "stats": payload.get("stats", {})}

    if payload.get("depth_png_base64"):
        depth_name = f"{record_id}_depth.png"
        with open(os.path.join(DEPTH_DIR, depth_name), "wb") as f:
            f.write(base64.b64decode(payload["depth_png_base64"]))
        out["depth_url"] = f"/files/depth/{depth_name}"

    if payload.get("overlay_png_base64"):
        overlay_name = f"{record_id}_overlay.png"
        with open(os.path.join(DEPTH_DIR, overlay_name), "wb") as f:
            f.write(base64.b64decode(payload["overlay_png_base64"]))
        out["overlay_url"] = f"/files/depth/{overlay_name}"

    return out


def _save_b64_file(data_b64: str, folder: str, name: str) -> str:
    """Decode one base64 artifact to disk, return its /files URL."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    with open(path, "wb") as f:
        f.write(base64.b64decode(data_b64))
    rel = os.path.relpath(path, HISTORY_DIR).replace(os.sep, "/")
    return f"/files/{rel}"


def _decode_stepwise_response(resp, record_id: str) -> dict:
    """Unpack a /scene/stepwise response: every stage's artifact saved to
    its own file, base64 replaced with URLs the frontend can load directly.

    Each stage type carries a different shape of payload (a single image, a
    before/after pair, a gallery of crops, a gallery of individual .glb
    files, or the final composed .glb), so this switches on `type` rather
    than assuming one shape fits all of them.
    """
    try:
        payload = resp.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="Colab returned a non-JSON response")

    if "stages" not in payload:
        raise HTTPException(
            status_code=502,
            detail=f"Colab response missing 'stages': {str(payload)[:300]}",
        )

    stage_dir = os.path.join(STAGES_DIR, record_id)
    out_stages = []
    model_url = None

    for stage in payload["stages"]:
        sid = stage["id"]
        entry = {"id": sid, "title": stage.get("title", sid),
                 "caption": stage.get("caption", ""), "type": stage["type"]}

        if stage["type"] == "image":
            entry["image_url"] = _save_b64_file(
                stage["image_base64"], stage_dir, f"{sid}.png"
            )
        elif stage["type"] == "compare":
            entry["before_url"] = _save_b64_file(
                stage["before_base64"], stage_dir, f"{sid}_before.png"
            )
            entry["after_url"] = _save_b64_file(
                stage["after_base64"], stage_dir, f"{sid}_after.png"
            )
        elif stage["type"] == "gallery":
            entry["items"] = [
                {
                    "id": it["id"], "label": it.get("label"),
                    "warnings": it.get("warnings", []),
                    "image_url": _save_b64_file(
                        it["image_base64"], stage_dir, f"{sid}_{it['id']}.png"
                    ),
                }
                for it in stage["items"]
            ]
        elif stage["type"] == "gallery3d":
            items = []
            for it in stage["items"]:
                item = {"id": it["id"], "label": it.get("label"),
                        "error": it.get("error"), "faces": it.get("faces")}
                if it.get("glb_base64"):
                    item["model_url"] = _save_b64_file(
                        it["glb_base64"], stage_dir, f"{sid}_{it['id']}.glb"
                    )
                items.append(item)
            entry["items"] = items
        elif stage["type"] == "glb":
            entry["model_url"] = _save_b64_file(
                stage["glb_base64"], stage_dir, f"{sid}.glb"
            )
            model_url = entry["model_url"]  # the final stage's scene

        out_stages.append(entry)

    out = {"stages": out_stages, "stats": payload.get("stats", {})}
    if model_url:
        # Keeps stepwise records compatible with everything that expects a
        # top-level model_url (the history thumbnail row, the default 3D
        # view when a record is selected).
        out["model_url"] = model_url
    return out


def _record(record_id: str, mode: str, filename: str, image_name: str, extra: dict) -> dict:
    record = {
        "id": record_id,
        "mode": mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "image_url": f"/files/images/{image_name}",
        **extra,
    }
    history = load_history()
    history.append(record)
    save_history(history)
    return record


@app.post("/convert")
async def convert(file: UploadFile = File(...)):
    """v1 behaviour: a single object, straight through TripoSR.

    Kept working so the two approaches can be demoed side by side on the
    same photo — the whole point of the project is the difference between
    "the main object" and "the whole scene".
    """
    record_id, image_name, image_path = _save_upload(file)
    resp = _post_to_colab("/convert", image_path, {}, TIMEOUTS["convert"])

    model_name = f"{record_id}.glb"
    with open(os.path.join(MODELS_DIR, model_name), "wb") as f:
        f.write(resp.content)

    return _record(record_id, "object", file.filename, image_name,
                   {"model_url": f"/files/models/{model_name}"})


@app.post("/convert/triposg")
async def convert_triposg(file: UploadFile = File(...)):
    """Single object -> mesh via TripoSG, for direct comparison against
    /convert (TripoSR) on the identical photo.

    A standalone generator test, not part of the Tier 1/2/3 pipeline — the
    point is to see TripoSG's output on a few basic objects before deciding
    whether it's worth swapping into the real pipeline anywhere. Colab
    reports 503 if TripoSG failed to load in that session; every other mode
    keeps working either way.
    """
    record_id, image_name, image_path = _save_upload(file)
    resp = _post_to_colab("/object/triposg", image_path, {}, TIMEOUTS["convert"])

    model_name = f"{record_id}.glb"
    with open(os.path.join(MODELS_DIR, model_name), "wb") as f:
        f.write(resp.content)

    return _record(record_id, "triposg", file.filename, image_name,
                   {"model_url": f"/files/models/{model_name}"})


@app.post("/scene/tier1")
async def scene_tier1(
    file: UploadFile = File(...),
    hfov_deg: float = Form(60.0),
    max_grid_side: int = Form(480),
    max_relative_depth_jump: float = Form(0.06),
    max_grazing_angle_deg: float = Form(87.0),
):
    """Whole image -> one continuous textured mesh.

    The culling thresholds are exposed rather than hard-coded because the
    right value is scene-dependent, and being able to move them during a
    demo is the clearest way to show what edge-aware culling actually does.
    """
    record_id, image_name, image_path = _save_upload(file)
    resp = _post_to_colab(
        "/tier1",
        image_path,
        {
            "hfov_deg": str(hfov_deg),
            "max_grid_side": str(max_grid_side),
            "max_relative_depth_jump": str(max_relative_depth_jump),
            "max_grazing_angle_deg": str(max_grazing_angle_deg),
        },
        TIMEOUTS["tier1"],
    )
    return _record(record_id, "tier1", file.filename, image_name,
                   _decode_scene_response(resp, record_id))


@app.post("/scene/tier2")
async def scene_tier2(
    file: UploadFile = File(...),
    hfov_deg: float = Form(60.0),
    max_objects: int = Form(6),
    plane_threshold: float = Form(0.03),
    include_tier1_fallback: bool = Form(True),
):
    """Compositional scene: room shell + one generated mesh per object.

    `include_tier1_fallback` asks the server to fall back to the Tier 1 mesh
    if the compositional path fails outright, so a demo never ends with an
    empty viewer.
    """
    record_id, image_name, image_path = _save_upload(file)
    resp = _post_to_colab(
        "/scene",
        image_path,
        {
            "hfov_deg": str(hfov_deg),
            "max_objects": str(max_objects),
            "plane_threshold": str(plane_threshold),
            "include_tier1_fallback": "1" if include_tier1_fallback else "0",
        },
        TIMEOUTS["tier2"],
    )
    return _record(record_id, "tier2", file.filename, image_name,
                   _decode_scene_response(resp, record_id))


@app.post("/scene/tier3")
async def scene_tier3(
    file: UploadFile = File(...),
    hfov_deg: float = Form(60.0),
    max_objects: int = Form(6),
    plane_threshold: float = Form(0.03),
    include_tier1_fallback: bool = Form(True),
):
    """Identical generation to Tier 2 -- same Colab pipeline, same placement
    solver, same quality gate and support snapping. The only difference is
    what happens after the scene loads in the viewer: Tier 3 records let
    the user click an object and drag/rotate it freely, with no physics or
    collision constraint re-applied client-side. The placement solver still
    ran for real (this is a starting arrangement worth trusting, not a
    blank scene), it just isn't the last word.
    """
    record_id, image_name, image_path = _save_upload(file)
    resp = _post_to_colab(
        "/scene",
        image_path,
        {
            "hfov_deg": str(hfov_deg),
            "max_objects": str(max_objects),
            "plane_threshold": str(plane_threshold),
            "include_tier1_fallback": "1" if include_tier1_fallback else "0",
        },
        TIMEOUTS["tier2"],
    )
    return _record(record_id, "tier3", file.filename, image_name,
                   _decode_scene_response(resp, record_id))


@app.post("/scene/stepwise")
async def scene_stepwise(
    file: UploadFile = File(...),
    hfov_deg: float = Form(60.0),
    max_objects: int = Form(6),
    plane_threshold: float = Form(0.03),
    background_mode: str = Form("depth"),
):
    """Every stage of Tier 2 as its own inspectable artifact.

    Same pipeline as /scene/tier2, run once — this does not trade accuracy
    for visibility, it just keeps what each stage produced instead of
    discarding it once the next stage consumed it.
    """
    record_id, image_name, image_path = _save_upload(file)
    resp = _post_to_colab(
        "/scene/stepwise",
        image_path,
        {
            "hfov_deg": str(hfov_deg),
            "max_objects": str(max_objects),
            "plane_threshold": str(plane_threshold),
            "background_mode": background_mode,
        },
        TIMEOUTS["stepwise"],
    )
    return _record(record_id, "stepwise", file.filename, image_name,
                   _decode_stepwise_response(resp, record_id))


@app.delete("/history/{record_id}")
def delete_record(record_id: str):
    history = load_history()
    record = next((r for r in history if r["id"] == record_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail="No record with that id")

    for field, folder in (
        ("image_url", IMAGES_DIR),
        ("model_url", MODELS_DIR),
        ("depth_url", DEPTH_DIR),
        ("overlay_url", DEPTH_DIR),
    ):
        url = record.get(field)
        if not url:
            continue
        path = os.path.join(folder, os.path.basename(url))
        if os.path.exists(path):
            os.remove(path)

    # Stepwise records scatter many files (per-stage images, per-object
    # crops, per-object .glb) under one folder keyed by record id, rather
    # than a handful of named fields — remove the whole thing at once.
    stage_dir = os.path.join(STAGES_DIR, record_id)
    if os.path.isdir(stage_dir):
        shutil.rmtree(stage_dir, ignore_errors=True)

    save_history([r for r in history if r["id"] != record_id])
    return {"deleted": record_id}
