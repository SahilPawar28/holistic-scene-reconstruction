# Single-Image Holistic Scene Reconstruction

One photo in, one 3D scene out — not just the main object, but the room:
floor, walls, and every distinct piece of furniture, each as its own mesh,
positioned relative to the others the way the photograph shows them.

This is the successor to [image-to-3d](../image-to-3d), which did
single-object reconstruction with TripoSR. That version answered "turn this
object into a mesh". This one answers "turn this *photograph* into a scene".

---

## Where this currently stands

| Stage | Status |
|---|---|
| Camera model + projection math | **done**, checked against ground truth |
| Depth (Depth Anything V2) + relative-depth handling | **done** |
| Tier 1 — whole-scene textured mesh | **done**, end to end |
| Instance segmentation (SAM2) + object filtering | **done** |
| Room shell — RANSAC plane fitting | **done**, checked against ground truth |
| Per-object generation (TripoSR, batched) | **done** |
| **Placement / assembly solver** | **done**, checked against ground truth |
| Scene composition → single .glb | **done** |
| Baseline comparison writeup | not started |

Every stage is implemented. Tier 1 is verified end to end on real photos;
Tier 2's geometry — shell fitting, placement, composition — is verified
against ground truth offline, but has **not yet been run with real SAM2 and
TripoSR output**, because that needs the Colab GPU session. That is the
remaining unknown.

Tier 2's geometry can be seen working without a GPU:

```bash
python scripts/seed_demo.py --tier2
```

This runs the real shell fitter, the real placement solver and the real
composition on the synthetic room, substituting ground-truth box meshes for
TripoSR. The resulting record is labelled as such in its own stats.

---

## Architecture

```
                    ┌──────────────────────────────────────────┐
   photo ──────────▶│  Depth Anything V2   →  depth map        │
                    │  SAM2                →  instance masks   │  Colab T4
                    │  TripoSR             →  per-object mesh  │  (pretrained)
                    └──────────────────────────────────────────┘
                                     │
                    ┌────────────────▼─────────────────────────┐
                    │  pinhole camera model  (camera.py)       │
                    │  edge-aware meshing    (meshing.py)      │  own code
                    │  RANSAC plane fitting  (room_shell.py)   │
                    │  object crop framing   (objects.py)      │
                    │  placement solver      (assembly.py) ◀── the hard part
                    └────────────────┬─────────────────────────┘
                                     ▼
                              one .glb scene
```

The split is deliberate and it is the point of the project. The pretrained
models are used as perception and generation building blocks. Everything
that turns their outputs into a coherent, correctly-scaled, correctly-placed
3D scene is code written here — and it is tested against ground truth rather
than eyeballed.

### Repository layout

```
pipeline/
  camera.py         pinhole model — the single source of truth for 2D ↔ 3D
  depth.py          Depth Anything V2 wrapper + disparity → depth conversion
  meshing.py        Tier 1: depth map → textured mesh, with edge-aware culling
  segmentation.py   SAM2 wrapper + the filtering that yields distinct objects
  room_shell.py     RANSAC plane fitting → floor/walls/ceiling mesh
  objects.py        per-object crop framing + TripoSR client
  assembly.py       the placement solver — the core of the project
  scene_compose.py  shell + placed objects → one .glb scene graph
  synthetic.py      ray-traced ground-truth room for testing without a GPU

scripts/
  selftest_geometry.py    ground-truth checks: camera, depth, Tier 1 meshing
  selftest_room_shell.py  ground-truth checks: RANSAC, plane classification
  selftest_assembly.py    ground-truth checks: solvers, placement, composition
  analyze_holes.py        why Tier 1 has gaps (occlusion vs over-culling)
  seed_demo.py            run Tier 1 offline and seed the backend's history
  build_notebook.py       generates colab/scene_pipeline.ipynb

backend/main.py             FastAPI proxy + persistence
frontend/index.html         three.js viewer (multi-mesh, photo view, stats)
colab/scene_pipeline.ipynb  all three models in one GPU session
colab/triposr_server_v1.ipynb  the v1 notebook, unchanged, for comparison
```

---

## The parts worth reading

Six decisions in here are load-bearing, and each one came from something
that actually broke. The measurements are from the ground-truth self-tests.

**1. Edge-aware culling** (`meshing.py`). Naively triangulating a depth map
welds every object to the background behind it, producing stretched "skin".
Two independent tests drop those triangles: a relative-depth-jump test, and
a grazing-angle test on the triangle normal. On the test scene this cuts the
longest triangle edge from 2.33 m to 0.27 m. Toggle **Wireframe** in the
viewer to see it.

**2. Relative depth is inverse depth** (`depth.py`). Depth Anything's
relative checkpoints emit affine-invariant *disparity*, not depth. Feeding
that straight into an unprojection turns the scene inside out. The
conversion interpolates in inverse-depth space, not depth space —
interpolating linearly in depth visibly bends flat walls.

**3. RANSAC inliers must agree in orientation, not just position**
(`room_shell.py`). Scoring planes purely by inlier count has a failure mode
that silently ruins the room fit: a plane tilted to *graze* a large surface
sweeps a band across it and collects thousands of inliers. On the test room
this produced a "floor" made entirely of back-wall pixels, tilted 20° and
13 cm out of place. Requiring each inlier's own PCA-estimated surface normal
to agree with the plane's fixed it — under 1% depth noise the floor went
from 16.4° / 13 cm off to **0.12° / 8 mm** off.

**4. The floor is found deliberately, not hoped for** (`room_shell.py`). A
ceiling is exactly as horizontal as a floor and usually less occluded, so a
plain "find a horizontal plane" pass returns the ceiling. The floor pass
restricts itself to points below the camera and retries past non-floor
horizontal surfaces such as table tops. This matters because the placement
step snaps objects to the floor plane — a wrong floor puts the whole scene
underground.

**5. The placement solver must know which parts of the mesh are visible**
(`assembly.py`). This is the single most important decision in the project.
The target cloud is only the object's *front* surface; the generated mesh is
a complete closed object. Register them naively and the mesh's back-side
vertices claim target points as nearest neighbours, dragging the object
forward and shrinking it until its back sits on the observed front. So every
ICP iteration re-computes visibility — front-facing test, z-buffer occlusion
test, and in-frame test — and only visible vertices participate.

Turning that off is the clearest demonstration of what it buys:

| | scale error with | without |
|---|---|---|
| table | 7.5% | 25.9% |
| pot | 13.5% | 40.6% |
| crate | 4.7% | 10.0% |

**6. The two Chamfer directions must be trimmed differently** (`assembly.py`).
Trimmed ICP discards the worst correspondences as outliers. Trimming both
directions equally biased *every* object about 10% too small, and it took a
sweep of the objective landscape to see why: the peripheral target points
near an object's silhouette are exactly the ones carrying the size signal, so
discarding them makes shrinking free. A target point is a real observation
that the mesh is obliged to explain (trim 5%); a visible mesh vertex may
legitimately have no observation (trim 20%). That asymmetry moved the
objective's minimum from 90% of true scale to 96-100%.

A related point worth stating, because it is a genuine limitation rather
than a fix: when an object runs off the edge of the frame, the visible patch
stops constraining its extent at all, and no objective can recover what was
never observed. The solver handles this by blending the fitted scale toward
the mask's angular size, weighted by how occluded the object is — two
estimators with different failure modes, combined by reliability. On the
frame-truncated table that took scale error from 12.1% to 7.5%.

### What the solver actually achieves

Against exact ground truth on the synthetic room, with a stand-in mesh of the
true shape (so this measures the solver, not the generator):

| object | true size | scale error | centre error | 90% of mesh within |
|---|---|---|---|---|
| table (truncated by frame) | 1.40 m | 7.5% | 5.8 cm | 5.9% of its size |
| pot (on the table) | 0.28 m | 13.5% | 2.7 cm | 13.6% |
| crate | 0.56 m | 4.7% | 3.7 cm | 5.6% |

A known yaw offset is recovered to **0.0° and 0.1°**. Support snapping puts
the table and crate on the floor and the pot **on the table** — not on the
floor, which is what a naive "snap everything to the floor plane" would do to
the brief's own example scene.

---

## Testing without a GPU

`pipeline/synthetic.py` ray-traces a room from exact planes and boxes and
returns a perfect depth map, colour image, per-object masks, **and the
ground-truth geometry**. That turns "does this look right?" into "is this
number correct?", which is the only way the geometry was debuggable at all
given that every neural stage is slow, non-deterministic, and Colab-only.

```bash
python scripts/selftest_geometry.py     # camera, depth conversion, Tier 1
python scripts/selftest_room_shell.py   # RANSAC, plane classification
python scripts/selftest_assembly.py     # closed-form solvers, placement, export
```

All three run on CPU — the first two in seconds, the assembly suite in a
couple of minutes — and every check compares against a known value — that the
floor plane comes back at exactly 1.400 m, that project/unproject round-trips
to 1e-13 px, that a tilted camera's up-direction is recovered to 0.02°
where assuming +Y would be 12° wrong. Each also runs a noisy variant, because
thresholds tuned on perfect depth do not survive contact with a real
monocular depth map.

The assembly suite also runs its key design decisions as **ablations**
rather than just asserting them: visibility filtering on vs off, and
scale-then-rotation vs rotating from the start. A design decision that
cannot be shown to matter is not worth defending in an interview.

---

## Setup

### 1. Model server (Colab)

Open `colab/scene_pipeline.ipynb` in Google Colab.

- `Runtime` → `Change runtime type` → **T4 GPU**
- Run cell 1, then **`Runtime` → `Restart session`** (required once — TripoSR
  pins `numpy<2` while Colab ships packages built against numpy 2)
- Set `REPO_URL` in cell 2 to this repository, then run cells 2–6
- Cell 6 prints an ngrok URL. Leave it running.

Free ngrok token: <https://dashboard.ngrok.com/get-started/your-authtoken>

**One notebook, not two.** Free Colab allows a single GPU session, so Depth
Anything V2, SAM2 and TripoSR are loaded together (~4 GB of a 15 GB T4). That
also means each model loads once and is reused across every object in a
scene, rather than paying the load cost per call.

### 2. Backend

```bash
cd backend && pip install -r requirements.txt && uvicorn main:app --reload --port 8000
```

### 3. Frontend

```bash
cd frontend && python -m http.server 8080
```

Open <http://127.0.0.1:8080>, paste the ngrok URL into the server field at
the top of the sidebar and press **Set** — the indicator dot goes green when
the Colab server answers. The URL changes every time the notebook restarts,
which is why it is settable from the UI rather than hard-coded (it was a
constant in v1, and editing a source file every session got old fast).

### Offline

To develop the UI, or demo Tier 1, with no Colab session at all:

```bash
python scripts/seed_demo.py                    # synthetic room
python scripts/seed_demo.py --image photo.jpg  # a real photo (needs torch + transformers)
```

This runs the real Tier 1 pipeline and writes a real record — mesh, depth
preview, mask overlay, per-stage stats — into the backend's history.

---

## The viewer

- **Photo view** puts the camera exactly where the photographer stood: the
  origin, looking down −Z, at the field of view the reconstruction was built
  with. From there the render matches the original photo, and dragging away
  from it is what makes the 3D real to a viewer.
- **Wireframe** shows the triangulation and, more usefully, the holes left
  where culling removed background-bridging triangles.
- **Stats** shows per-stage diagnostics — how many triangles each cull
  removed, how many planes were fitted and with what RMS error, how long each
  object took to generate. A bad result should be attributable to a stage,
  not guessed at.
- **Photo / Depth / Masks** tabs show what each perception stage actually saw.

---

## Known limitations — stated, not hidden

**Single-image reconstruction is ill-posed.** This is the honest headline.

- **Tier 1 reconstructs only visible surfaces.** The back of every object,
  everything behind anything else, and everything outside the frame is simply
  absent. That is a property of the problem, not a gap in the implementation:
  the information is not in the photograph. Orbit far enough around a Tier 1
  scene and you are looking at the back of a shell.

  This is the limitation people notice first, as black gaps that open up
  around every object the moment you orbit away from the photo viewpoint.
  It is worth being able to show that those gaps are *occlusion*, not a
  meshing bug — `scripts/analyze_holes.py` measures exactly that. Across
  three real photos (museum display case, cluttered desk, shop shelving):

  | | holes at default | holes with culling **fully disabled** |
  |---|---|---|
  | museum jug | 2.6% of pixels | 2.5% |
  | desk | 1.6% | 1.5% |
  | shop shelf | 1.7% | 1.6% |

  Turning edge-aware culling completely off recovers ~0.1% of pixels. The
  gaps are not over-culling — there is no surface there to keep. The hole
  map (`assets/diagnostics/*_holes.png`) shows the culled pixels as a thin
  outline tracing each object's silhouette, which is precisely where an
  occlusion boundary should be.

  They *look* far bigger than 2% because a culled triangle spans a depth
  jump over about one pixel: seen head-on it is a sliver of near-zero area,
  but rotate the camera by θ and it opens into a gap roughly
  `depth_jump × sin(θ)` wide. With a median depth discontinuity of 0.2 m,
  a 30° orbit turns each sliver into an ~8 cm gap. Small culled fraction,
  large black region — those are consistent, not contradictory.
- **Tier 2 fills those gaps by inventing them.** TripoSR generates each
  object's unseen sides from a prior learned over ~800,000 3D models.
  Plausible and often convincing — but a guess, not a measurement. The
  distinction between Tier 1's "absent" and Tier 2's "invented" is worth
  being precise about.
- **Scale is ambiguous.** Without known intrinsics, a small nearby object and
  a large distant one produce the same photo. The pipeline holds one
  estimated focal length constant across the scene, which keeps objects
  consistent *relative to each other* without claiming real-world
  measurements. EXIF focal length is used when the photo carries it. The
  metric-indoor depth checkpoint reduces but does not eliminate this.
- **The room shell paints occluded background onto the walls.** A wall
  polygon covers the region an object was hiding, so that object's pixels get
  textured onto the wall behind it. Tier 2 then places a real 3D object in
  front of the smear. Inpainting occluded background is a separate problem
  and is out of scope.
- **Room shells are convex hulls / bounding rectangles of fitted planes**, so
  L-shaped rooms, alcoves and doorways come out wrong. Windows and mirrors
  break the depth model outright.
- **Placement degrades with occlusion, and cannot do otherwise.** An
  object's scale is only constrained by the part of it the camera saw. The
  solver reports per-object `coverage`, `rms_error` and `occlusion` for
  exactly this reason — a placement with low coverage should be read as a
  guess, not a measurement.
- **Rotation is the weakest axis.** It is constrained to yaw by default,
  because a single view constrains the other two rotational degrees of
  freedom worst and nearly everything in a room stands upright. Objects
  lying on their side will be placed upright.
- **Cluttered scenes are much harder than clean ones.** A single object on a
  table against a plain wall is the case this is scoped to. Segmentation
  merges touching objects; heavily occluded objects produce unreliable target
  clouds; thin structures (chair legs, plant stems) survive neither the depth
  model nor the culling thresholds.

---

## Context: how this compares to published work

The staged "depth + segmentation → per-object generation → custom placement"
architecture mirrors how current published systems approach single-view
holistic reconstruction — Gen3DSR, DeepPriorAssembly (NeurIPS 2024),
3D-RE-GEN, MIDI-3D, SceneGen. Those systems need 16–30 GB of VRAM and are
research-grade installs, so none of them run on free Colab. This is a
scoped-down implementation of the same architecture, built to fit a free T4,
with the assembly logic written here rather than imported.

Reference points worth comparing output against (not yet done — see status
table): MIDI-3D's Hugging Face Space as a SOTA compositional baseline, and
Immersity AI as the naive "whole-image depth pop-out" baseline that Tier 1
should be judged against.

---

## Credits

- [TripoSR](https://github.com/VAST-AI-Research/TripoSR) — Stability AI / Tripo
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [SAM2](https://github.com/facebookresearch/sam2) — Meta
- [DeepPriorAssembly](https://github.com/junshengzhou/DeepPriorAssembly) —
  reference for the placement approach
- [3D-RE-GEN](https://github.com/cgtuebingen/3D-RE-GEN) — architectural reference
