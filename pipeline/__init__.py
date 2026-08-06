"""
Single-image holistic scene reconstruction.

Stage order (see README):

    camera.py        pinhole model — the one source of truth for 2D <-> 3D
    depth.py         Depth Anything V2 + relative-depth handling
    meshing.py       Tier 1: depth map -> one textured scene mesh
    segmentation.py  SAM2 + the filtering that yields distinct objects
    room_shell.py    RANSAC plane fitting -> walls/floor/ceiling mesh
    objects.py       per-object crops -> TripoSR meshes
    assembly.py      placement solver: generated mesh -> its place in the scene
    scene_compose.py shell + placed objects -> one .glb

`synthetic.py` renders an exact ground-truth room for testing any of the
above without a GPU.
"""

__version__ = "0.1.0"
