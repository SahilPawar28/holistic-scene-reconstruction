"""
Per-object 3D generation: turn each segmented instance into its own complete
mesh via TripoSR.

TripoSR is the same pretrained model the v1 single-object pipeline already
used; the only change is that it is now called once per detected object
instead of once per photo. What is new here is the preparation around it,
which matters more than it sounds:

  - **The mask replaces rembg.** v1 ran TripoSR's `remove_background`, which
    guesses the foreground with a general-purpose matting model. We already
    have an exact instance mask from SAM2, and using it is strictly better:
    rembg on a crop of a pot standing on a table frequently keeps a slab of
    table with it, and TripoSR then faithfully reconstructs a pot fused to a
    lump of wood.

  - **Occlusion is measured, not ignored.** An object whose mask runs off
    the frame or is cut into by a nearer object is one TripoSR will have to
    invent most of. That is not a reason to skip it, but the placement stage
    should trust its scale far less, so the fraction is recorded here.

  - **Framing is fixed and recorded.** TripoSR is sensitive to how much of
    the crop the object fills, so every crop is padded to a square and the
    object scaled to a constant fraction of it. That framing is invertible
    and stored, because the placement solver has to relate the generated
    mesh back to the pixels it came from.

Everything downstream of this file (the placement solver) is what turns
these canonical, unit-scale meshes into a scene.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io

import numpy as np
from PIL import Image

from .camera import PinholeCamera
from .segmentation import Instance

# Fraction of the square crop that the object should span. TripoSR's
# training data is framed roughly this tightly; filling the frame edge to
# edge makes it clip the object, and framing it too small wastes resolution.
FOREGROUND_FRACTION = 0.85

# Grey rather than white or transparent-to-black. TripoSR composites RGBA
# onto a constant background internally, and mid-grey biases the shading
# prior least — the same 0.5 the v1 notebook used.
BACKGROUND_GREY = 127


@dataclass
class ObjectCrop:
    """One object, prepared for TripoSR, with the framing kept invertible."""

    instance_id: int
    image: Image.Image  # square RGB, background flattened to grey
    rgba: Image.Image  # same crop with the mask as alpha (for debugging)
    source_bbox: tuple[int, int, int, int]  # crop box in full-image pixels
    crop_size: int  # side length of the square crop, in source pixels
    crop_origin: tuple[float, float]  # top-left of the square, source pixels
    occlusion: float  # 0 = fully visible, 1 = almost entirely hidden
    mask_pixels: int
    label: str | None = None
    meta: dict = field(default_factory=dict)

    def crop_to_source(self, xy: np.ndarray) -> np.ndarray:
        """Map coordinates in the square crop back to full-image pixels.

        The placement solver works in full-image pixel space, so anything
        derived from the crop has to come back through here.
        """
        xy = np.asarray(xy, dtype=np.float64)
        scale = self.crop_size / self.image.size[0]
        return xy * scale + np.asarray(self.crop_origin, dtype=np.float64)


@dataclass
class GeneratedObject:
    """A TripoSR mesh for one instance, in its own canonical frame."""

    instance_id: int
    mesh: object  # trimesh.Trimesh, centred at the origin, unit-scaled
    canonical_scale: float  # multiply by this to undo the normalisation
    canonical_center: np.ndarray  # add this back to undo the normalisation
    crop: ObjectCrop
    meta: dict = field(default_factory=dict)


def measure_occlusion(instance: Instance, depth: np.ndarray | None = None) -> float:
    """Rough estimate of how much of this object the photo cannot see.

    Two independent signals, combined by taking the worse:

      1. **Frame truncation** — how much of the mask's bounding box sits on
         the image border. An object half out of frame is half unseen.
      2. **Silhouette concavity** — the mask's area versus its convex hull's.
         A pot occluded by a book in front of it has a bite taken out of its
         silhouette, which shows up as a low area/hull ratio.

    Neither is exact, and they are not meant to be. The number is used to
    decide how much to trust an object's apparent size, not to make a
    geometric claim.
    """
    h, w = instance.mask.shape[:2]
    x0, y0, x1, y1 = instance.bbox
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)

    # How much of the bounding box perimeter is flush against the frame.
    touching = sum(
        [
            bh if x0 <= 1 else 0,
            bh if x1 >= w - 1 else 0,
            bw if y0 <= 1 else 0,
            bw if y1 >= h - 1 else 0,
        ]
    )
    truncation = min(1.0, touching / (2.0 * (bw + bh)))

    concavity = 0.0
    try:
        import cv2

        contours, _ = cv2.findContours(
            instance.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(biggest)
            hull_area = cv2.contourArea(cv2.convexHull(biggest))
            if hull_area > 1e-6:
                concavity = float(np.clip(1.0 - area / hull_area, 0.0, 1.0))
    except ImportError:
        pass

    return float(max(truncation, concavity))


def prepare_crop(
    image: Image.Image,
    instance: Instance,
    depth: np.ndarray | None = None,
    output_size: int = 512,
    foreground_fraction: float = FOREGROUND_FRACTION,
    feather_px: int = 2,
) -> ObjectCrop:
    """Cut one object out of the photo and frame it the way TripoSR wants.

    The crop is square and centred on the object, with the object scaled to
    `foreground_fraction` of the frame. Everything outside the mask is
    replaced with flat grey — not left as photo background, which TripoSR
    would try to reconstruct as part of the object.

    The mask edge is feathered by a couple of pixels. Segmentation
    boundaries are hard and pixel-exact; a hard cut leaves a ring of
    high-frequency edge that TripoSR reads as real geometry and turns into a
    thin rim around the object.
    """
    src = image.convert("RGB")
    w, h = src.size
    if instance.mask.shape[:2] != (h, w):
        raise ValueError(
            f"instance mask {instance.mask.shape[:2]} does not match image {(h, w)}"
        )

    x0, y0, x1, y1 = instance.bbox
    obj_w, obj_h = max(x1 - x0, 1), max(y1 - y0, 1)

    # Square window centred on the object, sized so the object occupies the
    # requested fraction of it. Allowed to run off the image — the region
    # outside is grey anyway, and keeping the window square-and-centred is
    # what makes the framing invertible.
    side = max(obj_w, obj_h) / foreground_fraction
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    ox, oy = cx - side / 2.0, cy - side / 2.0

    alpha = instance.mask.astype(np.float32)
    if feather_px > 0:
        try:
            import cv2

            k = feather_px * 2 + 1
            alpha = cv2.GaussianBlur(alpha, (k, k), 0)
        except ImportError:
            pass

    rgb = np.asarray(src, dtype=np.float32)
    composited = rgb * alpha[..., None] + BACKGROUND_GREY * (1.0 - alpha[..., None])
    composited = np.clip(composited, 0, 255).astype(np.uint8)

    full = Image.fromarray(composited)
    rgba_full = Image.fromarray(
        np.dstack([np.asarray(src, dtype=np.uint8), (alpha * 255).astype(np.uint8)])
    )

    box = (int(round(ox)), int(round(oy)), int(round(ox + side)), int(round(oy + side)))
    # `Image.crop` pads out-of-bounds regions with black, so paste onto a
    # grey canvas instead to keep the background constant.
    canvas = Image.new("RGB", (box[2] - box[0], box[3] - box[1]),
                       (BACKGROUND_GREY,) * 3)
    canvas.paste(full.crop(box), (0, 0))
    square = canvas.resize((output_size, output_size), Image.LANCZOS)

    rgba_canvas = Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), (0, 0, 0, 0))
    rgba_canvas.paste(rgba_full.crop(box), (0, 0))

    return ObjectCrop(
        instance_id=instance.id,
        image=square,
        rgba=rgba_canvas.resize((output_size, output_size), Image.LANCZOS),
        source_bbox=(x0, y0, x1, y1),
        crop_size=box[2] - box[0],
        crop_origin=(float(box[0]), float(box[1])),
        occlusion=measure_occlusion(instance, depth),
        mask_pixels=instance.area,
        label=instance.label,
        meta={"depth_median": instance.depth_median},
    )


def prepare_crops(
    image: Image.Image,
    instances: list[Instance],
    depth: np.ndarray | None = None,
    output_size: int = 512,
) -> list[ObjectCrop]:
    return [prepare_crop(image, i, depth, output_size) for i in instances]


def canonicalize_mesh(mesh, target_extent: float = 1.0):
    """Centre a generated mesh on its origin and scale it to unit size.

    TripoSR emits meshes in its own normalised frame, and the exact scale it
    picks varies with framing. Normalising here means the placement solver
    starts from a known state and only has to solve for one scale factor
    rather than untangling two.

    Returns (mesh, scale, center) such that applying `center` then `scale`
    to the returned mesh reproduces the original.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2.0
    extent = float(np.max(verts.max(axis=0) - verts.min(axis=0)))
    if extent < 1e-9:
        extent = 1.0
    scale = target_extent / extent

    out = mesh.copy()
    out.vertices = ((verts - center) * scale).astype(np.float32)
    return out, 1.0 / scale, center


class TripoSRClient:
    """Calls the per-object generation endpoint on the Colab server.

    Deliberately a client rather than an in-process model: TripoSR needs a
    GPU this laptop does not have, and the notebook already holds the
    weights loaded for the whole session, so per-object calls cost only
    inference time and not a reload each.
    """

    def __init__(self, server_url: str, timeout: int = 300):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def generate(self, crop: ObjectCrop, resolution: int = 256):
        """One crop -> one trimesh. Raises on transport or server failure."""
        import requests
        import trimesh

        buf = io.BytesIO()
        crop.image.save(buf, format="PNG")
        buf.seek(0)

        response = requests.post(
            f"{self.server_url}/object",
            files={"file": (f"object_{crop.instance_id}.png", buf, "image/png")},
            data={"resolution": str(resolution), "skip_background_removal": "1"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return trimesh.load(io.BytesIO(response.content), file_type="glb", force="mesh")

    def generate_all(
        self, crops: list[ObjectCrop], resolution: int = 256, on_error: str = "skip"
    ) -> list[GeneratedObject]:
        """Generate every crop, keeping going when one fails.

        A single object failing (TripoSR occasionally returns an empty mesh
        for a heavily occluded crop) should not cost the whole scene, so the
        default is to skip it and record why.
        """
        results: list[GeneratedObject] = []
        for crop in crops:
            try:
                raw = self.generate(crop, resolution=resolution)
                if raw is None or len(raw.faces) == 0:
                    raise RuntimeError("generator returned an empty mesh")
                mesh, scale, center = canonicalize_mesh(raw)
                results.append(
                    GeneratedObject(
                        instance_id=crop.instance_id,
                        mesh=mesh,
                        canonical_scale=scale,
                        canonical_center=center,
                        crop=crop,
                        meta={"faces": int(len(mesh.faces))},
                    )
                )
            except Exception as exc:
                if on_error == "raise":
                    raise
                results.append(
                    GeneratedObject(
                        instance_id=crop.instance_id,
                        mesh=None,
                        canonical_scale=1.0,
                        canonical_center=np.zeros(3),
                        crop=crop,
                        meta={"error": f"{type(exc).__name__}: {exc}"},
                    )
                )
        return results


def object_target_cloud(
    instance: Instance,
    depth: np.ndarray,
    camera: PinholeCamera,
    max_points: int = 8000,
) -> np.ndarray:
    """Where this object actually sits in space, according to the photo.

    Thin wrapper over `meshing.backproject_mask`, named for the role it
    plays: this is the target the placement solver aligns a generated mesh
    against. Kept here so the object stages read in one place.
    """
    from .meshing import backproject_mask

    return backproject_mask(instance.mask, depth, camera, max_points=max_points)
