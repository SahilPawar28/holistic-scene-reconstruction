"""
Instance segmentation (SAM2) plus the filtering that turns raw masks into a
usable object list.

SAM2's automatic mask generator is class-agnostic and deliberately
over-segments: on a normal room photo it happily returns 60-120 masks
covering wall patches, the floor, a table, the table's legs separately, a
pot, the pot's rim, and the highlight on the pot. Only a handful of those
are "a distinct foreground object I should generate a 3D mesh for".

The model is a pretrained building block. Everything below `SAM2Segmenter`
— the area gating, the containment/duplicate suppression, the depth-based
background rejection, the background-mask construction — is the part that
decides what the scene actually contains, and it is where most of the
practical quality of Tier 2 comes from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

DEFAULT_SAM2_CHECKPOINT = "facebook/sam2.1-hiera-small"


@dataclass
class Instance:
    """One candidate foreground object."""

    id: int
    mask: np.ndarray  # (H, W) bool
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 — half-open, pixel coords
    area: int  # pixel count
    score: float  # SAM2's predicted IoU, or 1.0 for supplied masks
    depth_median: float = float("nan")
    depth_p10: float = float("nan")
    label: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def area_fraction(self) -> float:
        return self.area / float(self.mask.size)

    def crop_box(self, pad: float = 0.08, image_size: tuple[int, int] | None = None):
        """Bounding box padded by a fraction of its longest side.

        TripoSR wants a little breathing room around the object — a crop cut
        exactly to the silhouette makes its reconstructions bulge at the
        edges — so every downstream crop goes through here.
        """
        h, w = self.mask.shape[:2]
        if image_size is not None:
            w, h = image_size
        x0, y0, x1, y1 = self.bbox
        margin = int(round(max(x1 - x0, y1 - y0) * pad))
        return (
            max(0, x0 - margin),
            max(0, y0 - margin),
            min(w, x1 + margin),
            min(h, y1 + margin),
        )

    def touches_border(self, tolerance: int = 2) -> bool:
        h, w = self.mask.shape[:2]
        x0, y0, x1, y1 = self.bbox
        return (
            x0 <= tolerance
            or y0 <= tolerance
            or x1 >= w - tolerance
            or y1 >= h - tolerance
        )


@dataclass
class SegmentationParams:
    """Thresholds for turning raw SAM2 masks into an object list."""

    # A mask covering more than this much of the frame is structure (wall,
    # floor, ceiling) rather than an object, and belongs in the room shell.
    max_area_fraction: float = 0.22
    # Below this it is a texture detail, a highlight, or noise.
    min_area_fraction: float = 0.004
    # Absolute floor as well — TripoSR needs enough pixels to work with.
    min_area_pixels: int = 900

    # Two masks overlapping by more than this IoU are the same thing.
    duplicate_iou: float = 0.75
    # If this much of mask A lies inside mask B, A is a *part* of B (a table
    # leg inside a table). Keep the whole, drop the part.
    containment_ratio: float = 0.85

    # Objects whose 10th-percentile depth sits beyond this fraction of the
    # scene's depth range are background structure, not foreground objects.
    background_depth_quantile: float = 0.85

    # --- "is this actually a foreground object?" -----------------------
    #
    # Everything above is a size/duplication filter. None of it can tell a
    # pot apart from a patch of the wall behind it, and on real photos that
    # is the failure that matters: SAM2's automatic mode segments the whole
    # image, so without a positive test for objecthood the pipeline happily
    # sends "a strip of shelf" to TripoSR and gets a blob back.
    #
    # The decisive signal is depth relief: a foreground object stands out
    # from what surrounds it. Comparing the mask's own depth against a ring
    # of pixels just outside it separates "thing sitting in front of stuff"
    # from "region of the stuff". Measured as a fraction of the object's own
    # distance, so it works at 30cm and at 6m.
    min_depth_relief: float = 0.015   # must be >=1.5% nearer than its ring
    relief_ring_px: int = 30

    # Background regions tend to be sprawling and thin — a strip along the
    # top of the frame, an L of counter around a subject. Objects tend to
    # fill their own bounding box.
    min_fill_ratio: float = 0.25      # mask area / bbox area
    max_aspect_ratio: float = 5.0     # bbox long side / short side

    # A region touching three or four frame edges is the background, not an
    # object in it.
    max_border_edges: int = 2

    # Cap on how many objects go to per-object 3D generation. Each one is a
    # separate TripoSR call, so this is a wall-clock budget as much as a
    # quality filter.
    max_instances: int = 8

    # How to rank when more candidates survive than max_instances allows.
    #
    # "area" keeps the biggest, which is wrong for the photos people
    # actually take. A close-up of a coffee cup on a cafe table has a
    # subject occupying maybe 15% of the frame, while the table, the chair
    # behind it and a stranger's jeans are all larger. Ranking by area sends
    # the jeans to TripoSR and drops the cup.
    #
    # "objectness" ranks by how much a candidate behaves like the subject of
    # the photograph: standing out in depth, reasonably large, and near the
    # middle of the frame. People centre what they are photographing.
    rank_by: str = "objectness"

    sam_points_per_side: int = 24
    sam_pred_iou_thresh: float = 0.8
    sam_stability_score_thresh: float = 0.9


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Tight half-open bounding box of a boolean mask."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    if inter == 0:
        return 0.0
    union = np.count_nonzero(a | b)
    return inter / union


def containment(inner: np.ndarray, outer: np.ndarray) -> float:
    """Fraction of `inner` that lies inside `outer`."""
    area = np.count_nonzero(inner)
    if area == 0:
        return 0.0
    return np.count_nonzero(inner & outer) / area


def clean_mask(mask: np.ndarray, min_component_fraction: float = 0.15) -> np.ndarray:
    """Fill pinholes and drop stray disconnected specks from a mask.

    SAM2 masks are usually clean, but a mask with a scatter of isolated
    pixels across the frame blows up its own bounding box, which then makes
    the object crop mostly background and the TripoSR output mostly
    hallucination. Keeping only components that are a meaningful fraction of
    the largest one fixes that at negligible cost.

    Falls back to the input unchanged if OpenCV isn't available.
    """
    try:
        import cv2
    except ImportError:
        return mask

    m = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n_labels <= 2:
        return m.astype(bool)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = areas.max()
    keep_labels = 1 + np.nonzero(areas >= largest * min_component_fraction)[0]
    return np.isin(labels, keep_labels)


def depth_relief(
    mask: np.ndarray, depth: np.ndarray, ring_px: int = 30
) -> float:
    """How much nearer is this region than the ring of pixels around it?

    Returned in the same units as `depth`; positive means the region stands
    in front of its surroundings. NaN when it cannot be measured (no OpenCV,
    or not enough valid depth on either side).

    This is the test that separates an object from a piece of background. A
    pot on a table is nearer than the table and wall around it. A strip of
    shelf is not nearer than the shelf around it — it *is* the shelf. Size
    and shape filters cannot make that distinction; this can.

    The ring is taken just outside a slightly dilated mask, because
    segmentation boundaries sit a pixel or two inside the true silhouette
    and sampling flush against the mask edge picks up the object itself.
    """
    try:
        import cv2
    except ImportError:
        return float("nan")

    m = mask.astype(np.uint8)
    inner = cv2.dilate(m, np.ones((7, 7), np.uint8), iterations=1).astype(bool)
    outer = cv2.dilate(
        m, np.ones((ring_px * 2 + 1, ring_px * 2 + 1), np.uint8), iterations=1
    ).astype(bool)
    ring = outer & ~inner

    d_obj = depth[mask.astype(bool)]
    d_ring = depth[ring]
    d_obj = d_obj[np.isfinite(d_obj) & (d_obj > 0)]
    d_ring = d_ring[np.isfinite(d_ring) & (d_ring > 0)]
    if len(d_obj) < 20 or len(d_ring) < 20:
        return float("nan")

    return float(np.median(d_ring) - np.median(d_obj))


def fill_ratio(instance: Instance) -> float:
    """Mask area over bounding-box area. Compact objects score high."""
    x0, y0, x1, y1 = instance.bbox
    box = max((x1 - x0) * (y1 - y0), 1)
    return instance.area / box


def aspect_ratio(instance: Instance) -> float:
    x0, y0, x1, y1 = instance.bbox
    w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    return max(w, h) / min(w, h)


def border_edge_count(instance: Instance, tolerance: int = 3) -> int:
    """How many of the four frame edges this mask's bbox touches."""
    h, w = instance.mask.shape[:2]
    x0, y0, x1, y1 = instance.bbox
    return sum(
        [x0 <= tolerance, y0 <= tolerance, x1 >= w - tolerance, y1 >= h - tolerance]
    )


def looks_like_object(
    instance: Instance,
    depth: np.ndarray | None,
    params: SegmentationParams,
) -> tuple[bool, str]:
    """Positive test for objecthood. Returns (keep, reason_if_rejected).

    When a semantic label is available and confident it takes precedence
    over the geometric tests, because it is simply better information: depth
    relief and compactness are proxies for "is this a thing", whereas a
    model that has seen sofas can answer directly. The geometric tests stay
    as the fallback for regions the labeller is unsure about, and the
    frame-span test still applies either way — a mask covering the whole
    frame is not a placeable object however sofa-like it looks.
    """
    category = instance.meta.get("semantic_category")
    trusted = instance.meta.get("semantic_trusted")
    label = instance.meta.get("semantic_label", category)

    if trusted and category in ("structure", "person"):
        return False, f"recognised as {label!r} ({category}), not an object"

    edges = border_edge_count(instance)
    if edges > params.max_border_edges:
        return False, f"spans {edges} frame edges"

    fill = fill_ratio(instance)
    if fill < params.min_fill_ratio:
        return False, f"fill ratio {fill:.2f} (sprawling, not compact)"

    ar = aspect_ratio(instance)
    if ar > params.max_aspect_ratio:
        return False, f"aspect ratio {ar:.1f} (strip-like)"

    if trusted and category == "object":
        # Recognised as a real object: skip the geometric proxies, which
        # exist only to approximate this decision.
        return True, ""

    if depth is not None:
        relief = depth_relief(instance.mask, depth, params.relief_ring_px)
        instance.meta["depth_relief"] = relief
        if np.isfinite(relief) and np.isfinite(instance.depth_median):
            relative = relief / max(instance.depth_median, 1e-6)
            instance.meta["relative_relief"] = relative
            if relative < params.min_depth_relief:
                return False, (
                    f"no depth relief ({relative*100:.1f}% vs its surroundings) "
                    f"— this is background, not an object"
                )

    return True, ""


def objectness(instance: Instance) -> float:
    """How much does this look like the thing the photo is *of*?

    Three signals multiplied together, each in [0, 1]-ish:

      relief      how far it stands out in depth from its surroundings
      size        sqrt of area fraction — bigger is more likely the subject,
                  but sub-linearly, so a huge background region cannot win
                  on size alone
      centrality  people frame their subject near the middle

    Used only to rank candidates that have already passed every filter, so
    it decides which survivors get a TripoSR call, not what counts as an
    object at all.
    """
    relief = instance.meta.get("relative_relief")
    if relief is None or not np.isfinite(relief):
        relief = 0.05          # unknown: neutral, don't zero the whole score
    relief = float(np.clip(relief, 0.0, 1.0))

    size = float(np.sqrt(np.clip(instance.area_fraction, 0.0, 1.0)))

    h, w = instance.mask.shape[:2]
    x0, y0, x1, y1 = instance.bbox
    dx = ((x0 + x1) / 2.0 - w / 2.0) / max(w, 1)
    dy = ((y0 + y1) / 2.0 - h / 2.0) / max(h, 1)
    centrality = float(np.clip(1.0 - 2.0 * np.hypot(dx, dy), 0.0, 1.0))

    return relief * size * (0.4 + 0.6 * centrality)


def attach_depth_stats(instances: list[Instance], depth: np.ndarray) -> None:
    """Fill in each instance's depth statistics, in place."""
    for inst in instances:
        d = depth[inst.mask]
        d = d[np.isfinite(d) & (d > 0)]
        if len(d) == 0:
            continue
        inst.depth_median = float(np.median(d))
        inst.depth_p10 = float(np.percentile(d, 10))


def filter_instances(
    instances: list[Instance],
    depth: np.ndarray | None = None,
    params: SegmentationParams | None = None,
    rejections: list | None = None,
) -> list[Instance]:
    """Raw SAM2 masks -> the distinct foreground objects worth reconstructing.

    Applied in this order, cheapest and most decisive test first:

      1. area gate            — too big is structure, too small is noise
      2. depth gate           — sitting at background depth is structure
      3. objecthood           — compact, not frame-spanning, and standing
                                out in depth from its surroundings
      4. containment / IoU    — parts of an already-kept object, or dupes
      5. count cap            — keep the largest N

    Step 3 is the one that matters on real photos. Steps 1, 2, 4 and 5 are
    all size and redundancy filters — none of them can tell a pot from a
    patch of the wall behind it. Running without step 3 on a shop-shelf
    photo passed six background strips through as "objects", each of which
    then got its own TripoSR mesh and its own placement, producing a scene
    of floating blobs several metres tall.

    Order matters for step 4: instances are sorted largest-first, so a whole
    object is always considered before its own parts, and the part is what
    gets dropped.

    `rejections` collects (instance, reason) so the caller can report *why*
    a photo yielded no objects, which is otherwise very hard to debug.
    """
    params = params or SegmentationParams()
    if rejections is None:
        rejections = []

    kept: list[Instance] = []
    for inst in sorted(instances, key=lambda i: i.area, reverse=True):
        if inst.area < params.min_area_pixels:
            rejections.append((inst, f"only {inst.area}px"))
            continue
        frac = inst.area_fraction
        if frac > params.max_area_fraction:
            rejections.append((inst, f"covers {frac:.0%} of frame (structure)"))
            continue
        if frac < params.min_area_fraction:
            rejections.append((inst, f"covers {frac:.1%} of frame (noise)"))
            continue
        kept.append(inst)

    if depth is not None and kept:
        attach_depth_stats(kept, depth)
        finite = depth[np.isfinite(depth) & (depth > 0)]
        if len(finite):
            # Compare against the scene's own depth distribution rather than
            # an absolute distance, so the same threshold works for a
            # close-up on a desk and a wide shot down a hallway.
            cutoff = float(np.quantile(finite, params.background_depth_quantile))
            survivors_depth = []
            for i in kept:
                if not np.isfinite(i.depth_p10) or i.depth_p10 <= cutoff:
                    survivors_depth.append(i)
                else:
                    rejections.append((i, "sits at background depth"))
            kept = survivors_depth

    # --- objecthood ---------------------------------------------------
    passed = []
    for inst in kept:
        ok, reason = looks_like_object(inst, depth, params)
        if ok:
            passed.append(inst)
        else:
            rejections.append((inst, reason))
    kept = passed

    survivors: list[Instance] = []
    for inst in kept:
        redundant = False
        for other in survivors:
            if (
                containment(inst.mask, other.mask) >= params.containment_ratio
                or mask_iou(inst.mask, other.mask) >= params.duplicate_iou
            ):
                redundant = True
                break
        if not redundant:
            survivors.append(inst)

    if params.rank_by == "objectness":
        for inst in survivors:
            inst.meta["objectness"] = objectness(inst)
        ranked = sorted(survivors, key=lambda i: i.meta["objectness"], reverse=True)
    else:
        ranked = sorted(survivors, key=lambda i: i.area, reverse=True)

    for inst in ranked[params.max_instances:]:
        rejections.append((inst, "ranked below the cap on objects to generate"))
    survivors = ranked[: params.max_instances]

    for new_id, inst in enumerate(survivors):
        inst.id = new_id
    return survivors


def occupancy_mask(
    instances: list[Instance], shape: tuple[int, int], grow_px: int = 0
) -> np.ndarray:
    """Union of the instance masks, optionally grown or shrunk.

    `grow_px` is signed, and the sign matters depending on what the mask is
    for. Growing (positive) is right when excluding object pixels from a
    plane-fitting cloud: segmentation boundaries sit slightly inside the true
    silhouette, so a halo of object-edge pixels would otherwise leak in at
    object depth and drag the fit.

    Shrinking (negative) is right when cutting the object's hole out of the
    background mesh. The generated mesh that fills that hole never matches
    the silhouette exactly, so a hole cut flush — let alone dilated — leaves
    a black rim around every object. Cutting slightly inside lets the placed
    mesh cover the seam.
    """
    h, w = shape
    occupied = np.zeros((h, w), dtype=bool)
    for inst in instances:
        occupied |= inst.mask
    if grow_px == 0:
        return occupied
    try:
        import cv2
    except ImportError:
        return occupied
    k = np.ones((abs(grow_px) * 2 + 1,) * 2, np.uint8)
    op = cv2.dilate if grow_px > 0 else cv2.erode
    return op(occupied.astype(np.uint8), k, iterations=1).astype(bool)


def background_mask(
    instances: list[Instance], shape: tuple[int, int], dilate_px: int = 3
) -> np.ndarray:
    """Everything not claimed by an object — the room shell's input.

    Object masks are dilated before subtraction. Segmentation boundaries sit
    a pixel or two inside the true silhouette, so without dilation a halo of
    object-edge pixels leaks into the background cloud, and those pixels sit
    at object depth rather than wall depth — exactly the outliers that make
    RANSAC fit a plane through the middle of the room.
    """
    h, w = shape
    occupied = np.zeros((h, w), dtype=bool)
    for inst in instances:
        occupied |= inst.mask

    if dilate_px > 0:
        try:
            import cv2

            k = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
            occupied = cv2.dilate(occupied.astype(np.uint8), k, iterations=1).astype(bool)
        except ImportError:
            pass

    return ~occupied


class SAM2Segmenter:
    """Wrapper over SAM2's automatic mask generator (pretrained, as-is).

    Lazy-loading, and tolerant about which SAM2 distribution is installed:
    the `sam2` package from Meta and the `transformers` port expose the
    generator differently, and Colab's environment changes often enough that
    pinning to one import path is a reliability problem.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_SAM2_CHECKPOINT,
        device: str | None = None,
        params: SegmentationParams | None = None,
    ):
        self.checkpoint = checkpoint
        self.params = params or SegmentationParams()
        self._device = device
        self._generator = None

    @property
    def device(self) -> str:
        if self._device is None:
            try:
                import torch

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:  # pragma: no cover
                self._device = "cpu"
        return self._device

    @property
    def predictor(self):
        """Box-prompted SAM2ImagePredictor, for detection-seeded masks.

        SAM2AutomaticMaskGenerator already owns one of these internally to
        do its own point-prompted predictions, so this reuses it rather
        than loading a second copy of the same model.
        """
        self.load()
        return self._generator.predictor

    def load(self) -> "SAM2Segmenter":
        if self._generator is not None:
            return self
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2_hf

        model = build_sam2_hf(self.checkpoint, device=self.device)
        self._generator = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=self.params.sam_points_per_side,
            pred_iou_thresh=self.params.sam_pred_iou_thresh,
            stability_score_thresh=self.params.sam_stability_score_thresh,
            # SAM2 defaults post-process small regions away; we do our own
            # component cleanup in clean_mask, so leave the raw masks alone.
            min_mask_region_area=0,
        )
        return self

    def raw_masks(self, image: Image.Image) -> list[dict]:
        self.load()
        return self._generator.generate(np.asarray(image.convert("RGB")))

    def segment(
        self,
        image: Image.Image,
        depth: np.ndarray | None = None,
        rejections: list | None = None,
    ) -> list[Instance]:
        """Full path: SAM2 -> cleanup -> filtering -> object list."""
        raw = self.raw_masks(image)
        instances = []
        for i, m in enumerate(raw):
            mask = clean_mask(np.asarray(m["segmentation"], dtype=bool))
            area = int(np.count_nonzero(mask))
            if area == 0:
                continue
            instances.append(
                Instance(
                    id=i,
                    mask=mask,
                    bbox=mask_to_bbox(mask),
                    area=area,
                    score=float(m.get("predicted_iou", 1.0)),
                    meta={"stability_score": float(m.get("stability_score", 0.0))},
                )
            )
        return filter_instances(instances, depth=depth, params=self.params,
                                rejections=rejections)

    def unload(self) -> None:
        self._generator = None
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass


def instances_from_detections(
    image, detections: list, predictor, start_id: int = 0
) -> list:
    """GroundingDINO boxes -> precise SAM2 masks, one instance each.

    `predictor` is a SAM2ImagePredictor (box-prompted mode, distinct from
    the automatic mask generator's point-grid mode). A detected box gives
    SAM2 a strong hint about *where* to look, which is exactly the
    information the automatic pass lacks — this is how a chair the point
    grid never sampled still ends up with a precise silhouette.

    Each instance carries its GroundingDINO label as a trusted semantic
    result already, so it does not need CLIP relabelling: the detector
    already answered "what is this" by construction, more directly than a
    zero-shot crop classification would.
    """
    import numpy as np

    from .detection import is_ambiguous_label

    predictor.set_image(np.asarray(image.convert("RGB")))
    instances = []
    for i, det in enumerate(detections):
        masks, scores, _ = predictor.predict(
            box=np.array(det.box), multimask_output=False
        )
        mask = np.asarray(masks[0], dtype=bool)
        area = int(mask.sum())
        if area == 0:
            continue
        inst = Instance(
            id=start_id + i, mask=mask, bbox=mask_to_bbox(mask), area=area,
            score=float(scores[0]),
        )
        inst.label = det.label
        # A label spanning several unrelated vocabulary phrases means the
        # detector was not confident which single object this is — its box
        # is kept (it is still evidence something is here), but the label
        # is not trusted, so run_segmentation sends it to CLIP for a real
        # answer instead of displaying the garbled text.
        ambiguous = is_ambiguous_label(det.label)
        inst.meta.update(
            semantic_label=det.label,
            semantic_category="object",
            semantic_confidence=float(det.score),
            semantic_margin=1.0,
            semantic_trusted=not ambiguous,
            source="detection",
        )
        instances.append(inst)
    return instances


def merge_instances(
    automatic: list, detected: list, iou_threshold: float = 0.5
) -> list:
    """Combine automatic-mask and detection-seeded instances, deduplicated.

    Detection-seeded instances win when they overlap an automatic one
    (their label comes directly from a text query rather than a zero-shot
    guess on whatever odd shape the automatic mask happened to be), but the
    automatic mask's own geometry is not discarded unless the detector's
    silhouette is what actually survives — in practice both come from the
    same SAM2 model, so this mostly decides *labels*, not shapes.
    New detections with no matching automatic mask are appended outright,
    which is the entire point of running detection in the first place: it
    recovers objects the automatic pass never proposed at all.
    """
    merged = list(automatic)
    for det_inst in detected:
        best_iou, best_idx = 0.0, -1
        for idx, auto_inst in enumerate(merged):
            iou = mask_iou(det_inst.mask, auto_inst.mask)
            if iou > best_iou:
                best_iou, best_idx = iou, idx
        if best_iou >= iou_threshold:
            merged[best_idx].label = det_inst.label
            merged[best_idx].meta.update(det_inst.meta)
        else:
            merged.append(det_inst)

    for new_id, inst in enumerate(merged):
        inst.id = new_id
    return merged


def instances_from_masks(
    masks: list[np.ndarray], depth: np.ndarray | None = None
) -> list[Instance]:
    """Build instances from externally supplied masks (manual, or another
    segmenter). Lets the rest of the pipeline be tested without SAM2."""
    instances = []
    for i, m in enumerate(masks):
        mask = np.asarray(m, dtype=bool)
        area = int(np.count_nonzero(mask))
        if area == 0:
            continue
        instances.append(
            Instance(
                id=i, mask=mask, bbox=mask_to_bbox(mask), area=area, score=1.0
            )
        )
    if depth is not None:
        attach_depth_stats(instances, depth)
    return instances
