"""
Open-vocabulary object detection: find objects SAM2's automatic mode misses.

SAM2's automatic mask generator proposes regions from a grid of sample
points, with no idea what it is looking at. It works well for objects that
are large, well-separated and high-contrast against their surroundings, and
misses ones that are thin, low-contrast, or sit in a spot the point grid
happened to skip. A wooden chair against a similarly-toned wooden floor is
exactly the case it drops — nothing in the grid ever landed a point on it
with enough local contrast to seed a mask.

GroundingDINO takes a different approach: given a list of text phrases
("a chair. a bed. a lamp."), it directly proposes bounding boxes for each
phrase it finds evidence of, independent of any point grid. It cannot
recover a mask that isn't there in the image, but it does not depend on
sampling luck either — it is *asking* for a chair rather than *hoping* to
land a point on one.

So detection and automatic segmentation are complementary, not competing:
automatic masks handle the general case cheaply; text-prompted detection
recovers specific named things the point grid missed. Both feed into SAM2
in the end — a detected box becomes a mask via SAM2's box-prompted mode,
the same underlying model the automatic path already uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

DEFAULT_CHECKPOINT = "IDEA-Research/grounding-dino-tiny"

# Reuses the object half of semantic.py's vocabulary rather than maintaining
# a second list that could drift out of sync with it. GroundingDINO expects
# phrases separated by ". " with a trailing period — its own documented
# prompt format, distinct from CLIP's "a photo of {}" template.
def _default_prompt() -> str:
    from .semantic import DEFAULT_VOCABULARY

    nouns = [text for text, category in DEFAULT_VOCABULARY if category == "object"]
    return ". ".join(nouns) + "."


@dataclass
class DetectedBox:
    box: tuple[float, float, float, float]  # x0, y0, x1, y1, pixel coords
    label: str
    score: float


@dataclass
class DetectionParams:
    checkpoint: str = DEFAULT_CHECKPOINT
    box_threshold: float = 0.30   # minimum box confidence to keep
    text_threshold: float = 0.25  # minimum per-token confidence for the label
    prompt: str | None = None     # None -> the default object vocabulary
    nms_iou: float = 0.5          # dedup boxes overlapping more than this


class GroundingDinoDetector:
    """Text-prompted object detector, lazy-loading like the other models."""

    def __init__(self, params: DetectionParams | None = None, device: str | None = None):
        self.params = params or DetectionParams()
        self._device = device
        self._model = None
        self._processor = None

    @property
    def device(self) -> str:
        if self._device is None:
            try:
                import torch

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:  # pragma: no cover
                self._device = "cpu"
        return self._device

    def load(self) -> "GroundingDinoDetector":
        if self._model is not None:
            return self
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.params.checkpoint)
        self._model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(self.params.checkpoint)
            .to(self.device)
            .eval()
        )
        return self

    def detect(self, image: Image.Image, prompt: str | None = None) -> list[DetectedBox]:
        """Text-prompted boxes for one image.

        `prompt` overrides the default object vocabulary; pass a custom one
        (e.g. from a user hint, or CLIP's own vocabulary) when the default
        general furniture/household list is not what the photo needs.
        """
        import torch

        self.load()
        text = prompt or self.params.prompt or _default_prompt()
        rgb = image.convert("RGB")

        inputs = self._processor(images=rgb, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._postprocess(outputs, inputs["input_ids"], rgb.size)

        boxes = np.asarray(results["boxes"].cpu(), dtype=np.float64)
        scores = np.asarray(results["scores"].cpu(), dtype=np.float64)
        labels = results.get("labels") or results.get("text_labels") or [""] * len(boxes)

        detections = [
            DetectedBox(box=tuple(b), label=str(l).strip(), score=float(s))
            for b, l, s in zip(boxes, labels, scores)
        ]
        return deduplicate_boxes(detections, self.params.nms_iou)

    def _postprocess(self, outputs, input_ids, image_size):
        """Call the box-post-processor across transformers' renamed kwarg.

        The confidence threshold parameter is named `box_threshold` in the
        transformers version this notebook pins (4.45-4.x) and `threshold`
        from a later release onward (a dev machine may have either). Same
        situation as the CLIP feature-output shim in semantic.py: introspect
        rather than guess, so this keeps working whichever lands underneath.
        """
        import inspect

        fn = self._processor.post_process_grounded_object_detection
        params = inspect.signature(fn).parameters
        threshold_kw = "box_threshold" if "box_threshold" in params else "threshold"

        kwargs = {
            threshold_kw: self.params.box_threshold,
            "text_threshold": self.params.text_threshold,
            "target_sizes": [image_size[::-1]],  # (height, width)
        }
        return fn(outputs, input_ids, **kwargs)[0]

    def unload(self) -> None:
        self._model = None
        self._processor = None
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass


def _iou(a: tuple, b: tuple) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = max(ax1 - ax0, 0.0) * max(ay1 - ay0, 0.0)
    area_b = max(bx1 - bx0, 0.0) * max(by1 - by0, 0.0)
    return inter / max(area_a + area_b - inter, 1e-9)


def deduplicate_boxes(boxes: list[DetectedBox], iou_threshold: float = 0.5) -> list[DetectedBox]:
    """Greedy NMS: highest-scoring box wins, suppress overlapping others.

    GroundingDINO's phrase grounding routinely returns several boxes for the
    same physical object — one per matched vocabulary phrase, sometimes with
    the phrases run together ("a television a computer monitor" for one TV,
    seen in testing). Text-level dedup cannot fix that, since the labels
    genuinely differ string-wise for the same box; box-overlap is what
    actually identifies them as the same detection.
    """
    ordered = sorted(boxes, key=lambda b: b.score, reverse=True)
    kept: list[DetectedBox] = []
    for candidate in ordered:
        if all(_iou(candidate.box, k.box) < iou_threshold for k in kept):
            kept.append(candidate)
    return kept


def _object_nouns() -> list[str]:
    from .semantic import DEFAULT_VOCABULARY

    return [text for text, category in DEFAULT_VOCABULARY if category == "object"]


def is_ambiguous_label(label: str, vocabulary: list[str] | None = None) -> bool:
    """True when a detection's label spans more than one distinct object.

    GroundingDINO's phrase grounding sometimes fuses several matched
    vocabulary phrases into one run-on string for a single box — observed
    on a real photo as "a sofa an armchair a wooden chair a bed" for what
    was, physically, one piece of furniture. When that happens the box
    itself is usually poorly localised too: the model was not confident
    enough about *which* object this is to commit to one phrase, and a
    label spanning four unrelated categories is not something the rest of
    the pipeline should treat as ground truth. Better to mark it untrusted
    and let CLIP's crop classification (which looks at the actual pixels,
    not just the token span) settle it instead.
    """
    vocabulary = vocabulary or _object_nouns()
    text = label.lower()
    matches = sum(1 for noun in vocabulary if noun.lower().strip(". ") in text)
    return matches > 1


def clip_box(box: tuple[float, float, float, float], width: int, height: int):
    x0, y0, x1, y1 = box
    return (
        max(0.0, min(x0, width)), max(0.0, min(y0, height)),
        max(0.0, min(x1, width)), max(0.0, min(y1, height)),
    )
