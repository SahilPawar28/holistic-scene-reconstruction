"""
Semantic labelling of segmentation masks: what *is* each region?

Everything the pipeline currently uses to decide "is this an object" is
geometric — depth relief, compactness, how much of the frame it spans. Those
are proxies. They work because objects tend to stand out from their
surroundings, but they cannot actually tell a sofa from the wall behind it;
they can only tell "nearer and more compact" from "flatter and larger". On a
cluttered photo that distinction gets thin, and background regions slip
through as objects.

This module replaces the proxy with the real thing: ask a model that has
actually seen sofas. Each mask is cropped and scored against a vocabulary of
structure terms ("a wall", "the floor") and object terms ("a sofa", "a
lamp") using CLIP, which gives a label and a confidence per region.

## Why CLIP rather than a vision-language model over an API

A VLM asked to describe numbered regions is the obvious approach and it does
work, but for *this* job CLIP wins on every axis that matters here:

  - it runs locally in the same Colab session, so no per-region HTTP round
    trip (a dozen regions at 2-5s each dominates the whole pipeline)
  - the output is a score vector, not prose that has to be parsed and
    error-handled
  - it is small enough to sit alongside the three models already loaded
  - no rate limits, no key, no network dependency mid-reconstruction

A VLM remains the better tool for something CLIP genuinely cannot do:
relational reasoning across the whole scene ("the cushion is on the sofa"),
which is useful for support inference. That is a separate question from
per-region labelling, and is left as a hook rather than assumed.

## Why the vocabulary is a list of (text, category) pairs

Classifying "structure vs object" directly with two prompts works badly —
CLIP's embedding of the word "structure" has little to do with a photograph
of a wall. Scoring against many concrete terms and mapping the winner back
to a category is far more reliable, because every prompt is something CLIP
has strong visual grounding for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

DEFAULT_CHECKPOINT = "openai/clip-vit-base-patch32"

# Concrete, visually grounded terms. `category` is what the pipeline acts
# on; the text is what CLIP actually scores.
DEFAULT_VOCABULARY: list[tuple[str, str]] = [
    # --- structure: belongs to the room shell, never to per-object 3D ---
    ("a blank interior wall", "structure"),
    ("a painted wall", "structure"),
    ("a tiled floor", "structure"),
    ("a wooden floor", "structure"),
    ("a carpet on the floor", "structure"),
    ("a plain ceiling", "structure"),
    ("a window with daylight outside", "structure"),
    ("a closed door", "structure"),
    ("a doorway", "structure"),
    ("a corner where two walls meet", "structure"),
    ("a countertop surface", "structure"),
    ("out of focus background", "structure"),
    # --- objects: worth reconstructing individually ---
    ("a sofa", "object"),
    ("an armchair", "object"),
    ("a wooden chair", "object"),
    ("a table", "object"),
    ("a desk", "object"),
    ("a bed", "object"),
    ("a cabinet", "object"),
    ("a bookshelf", "object"),
    ("a television", "object"),
    ("a computer monitor", "object"),
    ("a table lamp", "object"),
    ("a floor lamp", "object"),
    ("a potted plant", "object"),
    ("a vase", "object"),
    ("a cup or mug", "object"),
    ("a bowl of food", "object"),
    ("a bottle", "object"),
    ("a box or package", "object"),
    ("a pillow or cushion", "object"),
    ("a picture frame on the wall", "object"),
    ("a bag", "object"),
    ("a pair of shoes", "object"),
    ("a small household object", "object"),
    # --- neither: should be dropped entirely ---
    ("a person", "person"),
    ("a human hand", "person"),
]

PROMPT_TEMPLATE = "a photo of {}"


def _features(output):
    """Normalise CLIP's feature output across transformers versions.

    `get_text_features` / `get_image_features` return a plain tensor on
    transformers 4.x but a ModelOutput on 5.x. The notebook pins 4.x (TripoSR
    needs it) while a dev machine may well have 5.x, so this has to accept
    both or the module only works in one of the two places it runs.
    """
    for attr in ("text_embeds", "image_embeds", "pooler_output", "last_hidden_state"):
        if hasattr(output, attr):
            value = getattr(output, attr)
            if value is not None:
                return value
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


@dataclass
class SemanticResult:
    label: str
    category: str          # structure | object | person
    confidence: float      # softmax probability of the winning prompt
    margin: float          # winner minus best competing *category*
    scores: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "label": self.label,
            "category": self.category,
            "confidence": round(float(self.confidence), 3),
            "margin": round(float(self.margin), 3),
        }


@dataclass
class SemanticParams:
    checkpoint: str = DEFAULT_CHECKPOINT
    # Context around the mask. A silhouette cut flush is hard to recognise —
    # CLIP does better when it can see a little of what the thing sits on.
    context_pad: float = 0.25
    # Blend between the masked crop and the plain context crop. Pure masked
    # crops lose context; pure context crops let the background dominate a
    # small object. Both are scored and the results averaged.
    context_weight: float = 0.35
    # Below this confidence the label is not trusted enough to act on, and
    # the geometric tests decide instead.
    min_confidence: float = 0.16
    min_margin: float = 0.04
    batch_size: int = 16


class SemanticLabeler:
    """CLIP zero-shot labeller over segmentation masks.

    Lazy-loading, so importing this module costs nothing on a machine that
    will never run it.
    """

    def __init__(
        self,
        params: SemanticParams | None = None,
        device: str | None = None,
        vocabulary: list[tuple[str, str]] | None = None,
    ):
        self.params = params or SemanticParams()
        self.vocabulary = vocabulary or DEFAULT_VOCABULARY
        self._device = device
        self._model = None
        self._processor = None
        self._text_features = None

    @property
    def device(self) -> str:
        if self._device is None:
            try:
                import torch

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:  # pragma: no cover
                self._device = "cpu"
        return self._device

    def load(self) -> "SemanticLabeler":
        if self._model is not None:
            return self
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._processor = CLIPProcessor.from_pretrained(self.params.checkpoint)
        self._model = (
            CLIPModel.from_pretrained(self.params.checkpoint).to(self.device).eval()
        )

        # Text embeddings are fixed for the whole run, so compute them once
        # rather than per image. This is most of the cost of the model.
        prompts = [PROMPT_TEMPLATE.format(t) for t, _ in self.vocabulary]
        with torch.no_grad():
            inputs = self._processor(
                text=prompts, return_tensors="pt", padding=True
            ).to(self.device)
            feats = _features(self._model.get_text_features(**inputs))
        self._text_features = feats / feats.norm(dim=-1, keepdim=True)
        return self

    # ---- crops -------------------------------------------------------

    def _crops(self, image: Image.Image, instance) -> tuple[Image.Image, Image.Image]:
        """(masked crop, context crop) for one instance."""
        rgb = image.convert("RGB")
        w, h = rgb.size
        x0, y0, x1, y1 = instance.bbox
        pad = int(round(max(x1 - x0, y1 - y0) * self.params.context_pad))
        box = (max(0, x0 - pad), max(0, y0 - pad),
               min(w, x1 + pad), min(h, y1 + pad))

        context = rgb.crop(box)

        arr = np.asarray(rgb, dtype=np.float32)
        alpha = instance.mask.astype(np.float32)[..., None]
        # Grey rather than black: a black cut-out reads as a silhouette and
        # CLIP starts describing the shape rather than the thing.
        masked = Image.fromarray(
            np.clip(arr * alpha + 127.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
        ).crop(box)
        return masked, context

    # ---- classification ----------------------------------------------

    def classify(self, image: Image.Image, instances: list) -> list[SemanticResult]:
        """Label every instance. Returns results aligned with `instances`."""
        import torch

        if not instances:
            return []
        self.load()

        masked, context = [], []
        for inst in instances:
            m, c = self._crops(image, inst)
            masked.append(m)
            context.append(c)

        def embed(images):
            feats = []
            for i in range(0, len(images), self.params.batch_size):
                batch = images[i : i + self.params.batch_size]
                inputs = self._processor(images=batch, return_tensors="pt").to(
                    self.device
                )
                with torch.no_grad():
                    f = _features(self._model.get_image_features(**inputs))
                feats.append(f / f.norm(dim=-1, keepdim=True))
            return torch.cat(feats, dim=0)

        w = self.params.context_weight
        logits = (
            (1.0 - w) * (embed(masked) @ self._text_features.T)
            + w * (embed(context) @ self._text_features.T)
        ) * float(self._model.logit_scale.exp())
        probs = logits.softmax(dim=-1).cpu().numpy()

        results = []
        for row in probs:
            best = int(np.argmax(row))
            label, category = self.vocabulary[best]

            # Aggregate probability per category, so "how sure are we it is
            # an object at all" is separate from "which object".
            per_category: dict[str, float] = {}
            for (_, cat), p in zip(self.vocabulary, row):
                per_category[cat] = per_category.get(cat, 0.0) + float(p)
            winner = per_category[category]
            runner_up = max(
                (v for k, v in per_category.items() if k != category), default=0.0
            )
            results.append(
                SemanticResult(
                    label=label,
                    category=category,
                    confidence=float(winner),
                    margin=float(winner - runner_up),
                    scores={k: round(v, 4) for k, v in per_category.items()},
                )
            )
        return results

    def annotate(self, image: Image.Image, instances: list) -> list[SemanticResult]:
        """Classify and write the result onto each instance, in place.

        The filtering in `segmentation.looks_like_object` reads
        `meta["semantic_category"]`, so annotating is all that is needed to
        put semantics in charge of what counts as an object.
        """
        results = self.classify(image, instances)
        for inst, res in zip(instances, results):
            inst.label = res.label
            inst.meta["semantic_label"] = res.label
            inst.meta["semantic_category"] = res.category
            inst.meta["semantic_confidence"] = res.confidence
            inst.meta["semantic_margin"] = res.margin
            inst.meta["semantic_trusted"] = bool(
                res.confidence >= self.params.min_confidence
                and res.margin >= self.params.min_margin
            )
        return results

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._text_features = None
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass


def structure_instances(instances: list) -> list:
    """Instances CLIP considers part of the room, not objects in it."""
    return [
        i for i in instances
        if i.meta.get("semantic_category") == "structure"
        and i.meta.get("semantic_trusted")
    ]
