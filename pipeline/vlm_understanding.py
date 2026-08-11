"""
Scene-level object understanding via a vision-language model, for Tier 4.

Every earlier tier finds objects with a *detector*: SAM2's automatic mode
guesses from a blind point grid, GroundingDINO answers a fixed vocabulary
of yes/no questions ("is there a chair? a lamp? a bed?"). Both approaches
share the same ceiling — they can only ever find what their vocabulary or
sampling luck happens to cover, so an object outside that vocabulary (or
sitting where the point grid never landed) is invisible to the rest of the
pipeline no matter how obvious it is in the photo.

A VLM does not have that ceiling. Shown the photo and asked to *list* what
is in it, it is not confined to a fixed noun list, and it can reason about
relationships a box detector has no way to express — that the books belong
to the shelf they are sitting in, that the lamp is part of the same object
as the table it rests on, that a wall-mounted picture is not free-standing
furniture at all. That is genuinely new information this pipeline did not
have before, which is why it earns its own tier rather than replacing the
GroundingDINO vocabulary in Tier 2.

What a VLM is bad at, and known to be bad at across the field, is precise
pixel-space grounding: asked for a bounding box directly, it routinely
hallucinates coordinates that do not line up with the actual object. So
this module never asks it for one. Its only output is a structured object
list (label, group, flat_surface) — the *semantic* judgement calls a box
detector cannot make. The actual boxes still come from GroundingDINO,
queried with the VLM's own open-vocabulary labels instead of a fixed list;
see `pipeline.detection`. This keeps each model doing the part it is
reliable at.

GroundingDINO's returned label for a box is its own phrase grounding, which
often differs in wording from the VLM's exact label text for the same
physical object ("armchair" vs. "brown leather armchair") -- a literal
string match between the two would silently lose objects GroundingDINO
actually found and boxed correctly, purely over wording. `reconcile_labels`
below handles that mismatch with a second, lightweight text-only model call
rather than a rigid lookup.
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
from dataclasses import dataclass

from PIL import Image

DEFAULT_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"
# Text-only, no image -- used only to reconcile GroundingDINO's own detection
# labels against the VLM's object list (see reconcile_labels). A vision model
# isn't needed for that job, and this one is free, real (verified on
# OpenRouter as nvidia/nemotron-3-super-120b-a12b:free), and separate from
# the vision call above so a slow/failing comparison doesn't compete with or
# block scene understanding.
DEFAULT_COMPARISON_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# The free endpoint documents a 1024x1024 max input resolution; sending
# anything larger either gets silently downscaled server-side or rejected
# depending on provider, so resize before sending rather than find out.
MAX_SIDE = 1024

_PROMPT = """You are analyzing a photo of a room for 3D scene reconstruction.

List every distinct physical, movable object visible in the photo -
furniture, decor, appliances, containers, plants, books, anything sitting
in or on the room. Do NOT list structural elements: walls, floor, ceiling,
windows, doors.

Be exhaustive. Include small objects (books, bottles, cushions, remote
controls, plants) as well as large furniture. Do not omit anything visible
just because it is small, partially occluded, or you are unsure of its
exact name - give your best label instead of skipping it.

For each object give exactly these three fields:
- "label": a short noun phrase specific enough for an object detector to
  find it, e.g. "brown leather armchair" rather than just "chair".
- "group": objects that are physically part of one unit and should become
  a SINGLE combined 3D model share the same group string - for example a
  bookshelf and the books sitting in it both get group "bookshelf_1", or a
  table and the lamp resting on it both get group "table_lamp_1". An
  object with nothing resting on/in it gets a group equal to its own
  label.
- "flat_surface": true ONLY if the object is essentially flat/2D and lies
  flush against a wall, ceiling or floor with no real volume of its own -
  a framed picture or photo, a poster, a window, a mirror flush-mounted on
  a wall, a rug/carpet/floor mat lying flat on the floor, a wallpaper
  pattern or wall decal. false for EVERYTHING else, including things that
  are mounted to or pushed against a wall but still have real 3D shape and
  depth - a wall-mounted shelf, a mounted TV, a wall sconce, a wall clock,
  curtains, ceiling lights, an armchair or sofa against a wall, a lamp
  standing near a wall. When in doubt whether something has real volume,
  answer false - it should default to being modelled in 3D.

Respond with ONLY a JSON array of objects with these three fields. No
markdown fences, no prose before or after, just the JSON array."""

_STRICT_REMINDER = (
    "\n\nYour previous reply was not valid JSON. Reply again with ONLY the "
    "JSON array, nothing else - no markdown fences, no explanation."
)


@dataclass
class SceneObject:
    label: str
    group: str
    flat_surface: bool


class VLMError(RuntimeError):
    pass


def _encode_image(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    w, h = rgb.size
    scale = min(1.0, MAX_SIDE / max(w, h))
    if scale < 1.0:
        rgb = rgb.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _extract_json_array(text: str) -> list | None:
    text = text.strip()
    # Strip markdown fences if present despite instructions not to use them.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _parse_objects(raw: list) -> list[SceneObject]:
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        if not label:
            continue
        group = str(item.get("group") or label).strip()
        flat_surface = bool(item.get("flat_surface", False))
        out.append(SceneObject(label=label, group=group, flat_surface=flat_surface))
    return out


def _call_openrouter(
    image_b64: str | None, api_key: str, model: str, endpoint: str, prompt: str, timeout: float,
    max_tokens: int = 3000,
) -> str:
    """POST one chat completion to OpenRouter.

    `image_b64` is None for a text-only request (label reconciliation
    doesn't need the photo, just two lists of strings) -- content is then
    a plain string instead of the multimodal text+image_url block.
    """
    import requests

    content = (
        prompt if image_b64 is None
        else [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]
    )

    try:
        resp = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                # OpenRouter routes free-tier requests through whichever
                # provider is currently serving that model, and several of
                # them reject requests missing these -- not documented as
                # required, but a real, observed failure mode for free
                # models specifically (paid keys are more lenient).
                "HTTP-Referer": "https://github.com/sahilpawar28/holistic-scene-reconstruction",
                "X-Title": "Holistic Scene Reconstruction",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.0,
                # A room with 15-20 objects needs real room for the reply.
                # Without an explicit cap the default can be small enough to
                # truncate the JSON array mid-object, which then fails to
                # parse and looks like "the VLM is broken" when it was
                # actually just cut off.
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
    except requests.exceptions.RequestException as exc:
        raise VLMError(f"OpenRouter request failed: {type(exc).__name__}: {exc}") from exc

    if resp.status_code != 200:
        raise VLMError(f"OpenRouter request failed ({resp.status_code}): {resp.text[:500]}")
    data = resp.json()
    if "error" in data:
        # OpenRouter sometimes reports a routing/provider failure inside a
        # 200 response rather than a non-200 status.
        raise VLMError(f"OpenRouter reported an error: {data['error']}")
    try:
        choice = data["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise VLMError(f"Unexpected OpenRouter response shape: {data}") from exc
    if not content:
        raise VLMError(
            f"OpenRouter returned an empty reply (finish_reason="
            f"{choice.get('finish_reason')!r}) -- the model may have put "
            f"everything into a reasoning block instead of the final "
            f"answer, or hit the token limit before writing any content."
        )
    return content


# Free-tier models on OpenRouter are not kept warm. A cold instance can take
# long enough to start responding that OpenRouter's own gateway gives up
# waiting on it and reports a 5xx/429 before the model ever gets a chance to
# answer -- observed in practice as "Upstream idle timeout exceeded" (504).
# This is almost always transient: the same request a few seconds later, once
# something has warmed the instance up, typically succeeds. Worth retrying
# automatically rather than surfacing it as "the VLM is broken" and falling
# back to Tier 2 on what was really just bad timing.
_TRANSIENT_STATUS_CODES = (429, 500, 502, 503, 504)


def _is_transient(exc: VLMError) -> bool:
    text = str(exc).lower()
    if any(f"({code})" in text or f"'code': {code}" in text or f'"code": {code}' in text
           for code in _TRANSIENT_STATUS_CODES):
        return True
    return "timeout" in text or "idle" in text


def _call_with_retries(
    image_b64: str | None, api_key: str, model: str, endpoint: str, prompt: str, timeout: float,
    max_attempts: int = 3, backoff_seconds: float = 4.0, max_tokens: int = 3000,
) -> str:
    last_exc: VLMError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return _call_openrouter(image_b64, api_key, model, endpoint, prompt, timeout,
                                    max_tokens=max_tokens)
        except VLMError as exc:
            last_exc = exc
            if attempt == max_attempts or not _is_transient(exc):
                raise
            time.sleep(backoff_seconds * attempt)
    raise last_exc  # pragma: no cover — loop always returns or raises


def understand_scene(
    image: Image.Image,
    api_key: str,
    model: str = DEFAULT_MODEL,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout: float = 120.0,
) -> list[SceneObject]:
    """Ask the VLM to list every object in the photo, with grouping hints.

    Raises `VLMError` if the API call fails (after retrying transient
    gateway/timeout errors) or the model's reply cannot be parsed as the
    expected JSON shape even after one retry with a stricter reminder.
    Callers should treat that as "VLM understanding unavailable" and decide
    their own fallback rather than let it silently return an empty scene.
    """
    if not api_key:
        raise VLMError("no OpenRouter API key configured")

    image_b64 = _encode_image(image)
    content = _call_with_retries(image_b64, api_key, model, endpoint, _PROMPT, timeout)
    parsed = _extract_json_array(content)

    if parsed is None:
        content = _call_with_retries(
            image_b64, api_key, model, endpoint, _PROMPT + _STRICT_REMINDER, timeout
        )
        parsed = _extract_json_array(content)

    if parsed is None:
        raise VLMError(f"could not parse a JSON array from the VLM reply: {content[:500]!r}")

    objects = _parse_objects(parsed)
    if not objects:
        raise VLMError(f"VLM reply parsed but contained no usable objects: {parsed!r}")
    return objects


_RECONCILE_PROMPT_TEMPLATE = """You are matching two lists of object names that both describe the SAME photo of a room.

List A -- named by a vision-language model:
{a_list}

List B -- named by a separate object detector, which may use different
wording for the same physical objects (e.g. "armchair" instead of "brown
leather armchair"), or may include things List A didn't separately name:
{b_list}

For each item in List A, decide whether any item in List B refers to the
SAME physical object, just possibly worded differently. Do not match two
different physical objects just because they are the same general category
(a "wooden dining chair" and a "office chair" are NOT the same object even
though both are chairs).

Respond with ONLY a JSON array, one entry per List A item, in this exact
form: [{{"a": "<List A item, verbatim>", "b": "<matching List B item,
verbatim, or null if none of them refer to the same object>"}}]. No
markdown fences, no prose before or after, just the JSON array."""

_RECONCILE_STRICT_REMINDER = (
    "\n\nYour previous reply was not valid JSON. Reply again with ONLY the "
    "JSON array, nothing else."
)


def reconcile_labels(
    a_labels: list[str],
    b_labels: list[str],
    api_key: str,
    model: str = DEFAULT_COMPARISON_MODEL,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout: float = 60.0,
) -> dict[str, str | None]:
    """Match VLM object labels (A) against GroundingDINO detection labels (B)
    that refer to the same physical object, worded differently.

    This exists because GroundingDINO is queried with the VLM's own labels
    as the detection prompt -- open-vocabulary, not a fixed list -- but
    GroundingDINO's returned label for a matched box is its own phrase
    grounding, which routinely differs in wording from the VLM's exact
    label text. A literal string match then misses objects GroundingDINO
    genuinely found and boxed correctly, purely because the two labels for
    the same physical thing were worded differently. This call can only
    reconcile boxes GroundingDINO already found; it cannot invent a
    detection for something GroundingDINO never located at all -- that is
    a detector limitation this step is not meant to paper over.

    Text-only (no image) and deliberately a separate, non-vision model
    (DEFAULT_COMPARISON_MODEL) from the one that does the actual scene
    understanding -- this is a much lighter task than looking at a photo.

    Returns {a_label: matching_b_label_or_None} for every label in
    `a_labels`. Raises VLMError on request/parse failure (after retrying
    transient errors) -- callers should treat that as "reconciliation
    unavailable" and fall back to whatever direct matches they already had,
    the same way understand_scene's own callers fall back to Tier 2.
    """
    if not api_key:
        raise VLMError("no OpenRouter API key configured")
    if not a_labels:
        return {}

    a_list = "\n".join(f"- {a}" for a in a_labels)
    b_list = "\n".join(f"- {b}" for b in b_labels) if b_labels else "(empty -- nothing detected)"
    prompt = _RECONCILE_PROMPT_TEMPLATE.format(a_list=a_list, b_list=b_list)

    content = _call_with_retries(None, api_key, model, endpoint, prompt, timeout,
                                 max_tokens=1500)
    parsed = _extract_json_array(content)
    if parsed is None:
        content = _call_with_retries(None, api_key, model, endpoint,
                                     prompt + _RECONCILE_STRICT_REMINDER, timeout,
                                     max_tokens=1500)
        parsed = _extract_json_array(content)
    if parsed is None:
        raise VLMError(f"could not parse a JSON array from the reconciliation reply: {content[:500]!r}")

    mapping: dict[str, str | None] = {a: None for a in a_labels}
    b_set = set(b_labels)
    for item in parsed:
        if not isinstance(item, dict):
            continue
        a = str(item.get("a", "")).strip()
        b = item.get("b")
        b = str(b).strip() if b else None
        # Only trust a match that names something actually in List B --
        # a hallucinated "b" value that isn't one of the real detections
        # would silently attach the wrong box to this label downstream.
        if a in mapping and b in b_set:
            mapping[a] = b
    return mapping
