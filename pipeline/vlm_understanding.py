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
list (label, group, wall_mounted) — the *semantic* judgement calls a box
detector cannot make. The actual boxes still come from GroundingDINO,
called once per label the VLM names instead of a fixed vocabulary; see
`pipeline.detection`. This keeps each model doing the part it is reliable
at.
"""

from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass

from PIL import Image

DEFAULT_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"
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
- "wall_mounted": true if the object is fixed to a wall or ceiling rather
  than resting on the floor (framed pictures, wall-mounted shelves,
  sconces, mounted TVs, curtains, ceiling lights). false otherwise.

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
    wall_mounted: bool


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
        wall_mounted = bool(item.get("wall_mounted", False))
        out.append(SceneObject(label=label, group=group, wall_mounted=wall_mounted))
    return out


def _call_openrouter(
    image_b64: str, api_key: str, model: str, endpoint: str, prompt: str, timeout: float
) -> str:
    import requests

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
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                            },
                        ],
                    }
                ],
                "temperature": 0.0,
                # A room with 15-20 objects needs real room for the reply.
                # Without an explicit cap the default can be small enough to
                # truncate the JSON array mid-object, which then fails to
                # parse and looks like "the VLM is broken" when it was
                # actually just cut off.
                "max_tokens": 3000,
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


def understand_scene(
    image: Image.Image,
    api_key: str,
    model: str = DEFAULT_MODEL,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout: float = 90.0,
) -> list[SceneObject]:
    """Ask the VLM to list every object in the photo, with grouping hints.

    Raises `VLMError` if the API call fails or the model's reply cannot be
    parsed as the expected JSON shape even after one retry with a stricter
    reminder. Callers should treat that as "VLM understanding unavailable"
    and decide their own fallback rather than let it silently return an
    empty scene.
    """
    if not api_key:
        raise VLMError("no OpenRouter API key configured")

    image_b64 = _encode_image(image)
    content = _call_openrouter(image_b64, api_key, model, endpoint, _PROMPT, timeout)
    parsed = _extract_json_array(content)

    if parsed is None:
        content = _call_openrouter(
            image_b64, api_key, model, endpoint, _PROMPT + _STRICT_REMINDER, timeout
        )
        parsed = _extract_json_array(content)

    if parsed is None:
        raise VLMError(f"could not parse a JSON array from the VLM reply: {content[:500]!r}")

    objects = _parse_objects(parsed)
    if not objects:
        raise VLMError(f"VLM reply parsed but contained no usable objects: {parsed!r}")
    return objects
