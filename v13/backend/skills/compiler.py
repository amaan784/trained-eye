"""Skill compiler: spoken request + 3 teaching keyframes -> SkillSpec (Muse Wonder).

The local policy check runs first, so a clear refusal costs nothing. Otherwise Muse
Wonder fills a flat JSON schema (structured output), which is validated as a
SkillSpec. The model can also refuse; its refusal reason is shown to the user as is.
"""

from __future__ import annotations

from pydantic import ValidationError

from backend.meta.client import MetaClient, OutputError, image_part, text_part
from backend.skills.spec import SkillSpec, Target, policy_check, slugify

NO_REASON = frozenset({"", "none", "null", "n/a", "na", "-"})

COMPILE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "refused", "refusal_reason", "name", "sam_prompt", "target_description",
        "positive", "negatives", "alert", "cooldown_s", "confirm_with_cloud",
    ],
    "properties": {
        "refused": {"type": "boolean"},
        "refusal_reason": {"type": ["string", "null"], "maxLength": 300},
        "name": {"type": "string", "maxLength": 40},
        "sam_prompt": {"type": "string", "maxLength": 80},
        "target_description": {"type": "string", "maxLength": 200},
        "positive": {"type": "string", "maxLength": 300},
        "negatives": {"type": "array", "items": {"type": "string", "maxLength": 300}},
        "alert": {"type": "string", "maxLength": 120},
        "cooldown_s": {"type": "integer"},
        "confirm_with_cloud": {"type": "boolean"},
    },
}

SYSTEM = """You turn a spoken request into a watch skill for camera glasses.
The user looked at something for ten seconds; the images are frames from that look.
Fill every field:
- name: snake_case, at most 40 characters, e.g. washer_done.
- sam_prompt: a short noun phrase naming the one thing to look at, for a segmentation
  model (e.g. "washing machine display").
- target_description: that thing in plain words, as seen in the frames.
- positive: the visible condition that should trigger the alert, concrete enough to
  judge from a single frame.
- negatives: 2 to 5 visible near-misses that must NOT trigger it (other states of the
  same thing, similar things nearby).
- alert: what the glasses should say, under 10 words, friendly.
- cooldown_s: seconds before the same alert may repeat (60 to 3600).
- confirm_with_cloud: true unless a false alert would be harmless.
Refuse (refused=true, a one-sentence refusal_reason, other fields empty) if the request
would identify or track a specific person by face, voice or other traits; concerns
people's bodies, health, medication or other sensitive traits; monitors anyone
covertly or relates to a sensitive location; or is about sports performance or
coaching. Skills may only cover objects, states, text, displays, signs and vehicles."""


def compile_skill(
    client: MetaClient,
    request: str,
    keyframes: list[bytes],
    *,
    name: str | None = None,
) -> SkillSpec:
    """Compile a request. `name`, if given, overrides the model's choice and keys spend."""
    skill_name = name or slugify(request)
    refusal = policy_check(request)
    if refusal:
        return SkillSpec(name=skill_name, request=request, refused=True, refusal_reason=refusal)

    def build(data: dict) -> SkillSpec:
        reason = str(data.get("refusal_reason") or "").strip()
        if data.get("refused") or reason.lower() not in NO_REASON:
            return SkillSpec(name=skill_name, request=request, refused=True,
                             refusal_reason=(reason or "I can't learn that one.")[:300])
        try:
            spec = SkillSpec(
                name=name or slugify(data.get("name") or request),
                request=request,
                target=Target(sam_prompt=_clip(data["sam_prompt"], 80),
                              description=_clip(data["target_description"], 200)),
                positive=_clip(data["positive"], 300),
                negatives=[_clip(n, 300) for n in data["negatives"] if n.strip()][:8],
                alert=_clip(data["alert"], 120),
                cooldown_s=min(max(int(data["cooldown_s"]), 0), 86_400),
                confirm_with_cloud=bool(data["confirm_with_cloud"]),
            )
        except (KeyError, TypeError, ValueError, ValidationError) as err:
            raise OutputError(f"compiler output is not a valid skill spec: {err}") from None
        # The model's own wording gets the same local check as the request.
        written = " ".join([spec.target.sam_prompt, spec.target.description, spec.positive,
                            *spec.negatives, spec.alert])
        refusal = policy_check(written)
        if refusal:
            return SkillSpec(name=skill_name, request=request, refused=True,
                             refusal_reason=refusal)
        return spec

    content = [text_part(f'Request: "{request}"')] + [image_part(k) for k in keyframes[:3]]
    try:
        result = client.structured(
            client.settings.model("compiler"), content, COMPILE_SCHEMA, name="skill_spec",
            system=SYSTEM, skill=skill_name, max_tokens=4096, validate=build,
        )
    except OutputError as err:
        if str(err).startswith("model refused:"):
            reason = str(err).removeprefix("model refused:").strip() or "I can't learn that one."
            return SkillSpec(name=skill_name, request=request, refused=True,
                             refusal_reason=reason[:300])
        raise
    return build(result.data)


def _clip(text: str, limit: int) -> str:
    return " ".join(str(text).split())[:limit]
