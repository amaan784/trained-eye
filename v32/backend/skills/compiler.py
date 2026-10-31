"""Skill compiler: spoken request + 3 teaching keyframes -> SkillSpec (Muse Wonder).

The local policy check runs first, so a clear refusal costs nothing. Otherwise Muse
Wonder fills a flat JSON schema (structured output), which is validated as a
SkillSpec. The model can also refuse; its refusal reason is shown to the user as is.
"""

from __future__ import annotations

import re

from pydantic import ValidationError

from backend.meta.client import MetaClient, OutputError, image_part, text_part
from backend.skills.spec import REFUSALS, SkillSpec, Target, policy_check, slugify

# Placeholders a model may write in refusal_reason for an accepted skill, compared after
# lower-casing and turning punctuation into spaces ("N/A." -> "n a", "None." -> "none").
NO_REASON = frozenset({"", "none", "null", "nil", "na", "n a", "no", "nothing",
                       "not applicable", "no refusal", "not refused", "none needed"})
NO_REASON_GIVEN = "The model refused it without giving a reason."
TEXT_FIELDS = ("sam_prompt", "target_description", "positive", "alert")

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
- refusal_reason: null unless refused.
Refuse (refused=true, a one-sentence refusal_reason, other fields empty) if the request
would identify or track a specific person by face, voice or other traits, including a
named person's car, phone or belongings ("when Jake's car pulls in"); concerns
people's bodies, health, medication or other sensitive traits; monitors anyone
covertly or relates to a sensitive location; or is about sports performance or
coaching. Skills may only cover objects, states, text, displays, signs and vehicles."""


def compile_skill(
    client: MetaClient,
    request: str,
    keyframes: list[bytes],
    *,
    name: str | None = None,
    correction: str | None = None,
) -> SkillSpec:
    """Compile a request. `name`, if given, overrides the model's choice and keys spend.

    `correction` is the user's fix for a wrong read-back ("the light on the lid"); it is
    policy-checked like the request and sent with it, so the compile runs again.
    """
    skill_name = name or slugify(request)
    correction = " ".join((correction or "").split())[:300] or None
    refusal = policy_check(request) or (policy_check(correction) if correction else None)
    if refusal:
        return SkillSpec(name=skill_name, request=request, refused=True, refusal_reason=refusal)

    def build(data) -> SkillSpec:
        if not isinstance(data, dict):
            raise OutputError("compiler output is not a JSON object")
        reason = str(data.get("refusal_reason") or "").strip()
        if data.get("refused") or not _no_reason(reason):
            given = reason if not _no_reason(reason) else NO_REASON_GIVEN
            return SkillSpec(name=skill_name, request=request, refused=True,
                             refusal_reason=given[:300])
        negatives = data.get("negatives")
        if (any(not isinstance(data.get(k), str) for k in TEXT_FIELDS)
                or not isinstance(negatives, list)
                or any(not isinstance(n, str) for n in negatives)):
            raise OutputError("compiler output is not a valid skill spec: wrong field types")
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
        except (KeyError, OverflowError, TypeError, ValueError, ValidationError) as err:
            raise OutputError(f"compiler output is not a valid skill spec: {err}") from None
        # What is watched and what fires the alert get the request's local check too. The
        # description gets the person check only, and near-misses and the alert none: they
        # mention people and rooms in passing ("someone opens the door mid-cycle").
        refusal = policy_check(f"{spec.target.sam_prompt}. {spec.positive}")
        if not refusal and policy_check(spec.target.description) == REFUSALS["person"]:
            refusal = REFUSALS["person"]
        if refusal:
            return SkillSpec(name=skill_name, request=request, refused=True,
                             refusal_reason=refusal)
        return spec

    content = [text_part(f'Request: "{request}"')]
    if correction:
        content.append(text_part(f'Correction from the user (it wins over the request): '
                                 f'"{correction}"'))
    content += [image_part(k) for k in keyframes[:3]]
    try:
        result = client.structured(
            client.settings.model("compiler"), content, COMPILE_SCHEMA, name="skill_spec",
            system=SYSTEM, skill=skill_name, max_tokens=4096, validate=build,
        )
    except OutputError as err:
        if str(err).startswith("model refused:"):
            reason = str(err).removeprefix("model refused:").strip() or NO_REASON_GIVEN
            return SkillSpec(name=skill_name, request=request, refused=True,
                             refusal_reason=reason[:300])
        raise
    return build(result.data)


def _no_reason(reason: str) -> bool:
    return " ".join(re.sub(r"[\W_]+", " ", reason.lower()).split()) in NO_REASON


def _clip(text: str, limit: int) -> str:
    return " ".join(str(text).split())[:limit]
