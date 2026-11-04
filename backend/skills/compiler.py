"""Skill compiler: spoken request + 3 teaching keyframes -> SkillSpec (Muse Wonder).

The local policy check runs first, so a clear refusal costs nothing. Otherwise Muse
Wonder fills a flat JSON schema (structured output), which is validated as a
SkillSpec. The model can also refuse; its refusal reason is shown to the user as is.
"""

from __future__ import annotations

import re

from pydantic import ValidationError

from backend.meta.client import MetaClient, OutputError, image_part, text_part
from backend.skills.spec import (
    MAX_HOLD_S,
    OBJECT_PHRASES,
    REFUSALS,
    SENSITIVE_WORDS,
    SKILL_HOLDS,
    SkillSpec,
    Target,
    _has_word,
    condition_clause,
    distinct,
    normalise,
    policy_check,
    slugify,
    topic_check,
)

# Placeholders a model may write in refusal_reason for an accepted skill, compared after
# lower-casing and turning punctuation into spaces ("N/A." -> "n a", "None." -> "none").
NO_REASON = frozenset({"", "none", "null", "nil", "na", "n a", "no", "nothing",
                       "not applicable", "no refusal", "not refused", "none needed"})
NO_REASON_GIVEN = "The model refused it without giving a reason."
TEXT_FIELDS = ("sam_prompt", "target_description", "positive", "alert")
MAX_REQUEST = 500  # SkillSpec.request
# What SYSTEM asks for; a compiled cooldown is clamped to it (SkillSpec allows 0 to a day
# for hand-written specs). Under a minute, a condition that stays true alerts every few frames.
COOLDOWN_S = (60, 3600)
# Rooms at home may come up in a description ("the bathroom light"); other sensitive
# places may not.
PLACES = tuple(w for w in SENSITIVE_WORDS if w not in ("bathroom", "restroom", "toilet"))
# What a correction rules out ("not the bathroom light, the one on the lid") is not
# what it asks for, so it is left out of the correction's policy check: up to five words
# after "not", stopping at punctuation and at words that start a new clause, so "not the
# light tell me when Sam gets here" is still checked from "tell".
_CLAUSE = (r"but|and|or|so|when|whenever|if|once|until|while|because|to|i|tell|remind|let|"
           r"watch|notify|alert|warn|track|follow|record")
NEGATED = re.compile(r"\b(?:not|isn't|isnt|instead of|rather than)\s+"
                     rf"(?:(?!(?:{_CLAUSE})\b)[^\s,;.!?:\u2014\u2013-]+"
                     r"(?:\s+|(?=[,;.!?:\u2014\u2013-])|$)){1,5}",
                     re.IGNORECASE)

COMPILE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "refused", "refusal_reason", "name", "sam_prompt", "target_description",
        "positive", "negatives", "alert", "cooldown_s", "hold_s", "confirm_with_cloud",
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
        "hold_s": {"type": "integer"},
        "confirm_with_cloud": {"type": "boolean"},
    },
}

SYSTEM = """You turn a spoken request into a watch skill for camera glasses.
The user looked at something for ten seconds; the images are frames from that look.
Fill every field:
- name: snake_case, at most 40 characters, e.g. burner_left_on.
- sam_prompt: a short noun phrase naming the one thing to look at, for a segmentation
  model (e.g. "stove burner").
- target_description: that thing in plain words, as seen in the frames.
- positive: the visible condition that should trigger the alert, concrete enough to
  judge from a single frame, worded to follow "tell you when" (no "when" of its own).
- negatives: 2 to 5 visible near-misses that must NOT trigger it (other states of the
  same thing, similar things nearby).
- alert: what the glasses should say, under 10 words, friendly.
- cooldown_s: seconds before the same alert may repeat (60 to 3600).
- hold_s: seconds the condition must stay true before the alert (0 to 600): 0 when it
  is worth saying at once; longer for something left on, open or running, so that
  using it briefly says nothing (e.g. 30 for a fridge door left open).
- confirm_with_cloud: true unless a false alert would be harmless.
- refusal_reason: null unless refused.
Refuse (refused=true, a one-sentence refusal_reason, other fields empty) if the request
would identify or track a specific person by face, voice or other traits, including a
named person's car, phone or belongings ("when Jake's car pulls in"); concerns
people's bodies, health, medication or other sensitive traits; monitors anyone
covertly or relates to a sensitive location; is about sports performance or
coaching; or is about politics or war (campaigns, elections, protests, soldiers, the
military or weapons). Skills may only cover objects, states, text, displays, signs and
vehicles."""


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
    policy-checked like the request (except what it rules out with "not") and sent with
    it, so the compile runs again. A request over 500 characters is a ValueError, before
    any call.
    """
    request = " ".join(request.split())
    if len(request) > MAX_REQUEST:
        raise ValueError(f"the request is {len(request)} characters, over the "
                         f"{MAX_REQUEST} allowed: say it in one sentence")
    skill_name = name or slugify(request)
    correction = " ".join((correction or "").split())[:300] or None
    refusal = policy_check(request)
    if not refusal and correction:
        refusal = policy_check(NEGATED.sub(" ", normalise(correction)))
        if refusal:
            refusal = (f"{refusal} That came from your correction: reword it (te policy "
                       "checks a wording for free).")
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
                positive=_clip(condition_clause(data["positive"]), 300),
                negatives=distinct([_clip(n, 300) for n in data["negatives"] if n.strip()])[:8],
                alert=_clip(data["alert"], 120),
                cooldown_s=min(max(int(data["cooldown_s"]), COOLDOWN_S[0]), COOLDOWN_S[1]),
                hold_s=hold_for(name or slugify(data.get("name") or request),
                                data.get("hold_s", 0)),
                confirm_with_cloud=bool(data["confirm_with_cloud"]),
            )
        except (KeyError, OverflowError, TypeError, ValueError, ValidationError) as err:
            raise OutputError(f"compiler output is not a valid skill spec: {err}") from None
        # What is watched and what fires the alert get the request's local check too. The
        # rest may mention people and rooms at home in passing ("someone opens the door
        # mid-cycle"), so it gets the covert, health and sports rules, and the description
        # also the person and sensitive-place rules.
        # The trigger is checked with its "when" put back, as the user would say it.
        description = spec.target.description
        # With the request, so "the gun" in a negative reads as the request's glue gun.
        refusal = (policy_check(f"{spec.target.sam_prompt}. When {spec.positive}")
                   or topic_check(". ".join([request, description, *spec.negatives,
                                             spec.alert])))
        if not refusal and policy_check(description) == REFUSALS["person"]:
            refusal = REFUSALS["person"]
        if not refusal and _has_word(OBJECT_PHRASES.sub(" ", normalise(description)), PLACES):
            refusal = REFUSALS["sensitive"]
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


def hold_for(name: str, model_value) -> int:
    """The hold a compiled skill gets: the project's own for its skills (SKILL_HOLDS),
    else the model's value kept between 0 and MAX_HOLD_S."""
    if name in SKILL_HOLDS:
        return SKILL_HOLDS[name]
    return min(max(int(model_value), 0), MAX_HOLD_S)


def _no_reason(reason: str) -> bool:
    return " ".join(re.sub(r"[\W_]+", " ", reason.lower()).split()) in NO_REASON


def _clip(text: str, limit: int) -> str:
    return " ".join(str(text).split())[:limit]
