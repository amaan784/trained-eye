"""Skill spec: what one taught skill watches for (version 1).

A spec is produced by the skill compiler (compiler.py) from the spoken request and a
few teaching keyframes, validated here, and read back to the user in plain English
before anything is generated or trained.

Skills may cover objects, states, text, displays, signs and vehicles. They may never
identify or track a specific person, concern people's bodies or health, monitor
anyone covertly, or coach sports performance. `policy_check` catches the clear cases
locally (no API call, no cost); the compiler model refuses the rest.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SPEC_VERSION = 1
NAME_PATTERN = r"^[a-z][a-z0-9_]{1,39}$"


class Target(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sam_prompt: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=200)


class SkillSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = SPEC_VERSION
    name: str = Field(pattern=NAME_PATTERN)
    request: str = Field(min_length=1, max_length=500)
    target: Target | None = None
    positive: str | None = Field(default=None, max_length=300)
    negatives: list[str] = Field(default_factory=list, max_length=8)
    alert: str | None = Field(default=None, max_length=120)
    cooldown_s: int = Field(default=600, ge=0, le=86_400)
    confirm_with_cloud: bool = True
    refused: bool = False
    refusal_reason: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def _complete(self) -> SkillSpec:
        if self.refused:
            if not self.refusal_reason:
                raise ValueError("a refused skill needs a refusal_reason")
        else:
            missing = [f for f in ("target", "positive", "alert") if not getattr(self, f)]
            if missing:
                raise ValueError(f"an accepted skill needs {', '.join(missing)}")
            if not self.negatives:
                raise ValueError("an accepted skill needs at least one negative")
        return self


def check_name(name: str) -> str:
    """Raise ValueError unless `name` is a valid skill name (snake_case, 2 to 40 chars)."""
    if not re.fullmatch(NAME_PATTERN, name):
        raise ValueError(f"skill name {name!r} must be snake_case: lower-case letters, digits "
                         "and underscores, starting with a letter (e.g. washer_done)")
    return name


def lower_first(text: str) -> str:
    """Lower-case the first letter so a phrase can sit mid-sentence; keep acronyms."""
    text = text.strip().rstrip(". ")
    if not text:
        return text
    first = text.split()[0]
    if len(first) > 1 and first.isupper():
        return text
    return text[0].lower() + text[1:]


def read_back(spec: SkillSpec) -> str:
    """The sentence the user confirms or corrects before anything is generated."""
    if spec.refused:
        return f"I can't learn that one. {spec.refusal_reason}"
    return (
        f"I'll watch {_with_article(spec.target.description)} and tell you when "
        f"{lower_first(spec.positive)}. I'll say: \"{spec.alert}\""
    )


def _with_article(phrase: str) -> str:
    phrase = lower_first(phrase)
    if phrase.lower().startswith(("the ", "a ", "an ", "my ", "your ", "this ", "that ")):
        return phrase
    return f"the {phrase}"


# -- local policy pre-check --------------------------------------------------------

# Whole words or phrases, except entries ending in "*", which match as prefixes. Bare
# words that also name objects (man, weight, skin, stalk) are only used inside phrases,
# so walk signs, kitchen scales and celery are not refused.
PERSON_WORDS = (
    "person", "people", "someone", "somebody", "anyone", "anybody", "woman", "women",
    "a man", "the man", "a guy", "the guy", "girl", "boy", "kid", "kids", "child",
    "children", "baby", "toddler", "he", "she", "him", "her", "his", "hers",
    "wife", "husband", "partner", "girlfriend", "boyfriend", "son", "daughter", "mom",
    "mum", "dad", "mother", "father", "brother", "sister", "grandma", "grandpa", "friend",
    "roommate", "flatmate", "neighbor", "neighbour", "landlord", "landlady", "tenant",
    "boss", "coworker", "colleague", "employee", "stranger", "visitor", "guest", "nanny",
    "babysitter", "mailman", "postman", "ex", "who", "whoever",
)
IDENTIFY = re.compile(r"\b(?:recogni[sz]e|identify|track|follow)\s+(?:him|her|them|people|"
                      r"faces?|who|the person|someone|anyone)\b", re.IGNORECASE)
HEALTH_WORDS = (
    "pill", "pills", "meds", "medication*", "medicine", "prescription", "dose", "insulin",
    "inhaler", "glucose", "blood", "symptom*", "fever", "rash", "wound", "injury",
    "pregnan*", "calorie*", "diet", "drunk", "heart rate", "sick", "illness", "my weight",
    "her weight", "his weight", "lose weight", "weigh myself", "my skin", "her skin",
    "his skin",
)
COVERT_WORDS = ("secretly", "without them knowing", "without her knowing",
                "without him knowing", "without their knowledge", "behind her back",
                "behind his back", "spy", "spying", "covert*", "hidden camera", "stalking",
                "stalker")
SENSITIVE_WORDS = ("clinic", "hospital", "locker room", "changing room", "bathroom",
                   "restroom", "toilet", "church", "mosque", "synagogue", "temple",
                   "shelter", "polling", "courthouse", "police station")
SPORTS_WORDS = ("count my", "workout", "coach me", "coaching", "sports")
SPORTS_FORM = re.compile(r"\bmy\s+(?:\w+\s+)?(?:swing|form|serve|stroke|shot|technique|"
                         r"posture|reps|stance)\b", re.IGNORECASE)

# A capitalized word after "when" counts as a name only if an arrival or presence verb
# follows, so devices and display labels ("when Roomba is stuck", "until End lights up")
# are not mistaken for people.
NAMED_ARRIVAL = re.compile(
    r"\b(?i:when|if|once|until|whenever|as soon as)\s+([A-Z][a-z]+)(?:'s)?\s+"
    r"(?i:(?:\w+\s+)?(?:gets?|arrives?|comes?|leaves?|pulls?|walks?|drives?|enters?|"
    r"shows up|is (?:here|home|back|at|in|out|outside|leaving|coming)|home|here|back))\b"
)
NOT_NAMES = frozenset({"The", "This", "That", "It", "My", "Our", "Your", "There", "Its",
                       "Amazon", "UPS", "FedEx", "Uber", "Lyft", "Tesla"})

REFUSALS = {
    "person": "I can't watch for or recognize specific people. Skills can cover objects, "
              "states, text, displays, signs and vehicles.",
    "health": "I can't watch for anything about people's bodies, health or medication.",
    "covert": "I can't help monitor anyone without their knowledge.",
    "sensitive": "I can't watch sensitive places such as clinics, places of worship or "
                 "changing rooms.",
    "sports": "I can't do sports analysis or coaching.",
}


def _has_word(text: str, words: tuple[str, ...]) -> bool:
    lowered = text.lower()
    for word in words:
        if word.endswith("*"):
            pattern = rf"\b{re.escape(word[:-1])}"
        else:
            pattern = rf"\b{re.escape(word)}\b"
        if re.search(pattern, lowered):
            return True
    return False


def policy_check(request: str) -> str | None:
    """Return a one-sentence refusal if the request clearly breaks policy, else None."""
    for named in NAMED_ARRIVAL.finditer(request):
        if named.group(1) not in NOT_NAMES:
            return REFUSALS["person"]
    if IDENTIFY.search(request) or _has_word(request, PERSON_WORDS):
        return REFUSALS["person"]
    if _has_word(request, COVERT_WORDS):
        return REFUSALS["covert"]
    if _has_word(request, HEALTH_WORDS):
        return REFUSALS["health"]
    if _has_word(request, SENSITIVE_WORDS):
        return REFUSALS["sensitive"]
    if SPORTS_FORM.search(request) or _has_word(request, SPORTS_WORDS):
        return REFUSALS["sports"]
    return None


def slugify(text: str, fallback: str = "skill") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40].strip("_")
    if not slug or not slug[0].isalpha():
        slug = f"{fallback}_{slug}".strip("_")[:40]
    return slug
