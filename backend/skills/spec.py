"""Skill spec: what one taught skill watches for (version 1).

A spec is produced by the skill compiler (compiler.py) from the spoken request and a
few teaching keyframes, validated here, and read back to the user in plain English
before anything is generated or trained.

Skills may cover objects, states, text, displays, signs and vehicles. They may never
identify or track a specific person, concern people's bodies or health, monitor
anyone covertly, coach sports performance, or be about politics or war (an anti-project
of the hackathon). `policy_check` catches the clear cases
locally (no API call, no cost); the compiler model refuses the rest.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SPEC_VERSION = 1
NAME_PATTERN = r"^[a-z][a-z0-9_]{1,39}$"
# The project's skills and how long each condition must stay true before it alerts. The
# compiler gives these names these holds; any other skill takes the model's own value.
SKILL_HOLDS = {"fridge_left_open": 20, "tap_left_running": 20, "burner_left_on": 10,
               "egg_whisked": 0}
# How often the alert repeats while the condition stays true (the cooldown): a reminder
# until it is dealt with, the burner, the riskier one, more often. Any other skill takes
# the model's own cooldown.
SKILL_REPEATS = {"fridge_left_open": 30, "tap_left_running": 30, "burner_left_on": 20}
MAX_HOLD_S = 600


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
    # Seconds the condition must stay true (after the 3-of-4 smoothing) before it alerts:
    # 0 alerts at once; a fridge left open waits 30 s, so opening it briefly says nothing.
    hold_s: int = Field(default=0, ge=0, le=MAX_HOLD_S)
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
                         "and underscores, starting with a letter (e.g. burner_left_on)")
    return name


def lower_first(text: str) -> str:
    """Lower-case the first letter so a phrase can sit mid-sentence. Acronyms ("LED"),
    names with inner capitals ("McKenzie's") and two-word names ("Instant Pot") keep
    theirs. A final full stop, ! or ? goes."""
    text = text.strip().rstrip(".!? ")
    if not text:
        return text
    words = text.split()
    first = words[0]
    lowered = text[0].lower() + text[1:]
    if f"{first.lower()} " in DETERMINERS:  # "The LG display" -> "the LG display"
        return lowered
    if (len(first) > 1 and first.isupper()) or any(c.isupper() for c in first[1:]):
        return text
    if first[0].isupper() and len(words) > 1 and words[1][:1].isupper():
        return text
    if first.strip("'\"") in DISPLAY_LABELS and (len(words) == 1 or LABEL_VERB.match(words[1])):
        return text  # "End shows", but "ready light turns green"
    return lowered


LABEL_VERB = re.compile(r"(?:shows?|appears?|lights|comes?|is|flashes|blinks|displays?|turns?)$",
                        re.IGNORECASE)
LEADING_WHEN = re.compile(r"^\s*(?:when|whenever|if|once|as soon as)\s+", re.IGNORECASE)


def condition_clause(text: str) -> str:
    """A condition that reads after "tell you when": no leading "when" of its own."""
    return LEADING_WHEN.sub("", text.strip())


def read_back(spec: SkillSpec) -> str:
    """The sentence the user confirms or corrects before anything is generated."""
    if spec.refused:
        return f"I can't learn that one. {spec.refusal_reason}"
    return (
        f"I'll watch {_with_article(spec.target.description)} and tell you when "
        f"{lower_first(condition_clause(spec.positive))}. I'll say: \"{spec.alert}\""
    )


DETERMINERS = ("the ", "a ", "an ", "my ", "your ", "our ", "their ", "its ", "this ",
               "that ", "these ", "those ", "both ", "each ", "every ")


def _with_article(phrase: str) -> str:
    phrase = lower_first(phrase)
    if phrase.lower().startswith(DETERMINERS):
        return phrase
    return f"the {phrase}"


def distinct(conditions: list[str]) -> list[str]:
    """Conditions that differ in more than case or a final full stop ("Display off" and
    "display off." would give the same image prompt, so the same image twice)."""
    seen, out = set(), []
    for condition in conditions:
        key = lower_first(condition).lower()
        if key not in seen:
            seen.add(key)
            out.append(condition)
    return out


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
    "babysitter", "mailman", "postman", "ex", "who", "whoever", "men", "babies", "aunt",
    "uncle", "cousin", "niece", "nephew", "grandmother", "grandfather", "grandparent",
    "parent", "spouse", "fiance", "fianc\u00e9", "fianc\u00e9e", "teen", "teenager",
    "teacher", "nurse", "wives", "nannies", "landladies", "mailmen", "postmen",
)
# Words that are not nouns get no plural: "his" must not match "hiss" (a kettle).
NO_PLURAL = frozenset({"he", "she", "him", "her", "his", "hers", "who", "whoever", "ex",
                       "a man", "the man", "men", "people", "children", "babies", "kids",
                       "wives", "nannies", "landladies", "mailmen", "postmen"})
IDENTIFY = re.compile(r"\b(?:recogni[sz]e|identify|track|follow)\s+(?:him|her|them|people|"
                      r"faces?|who|the person|someone|anyone)\b", re.IGNORECASE)
HEALTH_WORDS = (
    "pill", "pills", "meds", "medication*", "medicine", "prescription", "dose", "insulin",
    "inhaler", "glucose", "blood", "symptom*", "fever", "rash", "wound", "injury",
    "pregnan*", "calorie*", "diet", "drunk", "heart rate", "sick", "illness", "my weight",
    "her weight", "his weight", "lose weight", "weigh myself", "my skin", "her skin",
    "his skin", "kcal", "nutrition*", "pillbox", "oximeter", "cgm", "vitamin*",
)
COVERT_WORDS = ("secretly", "without them knowing", "without her knowing",
                "without him knowing", "without their knowledge", "behind her back",
                "behind his back", "spy", "spying", "covert*", "hidden camera", "stalking",
                "stalker", "nobody notices", "no one notices", "without them noticing",
                "without anyone noticing", "without anyone knowing", "snoop", "snoops",
                "snooping", "snooped", "eavesdrop*")
COVERT_PHRASES = re.compile(r"\bdiscreetly\s+(?:watch|monitor|record|film|track)\b",
                            re.IGNORECASE)
# The glasses' capture light always shows when the camera is on (a fixed rule, not a skill).
# Only the glasses' light counts: "turn off the recording light" on a camcorder is fine.
_GLASSES_LIGHT = (r"(?:the\s+|my\s+|its\s+)?(?:glasses'?\s+(?:camera\s+|recording\s+)?"
                  r"(?:light|led|indicator)|(?:capture|privacy)\s+(?:light|led|indicator))")
_HIDE = (r"hide|hiding|cover|covering|tape over|taping over|disable|disabling|block|blocking|"
         r"black out|blacking out")
CAPTURE_LIGHT = re.compile(
    rf"\b(?:{_HIDE}|turn off|turning off|switch off|switching off|shut off|dim|dimming)\s+"
    rf"{_GLASSES_LIGHT}"
    rf"|\b(?:{_HIDE})\s+(?:the\s+|my\s+|its\s+)?(?:led|recording light|camera light)\b"
    rf"|\b(?:turn|switch|shut|keep)\s+{_GLASSES_LIGHT}\s+off\b"
    r"|\b(?:turn|switch|shut)\s+off\s+the\s+led\s+on\s+(?:the|my)\s+glasses\b"
    r"|\bkeep\s+(?:the\s+)?(?:led|capture light|privacy light)\s+off\b"
    r"|\bwithout\s+(?:the\s+)?(?:led|capture light|privacy light)\s+on\b",
    re.IGNORECASE)
SENSITIVE_WORDS = ("clinic", "hospital", "locker room", "changing room", "bathroom",
                   "restroom", "toilet", "church", "mosque", "synagogue", "temple",
                   "shelter", "polling", "courthouse", "police station")
SPORTS_WORDS = ("workout", "coach me", "coach my", "coaching", "sports", "squat depth",
                "yoga pose")
# "my shot" and "my form" count bare or after a sport word, so "my espresso shot" and
# "my tax form" pass; "my lift" (a ride) and "my pitch" (a deck) only after a sport word.
# Counting counts only exercises, and without "my" only words that are nothing else.
SPORT = (r"golf|tennis|basketball|baseball|softball|football|soccer|hockey|volleyball|"
         r"cricket|rugby|lacrosse|badminton|squash|pickleball|bowling|archery|boxing|yoga|"
         r"pilates|jump|running|swimming|swim|rowing|climbing|skating|skiing|squat|lifting|"
         r"deadlift|bench|throwing|batting|shooting|pitching|putting|kicking|passing|"
         r"free[ -]throw|three[ -]point|penalty|tee")
EXERCISES = (r"reps|push[- ]?ups|pull[- ]?ups|chin[- ]?ups|sit[- ]?ups|burpees|"
             r"jumping[- ]jacks|lunges|squats|crunches")
SPORTS_FORM = re.compile(
    r"\bmy\s+(?:\w+\s+)?(?:swing|serve|stroke|technique|posture|reps|stance|backhand|"
    r"forehand|putt|putts|squat|squats)s?\b"
    rf"|\bmy\s+(?:(?:{SPORT})\s+)?(?:shot|form)\b"
    rf"|\bmy\s+(?:{SPORT})\s+(?:lift|kick|pitch|throw|dive)s?\b"
    rf"|\bcount\s+(?:my\s+|the\s+)?(?:[\w-]+\s+)?(?:{EXERCISES})\b"
    r"|\bcount\s+my\s+(?:[\w-]+\s+)?(?:laps|steps|miles|curls|jumps)\b",
    re.IGNORECASE)
# Politics and war, a hackathon anti-project. Phrases, so a war film poster, a campaign of
# laundry loads, election day on a calendar, military time, a glue gun or a Swiss army
# knife pass; ARMS_LOOKALIKES (toy weapons, a Salvation Army truck) are removed first.
POLITICS_WORDS = ("politic*", "propaganda", "partisan*", "democrat*", "republican*", "maga",
                  "ballot", "ballots", "voter*", "vote for", "voting for", "voted for", "protest",
                  "protests", "protester*", "protestor*", "protesting",
                  "picket line*", "riot police", "riot gear", "the president",
                  "prime minister", "senator", "senators", "soldier*", "troops", "militia*",
                  "weapon*", "firearm*", "rifle*", "handgun*", "a gun", "the gun", "my gun",
                  "guns", "pistol*", "shotgun*", "revolver*", "ammo", "bullets", "gunfire",
                  "gunshot*", "gunman", "gunmen", "missile*", "rocket launcher*", "grenade*",
                  "landmine*", "land mine*", "artillery", "ammunition", "a bomb", "the bomb",
                  "bombing*", "car bomb*", "airstrike*", "air strike*", "air raid*",
                  "drone strike*", "warplane*", "war plane*", "fighter jet*", "war zone",
                  "warzone",
                  "war crime*", "at war")
POLITICS_FORM = re.compile(
    r"\b(?:campaign|election)\s+(?:\w+\s+)?(?:signs?|posters?|banners?|flyers?|stickers?|"
    r"buttons?|billboards?|rally|rallies|office)\b"
    r"|\belection\s+(?:\w+\s+)?(?:ads?|adverts?|results?|coverage)\b"  # a campaign's may sell
    r"|\b(?:military|army)\s+(?:vehicles?|trucks?|convoys?|tanks?|aircraft|planes?|jets?|"
    r"helicopters?|drones?|bases?|parades?|police|checkpoints?|uniforms?|personnel|units?|"
    r"forces?|equipment|activity|movements?|exercises?|operations?|targets?)\b"
    r"|\bbattle\s+tanks?\b|\btanks?\s+(?:rolls?|rolling|drives?|driving|fires?|firing)\b",
    re.IGNORECASE)
TOOL_GUNS = (r"\b(?:glue|staple|nail|heat|caulk|caulking|spray|sprayer|paint|grease|solder|"
             r"soldering|massage|lint|price|label|tape|foam|nerf|water|squirt|toy|tin|lego|"
             r"plastic|bubble|confetti)\s+guns?\b")
ARMS_LOOKALIKES = re.compile(
    r"\b(?:toy|tin|lego|plastic|foam|nerf|water|squirt)\s+(?:soldiers?|tanks?|guns?|pistols?|"
    r"rifles?|weapons?|missiles?|grenades?|fighter jets?|warplanes?)\b"
    r"|\bsalvation\s+army\b|\bsoldier\s+(?:fly|flies|ants?|beetles?)\b"
    rf"|{TOOL_GUNS}|\bgun[- ]?metal\b|\bpistol[- ]grip\b|\bbath\s+bombs?\b"
    r"|\bsoldiers?\s+(?:figurines?|figures?|miniatures?|models?)\b"
    r"|\b(?:ad|advertising|marketing|sales|brand|product|promo|promotional|fundraising|"
    r"charity)\s+campaigns?\b|\bpresidents?'?s?\s+(?:choice|day)\b",
    re.IGNORECASE)
# A compound the text names makes a bare mention later in it ("the gun is unplugged" after
# "the glue gun") that same thing, so the bare word goes too before the check.
REFERS_BACK = (
    (re.compile(TOOL_GUNS, re.IGNORECASE), re.compile(r"\b(?:a|the|my|this|that)\s+guns?\b|\bguns\b",
                                                       re.IGNORECASE)),
    (re.compile(r"\bbath\s+bombs?\b", re.IGNORECASE),
     re.compile(r"\b(?:a|the|this|that)\s+bombs?\b", re.IGNORECASE)),
    (re.compile(r"\b(?:toy|tin|lego|plastic)\s+soldiers?\b", re.IGNORECASE),
     re.compile(r"\bsoldiers?\b", re.IGNORECASE)),
)
# Object names that contain a person word; removed before the word lists are checked.
# "child lock" stays a phrase only before a word break, so "the child locks the door" counts.
OBJECT_PHRASES = re.compile(r"\b(?:child[ -]?lock|child[ -]?proof|baby (?:bottle|food|"
                            r"spinach|carrots?)|bus shelter|green men|men at work|"
                            r"nurse sharks?|parent (?:folder|directory)|uncle ben's)\b",
                            re.IGNORECASE)

# A capitalized word after "when" counts as a name only if an arrival or presence verb
# follows, so devices and display labels ("when Roomba is stuck", "until End lights up")
# are not mistaken for people. "gets" and "comes" do not count before a device state
# ("Instant Pot comes to pressure", "Nest gets to 70 degrees", "Brita's light comes on");
# without a possessive, also not before "stuck" ("Roomba gets stuck", but "when Jake's
# car gets stuck" counts). Between the name and the verb: one word, another name ("Sam
# and Jake arrive", "Mary Jane Watson arrives"), or with a possessive up to two words.
# After a title (Dr. Patel) the word is a name even if it is also a brand.
_LEAD = (r"\b(?i:when|if|once|until|whenever|as soon as)\s+"
         r"(?P<title>(?i:dr|mr|mrs|ms|mx|miss)\.?\s+)?")
_ARRIVAL = (r"arrives?|leaves?|pulls?|walks?|drives?|enters?|shows up|"
            r"is (?:here|home|back|at|in|out|outside|leaving|coming)|home|here|back")
_READING = r"\d+(?:\.\d+)?\s*(?:%|\u00b0\s*[fc]?|degrees?|percent|[fc]|psi|minutes?)?\W*$"
_DEVICE_STATE = (r"(?:full|empty|hot|cold|warm|low)\b|to\s+(?:" + _READING + r"|pressure|temp)|"
                 r"up\s+on\b|on\W*$")
_NAME = r"([^\W\d_][\w'-]*?)"
NAMED_ARRIVAL = (
    re.compile(_LEAD + _NAME + r"'s\s+"
               rf"(?i:(?:\w+\s+){{0,2}}(?:(?:gets?|comes?)(?!\s+(?:{_DEVICE_STATE}))|"
               rf"{_ARRIVAL})|at|in|out|outside)\b"),
    re.compile(_LEAD + _NAME + r"(?:'s)?\s+"
               r"(?:(?:[A-Z][\w'-]*\s+){1,2}|(?i:and|or|&)\s+[^\W\d_][\w'-]*\s+|"
               r"(?i:\w+\s+))?"
               rf"(?i:(?:gets?|comes?)(?!\s+(?:stuck\b|jammed\b|clogged\b|{_DEVICE_STATE}))|"
               rf"{_ARRIVAL})\b"),
)
# Words a display shows; they keep their capital in a read-back ("when End shows").
DISPLAY_LABELS = frozenset({"End", "Done", "Ready", "Off", "On", "Error", "Hot", "Low", "Full",
                            "Empty", "Open", "Closed", "Start", "Stop", "Pause", "Boil", "Warm"})
NOT_NAMES = DISPLAY_LABELS | frozenset({"The", "This", "That", "It", "My", "Our", "Your", "There", "Its",
                       "Amazon", "UPS", "FedEx", "Uber", "Lyft", "Tesla", "DHL", "USPS",
                       "DoorDash", "UberEats", "Grubhub", "GrubHub", "Instacart",
                       # devices
                       "Roomba", "Nest", "Bosch", "Kindle", "Whirlpool", "Dyson", "Keurig",
                       "Nespresso", "Breville", "Samsung", "Instant", "Ecobee", "Traeger",
                       "Meater", "Vitamix", "Brita", "Sonos", "KitchenAid", "Miele", "Rivian",
                       # days ("when Sunday's paper arrives")
                       "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
                       "Sunday"})
# Typed apostrophes (iOS and macOS turn ' into a curly one by default).
APOSTROPHES = {0x2019: "'", 0x2018: "'", 0x02BC: "'", 0x60: "'"}

REFUSALS = {
    "person": "I can't watch for or recognize specific people. Skills can cover objects, "
              "states, text, displays, signs and vehicles.",
    "health": "I can't watch for anything about people's bodies, health or medication.",
    "covert": "I can't help monitor anyone without their knowledge.",
    "sensitive": "I can't watch sensitive places such as clinics, places of worship or "
                 "changing rooms.",
    "sports": "I can't do sports analysis or coaching.",
    "politics": "I can't do skills about politics or war, such as campaigns, elections, "
                "protests, soldiers or weapons.",
    "capture_light": "I can't hide or turn off the glasses' capture light: it is on "
                     "whenever the camera is.",
}


def _has_word(text: str, words: tuple[str, ...], plurals: bool = False) -> bool:
    lowered = text.lower()
    for word in words:
        if word.endswith("*"):
            pattern = rf"\b{re.escape(word[:-1])}"
        elif plurals and word not in NO_PLURAL:
            pattern = rf"\b{re.escape(word)}(?:s|es)?\b"
        else:
            pattern = rf"\b{re.escape(word)}\b"
        if re.search(pattern, lowered):
            return True
    return False


def normalise(text: str) -> str:
    """NFKC, plain apostrophes, and no invisible format characters."""
    text = unicodedata.normalize("NFKC", text).translate(APOSTROPHES)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def policy_check(request: str) -> str | None:
    """Return a one-sentence refusal if the request clearly breaks policy, else None."""
    request = normalise(request)
    for pattern in NAMED_ARRIVAL:
        for named in pattern.finditer(request):
            word = named.group(2)
            if named.group("title") or (word[0].isupper() and not word.isupper()
                                        and word not in NOT_NAMES):
                return REFUSALS["person"]
    request = OBJECT_PHRASES.sub(" ", request)
    if IDENTIFY.search(request) or _has_word(request, PERSON_WORDS, plurals=True):
        return REFUSALS["person"]
    if CAPTURE_LIGHT.search(request):
        return REFUSALS["capture_light"]
    return topic_check(request) or (REFUSALS["sensitive"]
                                    if _has_word(request, SENSITIVE_WORDS) else None)


def topic_check(text: str) -> str | None:
    """The covert, health, sports and politics rules alone. They also apply to every word
    the compiler writes, where people and rooms may come up in passing."""
    text = OBJECT_PHRASES.sub(" ", normalise(text))
    if _has_word(text, COVERT_WORDS) or COVERT_PHRASES.search(text):
        return REFUSALS["covert"]
    if _has_word(text, HEALTH_WORDS):
        return REFUSALS["health"]
    if SPORTS_FORM.search(text) or _has_word(text, SPORTS_WORDS):
        return REFUSALS["sports"]
    for compound, back in REFERS_BACK:
        if compound.search(text):
            text = back.sub(" ", text)
    text = ARMS_LOOKALIKES.sub(" ", text)
    if POLITICS_FORM.search(text) or _has_word(text, POLITICS_WORDS):
        return REFUSALS["politics"]
    return None


def slugify(text: str, fallback: str = "skill") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40].strip("_")
    if not slug or not slug[0].isalpha():
        slug = f"{fallback}_{slug}".strip("_")[:40]
    return slug
