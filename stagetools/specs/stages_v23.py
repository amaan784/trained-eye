"""Per-stage metadata for stages v23 to v30: fixes from the review after M1 (F1 to F29)."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("stages_v22", Path(__file__).with_name("stages_v22.py"))
_v22 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v22)
_spec = importlib.util.spec_from_file_location("stages_m1", Path(__file__).with_name("stages_m1.py"))
_m1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m1)

INTRO, POLICY, PYTEST, RUFF = _v22.INTRO, _v22.POLICY, _v22.PYTEST, _v22.RUFF
COMPONENTS = _v22.COMPONENTS
DATE = "2026-10-31"
TEACH_ESTIMATE = ('python -m backend.cli teach --skill washer_done --clip /tmp/te-demo.mp4 '
                  '--request "let me know when the washer says done" --estimate-only')

STAGES = {
    23: dict(
        next="the settings fixes: a fallback switch that fails closed, quoted .env values, "
             "number checks, and data that belongs to the project folder.",
        layer="Fix: settings",
        adds="Fixes from the review after M1. TE_TRAIN_ON_MODEL_OUTPUTS now fails closed: "
             "only 1, true, yes, on, 0, false, no or off are accepted, anything else is an "
             "error (before, a quoted value with a comment, or a spelling such as 'none', "
             "left training on model outputs switched on), and the settings printout shows "
             "the mode. Quoted .env values followed by a comment lose their quotes (a "
             "quoted API key used to be sent with the quotes). Spend caps and retries must "
             "be finite and not negative (nan used to switch a cap off). The .env file, the "
             "ledger and the cache now belong to the project folder, so te sees the same "
             "spend and caps from any folder; the tests point that folder at a temporary "
             "one. The ledger report shows its path relative to where you are.",
        commands=[PYTEST, RUFF, "python -m backend.config"],
        date=DATE,
        summary="Fix settings: fail-closed switch, quotes, project paths",
        paragraph="Makes TE_TRAIN_ON_MODEL_OUTPUTS fail closed and shows it in the settings, "
                  "strips quotes from .env values that have a trailing comment, rejects nan "
                  "and negative numbers, and anchors .env, data/, the ledger and the cache "
                  "to the project folder. Tests cover each case and keep the real .env and "
                  "ledger out of the test run.",
    ),
    24: dict(
        next="the client fixes: the spend caps are checked again before a retry, every "
             "served reply is recorded, and cached replies keep their cost.",
        layer="Fix: API client",
        adds="The spend caps are checked again before retrying an attempt that may have "
             "been billed (a timeout or a 500/502/504), so retries can no longer push spend "
             "past a cap. A reply that was served is recorded in the ledger whatever goes "
             "wrong reading it (usage that is not an object, a null choice, content that is "
             "not text, an infinite token count). The cache keeps what each reply cost, and "
             "CallInfo.billed_usd reports it on a cache hit too.",
        commands=[PYTEST, RUFF, "python -m backend.cli spend"],
        date=DATE,
        summary="Fix client: cap check on retries, record odd replies",
        paragraph="Re-checks the caps before a retry that may have been billed, records any "
                  "served reply in the ledger even when it cannot be read, and stores each "
                  "reply's cost in the cache (CallInfo.billed_usd). Tests cover a retry run "
                  "stopped at the cap and five malformed replies.",
    ),
    25: dict(
        next="the policy and compiler fixes: names in any script, fewer false refusals, and "
             "the user's correction of a wrong read-back.",
        layer="Fix: policy check and compiler",
        adds="The local policy check now catches names with accents or internal capitals, "
             "titles (Dr. Patel), curly apostrophes and invisible characters, and no longer "
             "refuses everyday requests: child lock lights, baby bottle warmers, bus "
             "shelter signs, my espresso shot, Roomba gets stuck, Instant Pot comes to "
             "pressure, Sunday's paper. The compiler re-checks only the model fields that "
             "decide what is watched (a near-miss such as 'someone opens the door' no "
             "longer refuses a washer skill), treats placeholder refusal reasons such as "
             "'None.' or 'N/A' as no refusal, rejects wrongly typed output, and takes an "
             "optional correction from the user, policy-checked like the request.",
        commands=[PYTEST, RUFF,
                  'python -m backend.cli policy "Tell me when José gets home"',
                  'python -m backend.cli policy "Tell me when the child lock light turns off"'],
        date=DATE,
        summary="Fix policy check and compiler wording rules",
        paragraph="Widens the named-person rule (accents, titles, curly apostrophes), stops "
                  "false refusals of common object requests, narrows the re-check of the "
                  "model's wording to the target and trigger, accepts placeholder refusal "
                  "reasons, validates field types, and adds a correction argument to "
                  "compile_skill. Tests add 30 policy cases and the compiler cases.",
    ),
    26: dict(
        next="the keyframe fixes: photos come out upright and ffmpeg leaves the terminal "
             "alone.",
        layer="Fix: keyframes",
        adds="Frames from a folder that carry an EXIF Orientation tag (phone and glasses "
             "photos are often stored sideways) are turned upright once, when they are "
             "loaded, so every later step and the dataset see them the right way up; "
             "untagged frames stay byte for byte. ffmpeg now runs with -nostdin and no "
             "stdin, so a te teach run in the background no longer stops on the terminal, "
             "and answers piped to the prompts reach them.",
        commands=[PYTEST, RUFF],
        date=DATE,
        summary="Fix keyframes: EXIF orientation, ffmpeg stdin",
        paragraph="Bakes EXIF orientation into folder frames when they are loaded and runs "
                  "ffmpeg with -nostdin and stdin closed. Tests check a turned photo comes "
                  "out upright and that decoding a video leaves stdin unread.",
    ),
    27: dict(
        next="the data engine fixes: one rejected image or label no longer stops the run.",
        layer="Fix: data engine failures",
        adds="One image edit rejected for its prompt (400 or 422) or one reply that cannot "
             "be used no longer stops the whole teach run: that variation is recorded in "
             "the manifest with its error and skipped, and a rerun tries it again. More "
             "than max(5, 20 percent) failures still stop the run, so a systematic problem "
             "does not keep paying. A Spark label cut off for length is asked again with a "
             "bigger budget; any other unusable label becomes 'unsure' with an error flag "
             "and is counted apart in the label stats. The manifest records what each "
             "image cost even when it came from the cache, and SAM, the image edits and the "
             "labels can report progress.",
        commands=[PYTEST, RUFF],
        date=DATE,
        summary="Fix data engine: skip single failures, keep costs",
        paragraph="Skips and records a single rejected image edit or unusable label instead "
                  "of aborting, with a cap on failures; retries a label cut off for length "
                  "with 4096 tokens; writes the real cost of cached images to the manifest; "
                  "adds progress callbacks. Tests cover the label cases.",
    ),
    28: dict(
        next="the teach flow fixes: checked labels, keyframe names tied to their content, "
             "corrections and progress lines.",
        layer="Fix: teach flow",
        adds="Your own labels are normalised (keyframes/kf_00_....jpg, Yes, ' NO ' all "
             "work) and checked before any paid call; an entry that matches no keyframe of "
             "this clip stops the run with the list of valid names. Keyframes are named "
             "kf_<position>_<hash>, so a labels file made for another take cannot label the "
             "wrong frames, and an earlier take's keyframes are removed. A correction of a "
             "wrong read-back is sent to the compiler and recorded in request.json. The run "
             "prints a line per step and progress during SAM, the image edits and the "
             "labels, flushed so it shows in a log as it happens, plus how many images or "
             "labels failed.",
        commands=[PYTEST, RUFF],
        date=DATE,
        summary="Fix teach flow: labels, keyframe names, progress",
        paragraph="Normalises and checks the user's labels before any spend, names keyframes "
                  "by position and content hash, removes stale ones, passes corrections to "
                  "the compiler, handles per-item failures and prints progress. Tests cover "
                  "provenance byte for byte, costs across reruns, failures, labels and "
                  "corrections.",
    ),
    29: dict(
        next="the te teach fixes: labels checked before the estimate, a cap warning, "
             "--correct, and clear stops.",
        layer="Fix: te teach",
        adds="te teach checks a --labels file before printing the estimate, prints which "
             "training mode it runs in, and warns when the estimate is more than is left "
             "under the spend caps. --audio and --request can no longer be passed together "
             "(the text used to be ignored), and --correct fixes a wrong read-back; after a "
             "'no' it says how. End of input at a prompt counts as no, Ctrl+C and a cap stop "
             "say that a rerun continues for free, and te api-smoke never caches a reply "
             "that fails its check. .env.example lists the training switch.",
        commands=[PYTEST, RUFF, _m1.FFMPEG_CLIP, TEACH_ESTIMATE],
        note=_m1.FFMPEG_NOTE,
        date=DATE,
        summary="Fix te teach prompts, warnings and corrections",
        paragraph="Checks --labels before the estimate, shows the training mode, warns about "
                  "the caps, makes --audio and --request exclusive, adds --correct, treats "
                  "EOF as no, explains how to continue after Ctrl+C or a cap stop, and "
                  "validates the api-smoke reply. Tests cover both confirmations and every "
                  "new message.",
    ),
    30: dict(
        next="the docs for these fixes: the glasses reachability check and the visual "
             "design rules.",
        layer="Fix: docs and notes",
        adds="The glasses check now tests that the phone reaches the Mac while CameraAccess "
             "is streaming (ping from the Mac during a one-minute preview); before, it "
             "switched to Safari, which ends the stream, so it could not tell. HANDOVER.md "
             "gains the build spec's visual design rules (dark and minimal, one accent colour, "
             "Inter, no gradients, the four signature visuals) for the iOS app and the "
             "playground, and the notes record the review and these fixes.",
        commands=[PYTEST, RUFF, "python -m backend.cli spend"],
        date=DATE,
        summary="Update glasses check, handover and notes after review",
        paragraph="Rewrites step 7 of docs/glasses-check.md to test reachability during "
                  "streaming, adds the visual design rules to HANDOVER.md, and records the "
                  "review and fixes in PROGRESS.md and CLAUDE.md.",
    ),
}

NEXT_M0_DONE = ("The confirmed findings from the second half of the review (F30 to F62), "
                "then milestone M2: Modal QLoRA training on the washer skill, export and "
                "serving, and the watch loop with smoothing, the Spark confirmation and "
                "spoken alerts on the glasses.")
