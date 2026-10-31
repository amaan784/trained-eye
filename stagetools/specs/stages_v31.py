"""Per-stage metadata for stages v31 to v38: the second half of the review after M1
(F30 to F62) and the faults a second check found in the v23 to v30 fixes."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("stages_v23", Path(__file__).with_name("stages_v23.py"))
_v23 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v23)
_spec = importlib.util.spec_from_file_location("stages_m1", Path(__file__).with_name("stages_m1.py"))
_m1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m1)

INTRO, POLICY, PYTEST, RUFF = _v23.INTRO, _v23.POLICY, _v23.PYTEST, _v23.RUFF
COMPONENTS = _v23.COMPONENTS
TEACH_ESTIMATE = _v23.TEACH_ESTIMATE
DATE = "2026-10-31"
# python3 on a Mac can be Apple's 3.9, which pyproject.toml rejects.
VENV = "python3.12 -m venv .venv && source .venv/bin/activate   # Mac: brew install python@3.12"

STAGES = {
    31: dict(
        next="the client fixes: Ctrl+C and failed downloads are charged, empty transcripts "
             "are not cached, stale cache entries are fetched again, and every failed call "
             "says what it cost.",
        layer="Fix: API client, second round",
        adds="A call stopped with Ctrl+C while Meta works on it is now charged in the "
             "ledger (before, it left no entry). An empty transcript is billed but not "
             "cached, with a message that says to check the audio or pass --request. A "
             "cached reply that today's checks reject is fetched again instead of failing "
             "forever. A generated image whose download fails is tried twice, then becomes "
             "an unusable output that the data engine can skip (before, a raw network error "
             "ended the run), and a reply without usable images is charged for the images "
             "asked for. Every failed call carries what it was charged (cost_usd), a reply "
             "cut off for length says so whether it is empty or half a JSON object "
             "(finish_reason), and structured calls can ask the cache alone (cache_only).",
        commands=[PYTEST, RUFF, "python -m backend.cli spend"],
        date=DATE,
        summary="Fix client: charge interrupts, refetch stale cache",
        paragraph="Charges a call interrupted with Ctrl+C, stops caching empty transcripts, "
                  "fetches again a cached reply that no longer passes its check, turns a "
                  "failed image download into a skippable output error, and charges an "
                  "image reply without images for what was asked. Failed calls carry their "
                  "charge, cut-off replies their finish reason, and structured() gains a "
                  "cache-only lookup. Tests cover each case.",
    ),
    32: dict(
        next="the settings and spend report fixes: ~ in TE_DATA_DIR, and a spend table "
             "that lines up for long skill names.",
        layer="Fix: settings and spend report",
        adds="TE_DATA_DIR=~/te-data now means the home folder (before, a folder literally "
             "named ~ was made inside the project, outside .gitignore). The spend report "
             "widens its name column for skill names up to 40 characters and says 1 call, "
             "not 1 calls.",
        commands=[PYTEST, RUFF, "python -m backend.config", "python -m backend.cli spend"],
        date=DATE,
        summary="Fix settings home folder and spend table width",
        paragraph="Expands ~ in TE_DATA_DIR, since .env files are not read by a shell, and "
                  "sizes the spend report's name column to the longest skill name. Tests "
                  "cover both.",
    ),
    33: dict(
        next="the policy and compiler fixes: named people and coaching that the first fixes "
             "let through, corrections that rule something out, and the read-back wording.",
        layer="Fix: policy check and compiler, second round",
        adds="The second check of the earlier policy fixes found that they let through "
             "named people ('when Jake gets ready', 'when Jake's car gets stuck') and "
             "coaching ('count my burpees', 'check my pitching form'); both are refused "
             "again, without refusing Roomba, Instant Pot or Nest requests. Person words "
             "match plurals and more relatives, and the check refuses hiding the glasses' "
             "capture light and more covert, nutrition and supplement wording. A correction "
             "that rules something out ('not the bathroom light, the one on the lid') is no "
             "longer refused for what it rules out, and a refused correction says the "
             "correction caused it. Requests over 500 characters stop before the paid call. "
             "Everything the model writes gets the covert, health and sports rules (before, "
             "only the target and the trigger were checked). The read-back keeps names "
             "such as Instant Pot and display labels such as End, never says 'when when' "
             "or 'the our oven', and duplicate near-misses are dropped.",
        commands=[PYTEST, RUFF,
                  'python -m backend.cli policy "Tell me when Jake gets ready"',
                  'python -m backend.cli policy "Tell me when Roomba\'s bin gets full"'],
        date=DATE,
        summary="Fix policy gaps, corrections and read-back wording",
        paragraph="Splits the named-person rule so possessives always count and 'gets "
                  "ready' is no device state, widens the sports, covert, health and person "
                  "lists, adds a capture-light rule, leaves negated phrases out of a "
                  "correction's check, limits requests to 500 characters, checks all of the "
                  "model's wording against the topic rules, and tidies the read-back. Tests "
                  "cover every new refusal and every request that must still pass.",
    ),
    34: dict(
        next="the keyframe and image fixes: hidden macOS files, WebP and HEIC folders, EXIF "
             "turns, and a manifest that is written even when generation stops.",
        layer="Fix: keyframes and generated images",
        adds="Frame folders skip hidden files such as the ._ copies macOS writes on "
             "external drives (before, they crashed te teach), read WebP, and say clearly "
             "when they hold HEIC photos or an unreadable file. EXIF turns are applied by "
             "the tag alone, so an odd EXIF block can no longer crash the run. The image "
             "manifest is written even when generation stops for too many failures, so it "
             "always describes this run and lists what failed; those stops raise "
             "TooManyFailures, and near-misses that differ only in case or a full stop are "
             "planned once.",
        commands=[PYTEST, RUFF],
        date=DATE,
        summary="Fix frame folders and keep the image manifest current",
        paragraph="Skips hidden files in frame folders, reads WebP, names HEIC and unreadable "
                  "frames in the error, turns EXIF-tagged frames without rewriting EXIF, "
                  "writes manifest.jsonl in a finally block, adds TooManyFailures, and plans "
                  "duplicate near-misses once. Tests cover every EXIF orientation and each "
                  "folder case.",
    ),
    35: dict(
        next="the label and SAM fixes: a label that needed the larger budget is free on a "
             "rerun, and SAM skips a rejected frame like the other steps.",
        layer="Fix: labels and region of interest",
        adds="A label that had to be asked again with the larger token budget is now looked "
             "up under that budget first, so a rerun no longer pays for the cut-off attempt "
             "each time (before, a finished run kept paying for it on every rerun). A label "
             "cut off halfway through its JSON is asked again too. Progress lines count "
             "what failed attempts cost. SAM skips a frame it rejects (400 or 422) or "
             "answers unreadably, and stops after the same number of failures as the other "
             "steps.",
        commands=[PYTEST, RUFF],
        date=DATE,
        summary="Fix label reruns and SAM failure handling",
        paragraph="Looks up the larger-budget label in the cache before sending the small "
                  "one, retries labels cut off mid-answer, adds failed attempts to each "
                  "label's cost, and makes roi.locate skip rejected frames with the same "
                  "failure cap as synth and labels. Tests cover the rerun, the cut-off "
                  "reply and both SAM cases.",
    ),
    36: dict(
        next="the teach flow and te teach fixes: inputs checked before the estimate, a "
             "worst-case figure, and datasets that must hold both answers.",
        layer="Fix: teach flow and te teach",
        adds="te teach now checks the clip, the audio file, the request (its length and the "
             "local policy) and the labels file before it shows the estimate, and the key "
             "before it asks; a mistake is one 'te:' line, never a traceback after you said "
             "yes. The estimate adds a worst case for labels that need the larger budget. "
             "A dataset that would hold only yes or only no stops the run, and the dataset "
             "line shows both counts. With training on model outputs off, your labels must "
             "include both answers. A new spec removes the old dataset, the clip may not be "
             "the skill's own keyframes folder, only keyframe files are cleaned up, "
             "contradicting labels are refused, and labels are saved when a run stops. "
             "After a stop for too many failures, te teach says a rerun pays for them "
             "again. te serve prints the address for the phone and keeps the server's own "
             "log lines.",
        commands=[PYTEST, RUFF, _m1.FFMPEG_CLIP, TEACH_ESTIMATE],
        date=DATE,
        summary="Fix teach input checks, class balance and te serve",
        paragraph="Moves every input check of te teach before the estimate, adds the worst "
                  "case to the estimate, stops one-answer datasets, requires both answers "
                  "among fallback labels, removes a stale dataset, guards the keyframes "
                  "folder, refuses contradicting labels, saves labels on a stop, adds the "
                  "failures hint, and makes te serve print the phone address with the "
                  "server's logs. Tests cover each stop and message.",
    ),
    37: dict(
        next="CI and setup: ffmpeg in CI, Python 3.12 and ffmpeg in the Mac setup, and the "
             "W&B team note.",
        layer="Fix: CI and setup",
        adds="CI installs ffmpeg, so the video tests run there instead of being skipped. "
             "The Mac setup uses python3.12 and installs ffmpeg (python3 on a Mac can be "
             "Apple's 3.9, which the project does not support). .env.example no longer "
             "offers WANDB_ENTITY, which nothing reads; it says to pass the W&B team to "
             "the Modal job instead.",
        commands=[PYTEST, RUFF],
        date=DATE,
        summary="Install ffmpeg in CI and fix the Mac setup",
        paragraph="Adds an ffmpeg install step to the CI workflow, switches the Mac setup in "
                  "CLAUDE.md to python3.12 with ffmpeg, and replaces the unused WANDB_ENTITY "
                  "line in .env.example with a note on passing the team to Modal.",
    ),
    38: dict(
        next="the docs for these fixes: the estimator figures for a real clip, the "
             "recording protocol and the notes.",
        layer="Fix: docs and notes, second round",
        adds="HANDOVER.md and PROGRESS.md give the estimator's figures for a real 10 s clip, "
             "with the worst case, record this round of fixes and the questions it leaves "
             "for the author and the organizers, and add notes for M2 (dataset paths on "
             "the Modal volume) and M4 (shipping what regenerates the generated images). "
             "The recording protocol asks for movement during the teaching look and, with "
             "training on model outputs off, for a look in which the condition becomes "
             "true.",
        commands=[PYTEST, RUFF, "python -m backend.cli spend"],
        date=DATE,
        summary="Update notes, handover and recording protocol",
        paragraph="Updates the cost figures, records the second review round, its open "
                  "questions and the M2 and M4 notes in HANDOVER.md and PROGRESS.md, and "
                  "extends the teaching-clip section of docs/recording-protocol.md.",
    ),
}

NEXT_M0_DONE = ("Milestone M2: Modal QLoRA training on the washer skill, export and serving, "
                "and the watch loop with smoothing, the Spark confirmation and spoken alerts "
                "on the glasses.")
