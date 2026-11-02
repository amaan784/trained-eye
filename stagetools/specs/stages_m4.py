"""Per-stage metadata for stages v90 to v105: M4's code (release tools, the playground)
and the notes."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("stages_late", Path(__file__).with_name("stages_late.py"))
_late = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_late)

INTRO, POLICY, PYTEST, RUFF = _late.INTRO, _late.POLICY, _late.PYTEST, _late.RUFF
COMPONENTS, VENV, DATE, stage = _late.COMPONENTS, _late.VENV, _late.DATE, _late.stage

STAGES = {
    90: stage(
        "face blurring for anything published.",
        "Face blurring",
        "backend/release/faces.py blurs boxes with Pillow and leaves out EXIF and other "
        "metadata; face detection uses OpenCV's bundled cascades, which need "
        "opencv-python-headless (an approved download for the Mac, not installed here). "
        "Folders and videos can be blurred, and a mapping file records which original each "
        "blurred keyframe came from.",
        "Add face blurring",
        "Adds faces.py with tests; detection tests skip without OpenCV."),
    91: stage(
        "the prompt manifests and regenerating images from them.",
        "Manifests and regeneration",
        "Generated images are never published, so what regenerates them ships instead. "
        "backend/regenerate.py copies each skill's prompt manifest out of data/ without any "
        "image, and rebuilds the images from a manifest and the original keyframes, with the "
        "estimate first and a report of how many match the recorded hashes.",
        "Add the prompt manifests and regeneration",
        "Adds regenerate.py with tests against a fake Meta API."),
    92: stage(
        "the Hugging Face model card and release folder.",
        "Model card and release folder",
        "backend/release/model_card.py writes the model card from the bundle and, when they "
        "exist, the evaluation results, keeps the license a TODO until the Glimmer card is "
        "checked, assembles the release folder, and uploads only with an explicit "
        "confirmation, a license and the weights, checking afterwards that the repository "
        "has the visibility asked for. Its tests come with the next stage.",
        "Add the model card and release folder",
        "Adds model_card.py."),
    93: stage(
        "the model card's tests.",
        "Model card tests",
        "Tests for the card's wording from the bundle (real only, mixed, fixed steps), the "
        "release folder, and every refused upload: no confirmation, no license, no weights, "
        "and a repository that turns out public when private was asked for.",
        "Test the model card and release folder",
        "Adds tests/test_model_card.py."),
    94: stage(
        "the playground's local data: clips, timelines and results.",
        "Playground: local data",
        "backend/playground/data.py lists the recorded clips with their frame logs and labels, "
        "builds each clip's timeline, reads the evaluation charts and results, and decides "
        "which files may be served.",
        "Add the playground's data access",
        "Adds data.py with tests."),
    95: stage(
        "how each skill was built, for the playground.",
        "Playground: skills",
        "backend/playground/skills.py reads a skill's read-back, keyframes, the training wall "
        "of generated variations with Spark's labels, and the SAM outline when there is a "
        "box (the SAM parser is still a stub, which the page says).",
        "Add the playground's skill pages data",
        "Adds skills.py with tests."),
    96: stage(
        "the playground's spend caps.",
        "Playground: caps",
        "backend/playground/budget.py caps spend per visitor (a cookie and a salted IP hash, "
        "counted apart; an IPv6 address counts by its /64) and per UTC day.",
        "Add the playground's spend caps",
        "Adds budget.py with tests."),
    97: stage(
        "the playground's run accounting and live teaching.",
        "Playground: teach it yourself",
        "backend/playground/runs.py estimates a live run, meters every call against the run's "
        "reservation and against the ledger's caps with every run's calls in flight, and "
        "keeps the state the page polls. backend/playground/api.py checks the policy, the key "
        "and the caps before anything is spent, then compiles the visitor's condition, makes "
        "12 Muse Image variations of the chosen keyframe and labels each with Spark, "
        "returning the images as data URLs and writing them nowhere but the cache. A run "
        "always ends, even when the budget file cannot be saved. Their tests come with the "
        "next stage.",
        "Add the playground's run accounting and live teaching",
        "Adds runs.py and api.py."),
    98: stage(
        "the tests of the playground's runs and live teaching.",
        "Playground: run and teach tests",
        "Tests for the estimate, calls held to the reservation and to the ledger across "
        "runs, refusals, both caps, a budget file that cannot be saved and the files a live "
        "run writes, against the fake Meta API.",
        "Test the playground's runs and live teaching",
        "Adds tests/test_playground_runs.py and tests/test_playground_teach.py."),
    99: stage(
        "the playground page and its style.",
        "Playground: page and style",
        "The page and style sheet: dark and minimal, one accent colour, light and dark mode, "
        "phone and laptop.",
        "Add the playground page and style",
        "Adds index.html and style.css."),
    100: stage(
        "the live timeline and the training wall in the browser.",
        "Playground: signature visuals",
        "timeline.js draws the probability line under the video, synced to it, with the "
        "threshold dashed and the alerts, their labels in two rows so they do not collide; "
        "wall.js flips in the generated variations tagged yes, no or unsure and fades out "
        "the rejected ones.",
        "Add the playground's timeline and wall",
        "Adds timeline.js and wall.js."),
    101: stage(
        "the playground's page logic.",
        "Playground: page logic",
        "app.js wires the tabs, the clips, the skills, teach it yourself and the results to "
        "the API, and clears the last run before a new one starts.",
        "Add the playground page logic",
        "Adds app.js."),
    102: stage(
        "the playground app itself.",
        "Playground: app",
        "backend/playground/app.py serves the page, the static files and the API, sets the "
        "visitor cookie and a content security policy (no framing), takes live runs only as "
        "JSON from its own page with the body size checked as it streams, and serves only "
        "the files the answers link to. Its tests come two stages later.",
        "Add the playground app",
        "Adds app.py."),
    103: stage(
        "te manifests, te regenerate, te blur, te release and te playground.",
        "Release commands",
        "backend/cli_release.py adds the release commands: te manifests, te regenerate (the "
        "estimate first, then a question, always asked above $15), te blur, te release "
        "(assembles the folder; --upload asks) and te playground, which refuses to trust "
        "every proxy for the visitor's address. Its tests come with the next stage.",
        "Add the release and playground commands",
        "Adds the five commands, registered by the te command line."),
    104: stage(
        "the tests of the playground app and the release commands.",
        "Playground app and release tests",
        "Tests over HTTP for the page, the data APIs, a live run, both caps, the cross-site "
        "and size checks, and tests for the five release commands; nothing is sent, billed "
        "or uploaded.",
        "Test the playground app and the release commands",
        "Adds tests/test_playground_app.py and tests/test_cli_release.py."),
    105: stage(
        "the notes for M2 to M4 and the credits.",
        "Notes and credits",
        "HANDOVER.md, PROGRESS.md and CLAUDE.md record what M2 to M4 now have, the steps "
        "that need the Mac, Modal or footage, the decisions taken and the questions for the "
        "author. CONTRIBUTIONS.md credits Meta's cookbook recipe and dataset format, the DAT "
        "SDK with the two copied sample files, Axolotl and the Meta models used.",
        "Update the notes for M2 to M4 and add the credits",
        "Updates HANDOVER.md, PROGRESS.md and CLAUDE.md and adds CONTRIBUTIONS.md."),
}

NEXT_M0_DONE = ("The runs on the Mac and Modal: the first teach, the training smoke and "
                "runs, export and serving, the first Xcode build, the experiments, then the "
                "release and the demo.")
