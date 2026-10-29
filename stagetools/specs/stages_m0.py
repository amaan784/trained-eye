"""Per-stage metadata for M0: layer, description, commands, commit message."""

INTRO = """Meta glasses answer when you ask them something, but they can't keep an eye on
anything for you. With Trained Eye you look at something for about ten seconds and
say what to watch for ("let me know when the washer says done"). Meta's models label
that footage and generate training variations, a small open model (Muse Glimmer) is
fine-tuned on them, and from then on the glasses speak up when it happens.

Built for the Meta Global AI Developer Hackathon (October 29 to November 4, 2026).
All code here was written during the event."""

POLICY = """- No skill may identify or track a specific person.
- Watching never hides or suppresses the glasses' capture LED.
- Frames are not stored unless the user saves a clip.
- Muse Image outputs keep their watermark and provenance metadata. Generated images
  are never committed; only the prompt manifest and a script to regenerate them.
- Faces are blurred in anything published."""

PYTEST = "python -m pytest"
RUFF = "ruff check ."

# What each file is, for the cumulative "what is here" list. Shown once the file exists.
COMPONENTS = [
    ("backend/config.py", "settings: model IDs for every job, prices, spend caps and data "
     "paths; secrets come from the environment or a local .env file and are never printed"),
    ("backend/meta/cache.py", "content-addressed cache for Meta API responses: the key is a "
     "SHA-256 of model, endpoint and normalized request, with media replaced by its own hash"),
    ("backend/meta/ledger.py", "spend ledger: estimated cost checked against the per-skill "
     "($10) and global ($120) caps before each call, actual cost recorded after"),
    ("backend/cli.py", "the te command line"),
    ("backend/meta/client.py", "MetaClient, the only way the code calls Meta's API: cache "
     "first, then cap check, retries with backoff for 429, 5xx and timeouts, and spend "
     "recorded from the reported usage"),
    ("backend/server.py", "the backend the iPhone app talks to: JPEG frames arrive over a "
     "WebSocket at /ws and each one is acknowledged; frames stay in memory"),
    ("backend/training/wandb_smoke.py", "a Modal app that logs one Weights & Biases run, "
     "with the key coming only from a Modal Secret"),
    ("docs/glasses-check.md", "how to check, with Meta's own sample app, that the glasses "
     "stream camera frames, deliver the mic and play audio"),
    ("PROGRESS.md", "milestone checklist, decisions, docs-vs-spec differences, open issues "
     "and spend"),
    ("CLAUDE.md", "short setup, commands and conventions"),
]

STAGES = {
    1: dict(
        layer="Settings",
        adds="The project skeleton and the settings module every later layer reads: one "
             "model ID per job (never a '-contributor' variant), Meta's prices, the spend "
             "caps, the data folder, and the three keys, read from the environment or a "
             "local .env file. Keys are never printed, only whether each one is set. "
             "pyproject.toml holds the requirements and the pytest and ruff settings, and "
             "the CI workflow runs ruff and pytest on every push.",
        commands=[PYTEST, RUFF, "python -m backend.config"],
        summary="Add project skeleton and settings module",
        paragraph="Adds pyproject.toml with the test and lint setup, a GitHub Actions "
                  "workflow that runs ruff and pytest, and backend/config.py: model IDs for "
                  "every job, Meta's published prices, the $10 per-skill and $120 global "
                  "spend caps, and keys read from the environment or .env without ever "
                  "being printed. python -m backend.config shows the resolved settings.",
    ),
    2: dict(
        next='the response cache: identical Meta API requests are answered from disk for free.',
        layer="Response cache",
        adds="A content-addressed cache for Meta API calls. The key is a SHA-256 over the "
             "model, the endpoint and the normalized request: keys are sorted, empty values "
             "dropped, and every image or audio clip (raw bytes or a base64 data URL) is "
             "replaced by its own SHA-256, so the same frame always gives the same key. A "
             "schema's property order is kept, because strict structured output follows it. "
             "Generated images are stored byte for byte, so Muse Image watermarks and "
             "provenance metadata survive. Also adds the Trained Eye entries to .gitignore: "
             "data/, outputs/ and copies of .env.",
        commands=[PYTEST, RUFF,
                  "python -c \"from backend.meta.cache import cache_key; "
                  "print(cache_key('sam-3.1', '/responses', {'image': b'frame bytes'}))\""],
        summary="Add content-addressed cache for API responses",
        paragraph="Adds backend/meta/cache.py. Responses are stored under data/cache, keyed "
                  "by a SHA-256 of model, endpoint and normalized request with media "
                  "replaced by its own hash, so reruns of the pipeline cost nothing. Binary "
                  "outputs are written exactly as returned. Extends .gitignore so local "
                  "data, outputs and copies of .env are never committed.",
    ),
    3: dict(
        next='the spend ledger, with per-skill and global caps checked before every paid call.',
        layer="Spend ledger",
        adds="The spend ledger: an append-only JSONL file with one entry per paid call "
             "(model, endpoint, skill, estimated and actual cost, usage). Before a call, "
             "its estimate is checked against the per-skill and global caps; after it, the "
             "actual cost is recorded. Includes the cost helpers for token-priced and "
             "unit-priced models and the spend report.",
        commands=[PYTEST, RUFF,
                  "python -c \"from backend.meta.ledger import Ledger, format_report; "
                  "print(format_report(Ledger('data/ledger.jsonl', 10, 120)))\""],
        summary="Add spend ledger with per-skill and global caps",
        paragraph="Adds backend/meta/ledger.py: cost helpers, an append-only ledger in "
                  "data/ledger.jsonl, cap checks that raise BudgetExceeded before a call "
                  "could push spend past $10 for one skill or $120 overall, and a report "
                  "with totals by model and by skill.",
    ),
    4: dict(
        next='the te command line with a spend report, and test isolation from your shell settings.',
        layer="te command: spend",
        adds="The te command line with its first subcommand, te spend, which prints the "
             "ledger report. Also adds a test fixture that removes every TE_* variable and "
             "every key from the environment before each test, so a developer's own shell "
             "settings can't change test results.",
        commands=[PYTEST, RUFF, "python -m backend.cli spend"],
        note="te spend reads data/ledger.jsonl; no paid call has been made yet, so it shows "
             "zeros.",
        summary="Add te command with spend report",
        paragraph="Adds backend/cli.py and the te console script in pyproject.toml, with "
                  "te spend printing the ledger totals. Adds tests/conftest.py, which keeps "
                  "TE_* overrides and real keys in the developer's shell out of the tests.",
    ),
    5: dict(
        next='the Meta API client: structured output, retries with backoff and conservative spend accounting.',
        layer="Meta API client",
        adds="MetaClient, the single place that calls the Meta Model API. Every call looks "
             "in the cache first (a hit makes no network call and costs nothing), checks "
             "the caps, sends the request and retries 429s, 5xx errors and timeouts with "
             "backoff (honouring Retry-After), but never a 4xx. Spend is counted "
             "conservatively: a retried attempt that may have reached the server is charged "
             "at its estimate, and so is a reply without usable usage. This stage has "
             "structured output (Chat Completions with a strict json_schema, as in Meta's "
             "cookbook) and model listing. The key only goes to the Meta base URL and is "
             "removed from any error text.",
        commands=[PYTEST, RUFF],
        summary="Add Meta API client with cache, ledger and retries",
        paragraph="Adds backend/meta/client.py with structured() and list_models(), built on "
                  "httpx. Each call goes through the cache, the spend caps and a retry loop "
                  "for 429, 5xx and timeouts; 4xx errors fail at once. Refusals, empty or "
                  "non-JSON outputs are billed but never cached. Tests use "
                  "httpx.MockTransport, so nothing reaches the network.",
    ),
    6: dict(
        next='the media calls: Voice Transcribe, SAM 3.1 and Muse Image.',
        layer="Media endpoints",
        adds="The rest of MetaClient: Muse Voice Transcribe (file upload, priced per audio "
             "second), SAM 3.1 on one image (streamed Responses API; a stream that does not "
             "complete is billed but not cached), and Muse Image edits and generation, with "
             "images saved exactly as returned and downloads done without the API key. "
             "These request shapes come from the build spec and still need a live call to confirm.",
        commands=[PYTEST, RUFF],
        summary="Add transcription, SAM and Muse Image calls",
        paragraph="Extends backend/meta/client.py with transcribe(), sam_image(), "
                  "image_edit() and image_generate(). SAM's server-sent events are parsed "
                  "with any line ending, unfinished streams raise instead of being cached, "
                  "and generated images are stored byte for byte so their provenance "
                  "metadata stays intact.",
    ),
    7: dict(
        next='two commands to check a real key (one free, one under $0.01) and the key template.',
        layer="Key check commands",
        adds="Two commands for the first live check with a real key: te models (free) lists "
             "the models the key can use and names any configured model that is missing, "
             "and te api-smoke makes one tiny Muse Spark call through the cache and the "
             "ledger (under $0.01; a second run is a free cache hit). Configuration and API "
             "errors now end with a one-line message instead of a traceback. Adds "
             ".env.example.",
        commands=[PYTEST, RUFF, "python -m backend.cli models"],
        note="With a real MODEL_API_KEY in .env, te models lists the model IDs the key can "
             "use (free), and te api-smoke makes one call to muse-spark-1.3 costing at most "
             "about $0.0024. Neither was run here: the build machine had no key and no "
             "network access to Meta.",
        summary="Add te models and te api-smoke key checks",
        paragraph="Adds te models and te api-smoke to backend/cli.py and a template "
                  ".env.example. Errors from settings, the API or the spend caps print as "
                  "'te: ...' and exit with code 2. Without a key both commands stop before "
                  "any request is made.",
    ),
    8: dict(
        next='the WebSocket backend that receives camera frames from the phone.',
        layer="WebSocket frame server",
        adds="The backend the iPhone app will stream to: a Starlette app with /health and a "
             "WebSocket at /ws (protocol 1: a hello, then one JPEG frame per binary message, "
             "each acknowledged with its size). Bad frames get an error and the session "
             "carries on. Frames are decoded only to check them, never written to disk, and "
             "only their metadata is logged. Adds te serve and te ws-selftest.",
        commands=[PYTEST, RUFF, "python -m backend.cli ws-selftest",
                  "SERVE_AND_CURL"],
        note="te serve keeps running until you press Ctrl+C; the curl line was run in a "
             "second terminal while it was up. The WebSocket at /ws needs the websockets "
             "package, which pip install brings in through uvicorn[standard]. The build "
             "machine did not have it, so there the WebSocket was exercised in process by "
             "te ws-selftest and the tests.",
        summary="Add WebSocket backend that receives camera frames",
        paragraph="Adds backend/server.py (Starlette, /health and /ws) and the te serve and "
                  "te ws-selftest commands. Frames are checked as real JPEGs within size "
                  "limits, acknowledged, and dropped; tests confirm nothing is written to "
                  "disk or logged beyond metadata.",
    ),
    9: dict(
        next='a Modal app that logs one Weights & Biases run.',
        layer="W&B smoke run on Modal",
        adds="A Modal app that logs one Weights & Biases run (20 steps of a synthetic loss) "
             "to the trained-eye project, tagged with the git commit. The W&B key comes only "
             "from a Modal Secret named wandb; the tests parse the script to make sure it "
             "never reads, prints or hard-codes a key. Adds the train extra (modal, wandb).",
        commands=[PYTEST, RUFF, "MODAL_NOT_RUN"],
        summary="Add Modal script that logs a W&B smoke run",
        paragraph="Adds backend/training/wandb_smoke.py and its static tests, plus a train "
                  "extra in pyproject.toml. Running it needs a Modal token and a Modal "
                  "Secret called wandb holding WANDB_API_KEY.",
    ),
    10: dict(
        next="the guide for checking the glasses with Meta's sample app.",
        layer="Glasses check",
        adds="The step-by-step guide for checking the glasses with Meta's own CameraAccess "
             "sample before any Trained Eye iOS code exists: camera stream, glasses mic, "
             "speakers, whether the phone still reaches the Mac while streaming, and the "
             "Mock Device Kit. Also brings LICENSE and .gitattributes from the initial "
             "commit into the folder, so the last stage matches the repository.",
        commands=[PYTEST, RUFF],
        summary="Add glasses check guide",
        paragraph="Adds docs/glasses-check.md, written from the DAT iOS SDK 1.0.1 docs and "
                  "samples: Developer Mode in the Meta AI app, building CameraAccess, what "
                  "success looks like, common errors, and the Mock Device Kit. LICENSE and "
                  ".gitattributes are the files from the initial commit, unchanged.",
    ),
    11: dict(
        next='the project notes: setup and conventions, and the progress log.',
        layer="Project notes",
        adds="CLAUDE.md (short setup, commands and conventions) and PROGRESS.md (milestone "
             "checklist, decisions, a table of where the reachable docs differ from the "
             "spec, open issues and spend so far). This is the finished M0.",
        commands=[PYTEST, RUFF, "python -m backend.cli spend"],
        summary="Add project notes and M0 progress",
        paragraph="Adds CLAUDE.md and PROGRESS.md. PROGRESS.md records what M0 delivered, "
                  "the design decisions, 21 places where Meta's cookbook or the DAT SDK "
                  "differ from the spec, the checks still waiting on the Mac and the "
                  "glasses, and the spend so far ($0.00).",
    ),
}

NEXT_M0_DONE = ("Milestone M1: the teach flow for one skill (spoken request, skill spec, "
                "SAM region of interest, Muse Image variations, Muse Spark labels and an "
                "Axolotl dataset).")
