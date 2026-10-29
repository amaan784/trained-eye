"""Per-stage metadata for M1 (stages v12..v21)."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("stages_m0", Path(__file__).with_name("stages_m0.py"))
_m0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m0)

INTRO, POLICY, PYTEST, RUFF = _m0.INTRO, _m0.POLICY, _m0.PYTEST, _m0.RUFF

M1_CODE = [
    ("backend/skills/spec.py", "the skill spec (version 1), the local policy check that "
     "refuses skills about people, health, covert monitoring or sports, and the plain-English "
     "read-back"),
    ("backend/skills/compiler.py", "the skill compiler: the request and 3 keyframes go to "
     "Muse Wonder (structured output), which returns a spec or a refusal"),
    ("backend/dataengine/keyframes.py", "the keyframe picker: 20 to 30 sharp, distinct frames "
     "from the 10 second teaching clip"),
    ("backend/dataengine/dataset.py", "the Axolotl dataset writer, in Meta's recipe format "
     "(image path part, then text; 512x512 padded PNGs)"),
    ("backend/dataengine/roi.py", "the region of interest: SAM 3.1 on each keyframe; the raw "
     "output is kept and parsing is still to come"),
    ("backend/dataengine/synth.py", "the synthetic set: a deterministic plan of Muse Image "
     "edits of the keyframes (positives, hard negatives, nuisance variation), saved untouched "
     "with a manifest"),
    ("backend/dataengine/labels.py", "Muse Spark labels for every image, the filter that keeps "
     "generated images only when Spark agrees, and agreement stats"),
    ("backend/teach.py", "the teach flow end to end, cached so a rerun costs nothing"),
]
COMPONENTS = []
for path, text in _m0.COMPONENTS:
    COMPONENTS.append((path, text))
    if path == "backend/training/wandb_smoke.py":
        COMPONENTS += M1_CODE
    if path == "docs/glasses-check.md":
        COMPONENTS.append(("docs/recording-protocol.md", "what real clips to record for the "
                           "three skills, and how to keep training and test sessions apart"))

FFMPEG_CLIP = ("ffmpeg -v error -y -f lavfi -i testsrc=duration=10:size=1280x720:rate=30 "
               "-pix_fmt yuv420p /tmp/te-demo.mp4")
FFMPEG_NOTE = ("Video clips are decoded with ffmpeg (on the Mac: brew install ffmpeg). The "
               "demo clip is ffmpeg's built-in test pattern, written to /tmp, outside the "
               "folder.")

STAGES = {
    12: dict(
        next="the skill spec, the local policy check with its refusals, and te policy.",
        layer="Skill spec and policy check",
        adds="The skill spec as a versioned pydantic model (target with its SAM prompt, the "
             "positive condition, near-miss negatives, the alert, cooldown, cloud check, "
             "refusal), the plain-English read-back the user confirms, and a local policy "
             "check. The check refuses, before any API call and at no cost, requests that "
             "would identify or track a person (by name, pronoun or role), concern bodies, "
             "health or medication, monitor someone covertly, watch a sensitive place, or "
             "coach sports. Words that also name objects only count inside phrases, so walk "
             "signs, kitchen scales and meat thermometers pass. te policy runs it on any "
             "sentence.",
        commands=[PYTEST, RUFF, 'python -m backend.cli policy "Tell me when Sam gets here"',
                  'python -m backend.cli policy "let me know when the washer says done"'],
        summary="Add skill spec, read-back and local policy check",
        paragraph="Adds backend/skills/spec.py with the version 1 SkillSpec, the read-back "
                  "sentence, and a rule-based policy check that refuses person, health, "
                  "covert and sports requests without calling any model. Adds te policy and "
                  "pydantic as a dependency.",
    ),
    13: dict(
        next="the skill compiler, which turns a request and 3 keyframes into a spec with Muse Wonder.",
        layer="Skill compiler",
        adds="The skill compiler. The local policy check runs first; otherwise the request "
             "and 3 keyframes go to Muse Wonder with a flat strict JSON schema and a system "
             "prompt that spells out the refusal rules. A model refusal is kept as is (also "
             "a filled-in refusal reason, or a native refusal), and the spec the model wrote "
             "goes through the local check again. Fields are clamped to the spec's limits. "
             "MetaClient.structured gains a validate hook, so an answer that parses but "
             "cannot become a spec is billed once and never cached.",
        commands=[PYTEST, RUFF],
        summary="Add skill compiler using Muse Wonder",
        paragraph="Adds backend/skills/compiler.py: request plus 3 keyframes in, validated "
                  "SkillSpec or refusal out. Adds a validate hook to MetaClient.structured so "
                  "unusable answers are never cached. A local refusal costs nothing; tests run "
                  "against a fake Muse Wonder.",
    ),
    14: dict(
        next="the keyframe picker for the 10 second teaching clip.",
        layer="Keyframe picker",
        adds="Keyframe selection for the teaching clip, from a video file (decoded with "
             "ffmpeg at 3 frames a second) or a folder of frames as the phone sends them. The "
             "blurriest quarter is dropped by the variance of the Laplacian, then the "
             "sharpest frames are taken greedily while their difference hashes stay apart, "
             "giving 20 to 30 keyframes in time order; two frames with the same hash are "
             "never both kept. Decoding is bit-exact, so the same clip always gives the same "
             "bytes and the cache keeps working after an ffmpeg upgrade.",
        commands=[PYTEST, RUFF, FFMPEG_CLIP,
                  "python -c \"from backend.dataengine.keyframes import load_frames, "
                  "select_keyframes; f = load_frames('/tmp/te-demo.mp4'); print(len(f), "
                  "'frames,', len(select_keyframes(f)), 'keyframes')\""],
        note=FFMPEG_NOTE,
        summary="Add keyframe picker for teaching clips",
        paragraph="Adds backend/dataengine/keyframes.py: decode a clip or read a frame folder, "
                  "drop blurry frames, drop near-duplicates, keep 20 to 30 keyframes. Adds "
                  "numpy as a dependency.",
    ),
    15: dict(
        next="the dataset writer in Meta's Axolotl recipe format.",
        layer="Dataset writer",
        adds="The Axolotl dataset writer, following Meta's cookbook recipe: OpenAI-style "
             "messages, user content with the image part first (an absolute path to a PNG) "
             "and the text second, the assistant answering a plain yes or no. Images are "
             "padded to 512x512 with black bars, the recipe's default. The prompt template "
             "is the build spec's; its system turn can be folded into the user text. A "
             "manifest records counts and SHA-256 digests, and maps every example back to "
             "its source and the original image's hash.",
        commands=[PYTEST, RUFF],
        summary="Add Axolotl dataset writer",
        paragraph="Adds backend/dataengine/dataset.py: padded PNGs, the JSONL in the "
                  "cookbook's multimodal format, and a manifest with counts per skill, source "
                  "and answer plus SHA-256 digests of the JSONL and the example ids.",
    ),
    16: dict(
        next="the SAM 3.1 region-of-interest step.",
        layer="Region of interest (SAM 3.1)",
        adds="The region-of-interest step: SAM 3.1 on every keyframe with the spec's noun "
             "phrase, one concept per request. SAM's output format could not be checked "
             "while this was written, so the raw text is stored (and the call cached) and "
             "boxes come from a parser hook that returns none for now. The largest confident "
             "box per frame is picked once boxes exist.",
        commands=[PYTEST, RUFF],
        summary="Add SAM region-of-interest step",
        paragraph="Adds backend/dataengine/roi.py: SAM 3.1 per keyframe through the client, "
                  "raw output saved to roi.json, a parser hook for the boxes, and the "
                  "largest-confident-box rule.",
    ),
    17: dict(
        next="the synthetic set: Muse Image edits of the keyframes with a prompt manifest.",
        layer="Synthetic set (Muse Image)",
        adds="The synthetic set. A deterministic plan turns the spec into Muse Image edit "
             "prompts on the user's own keyframes: positives render the condition, hard "
             "negatives render each near-miss while ruling the condition out, with 11 "
             "nuisance variations (light, angle, distance, blur, occlusion). Every "
             "combination of condition, keyframe and nuisance is used once before any "
             "repeats, so no two requests are identical. Every prompt asks for no people in "
             "view. Images are saved exactly as returned, "
             "and the full prompt manifest is written next to them.",
        commands=[PYTEST, RUFF,
                  "python -c \"from backend.skills.spec import SkillSpec, Target; "
                  "from backend.dataengine.synth import plan_variations; "
                  "s = SkillSpec(name='washer_done', request='let me know when the washer "
                  "says done', target=Target(sam_prompt='washing machine display', "
                  "description='the washer display'), positive='the display shows the cycle "
                  "has finished', negatives=['display shows time remaining', 'display off'], "
                  "alert='Laundry done.'); p = plan_variations(s, ['a' * 64]); "
                  "print(len(p), 'variations'); print(p[0].prompt); print(p[150].prompt)\""],
        summary="Add Muse Image synthetic set with prompt manifest",
        paragraph="Adds backend/dataengine/synth.py: a deterministic plan of positives and "
                  "hard negatives (150 each by default), anchored on the keyframes, with "
                  "nuisance variation; generation through the cache; images stored byte for "
                  "byte; manifest.jsonl with every prompt, source keyframe hash and image "
                  "hash.",
    ),
    18: dict(
        next="Muse Spark labels for every image, with filtering and agreement stats.",
        layer="Labels (Muse Spark)",
        adds="Muse Spark labels. Every image, real or generated, gets a structured label "
             "(yes, no or unsure, with a one-sentence reason) against the skill's condition. "
             "A generated image is kept only when Spark agrees with the label it was "
             "generated for; unsure is dropped. The agreement rates, overall and per kind, "
             "are Experiment 2's raw data.",
        commands=[PYTEST, RUFF],
        summary="Add Muse Spark labels and filtering",
        paragraph="Adds backend/dataengine/labels.py: one structured Spark call per image, the "
                  "keep-only-if-Spark-agrees filter, and agreement statistics overall and for "
                  "positives and hard negatives.",
    ),
    19: dict(
        next="the teach flow end to end, with the fallback for training on real data only.",
        layer="Teach flow",
        adds="The teach flow end to end: request (typed, or audio through Voice Transcribe), "
             "keyframes, skill spec and read-back, then nothing more unless the user "
             "confirms; then region of interest, synthetic set, labels and dataset, each "
             "written under data/skills/<skill>/. Every call is cached, so a rerun costs "
             "nothing. A settings switch, TE_TRAIN_ON_MODEL_OUTPUTS=0, keeps generating and "
             "labelling for evaluation but trains on real frames with the user's own labels "
             "only (the build spec's fallback); without those labels it stops after the "
             "free keyframe step. The skill name is checked before anything else.",
        commands=[PYTEST, RUFF],
        summary="Add end-to-end teach flow",
        paragraph="Adds backend/teach.py, which chains transcription, keyframes, compiler, "
                  "read-back, SAM, Muse Image, Spark labels and the dataset, with a cost "
                  "estimate helper. Adds train_on_model_outputs to the settings. An offline "
                  "end-to-end test runs the whole flow twice against a fake API and checks "
                  "that the second run makes no calls.",
    ),
    20: dict(
        next="te teach, which prints a cost estimate and asks before spending.",
        layer="te teach",
        adds="The te teach command: it measures the clip and the audio, prints a cost "
             "estimate per step, asks before spending, reads the spec back and asks again, "
             "then runs the teach flow and prints the spend report. --estimate-only stops "
             "after the estimate, --labels takes your own keyframe labels, and --yes skips "
             "both questions. A skill name that is not snake_case is rejected first.",
        commands=[PYTEST, RUFF, FFMPEG_CLIP,
                  'python -m backend.cli teach --skill washer_done --clip /tmp/te-demo.mp4 '
                  '--request "let me know when the washer says done" --estimate-only'],
        note="The estimate is an upper bound: each label is priced at its full output budget "
             "and each image at 1,500 input tokens, since the real image token count is not "
             "known yet. " + FFMPEG_NOTE,
        summary="Add te teach command with cost estimate",
        paragraph="Adds te teach to backend/cli.py: per-step cost estimate, two confirmations "
                  "(spend, then the read-back), the teach run and the spend report. Tests "
                  "check that the estimate and a refused request make no API call.",
    ),
    21: dict(
        next="the recording protocol for real clips and the updated project notes.",
        layer="Recording protocol and notes",
        adds="The recording protocol for the real clips the experiments need (three skills, "
             "training and validation sessions apart from the test session, near-misses, "
             "people kept out of view), and the project notes updated for M1. This is the "
             "finished M1.",
        commands=[PYTEST, RUFF, "python -m backend.cli spend"],
        summary="Add recording protocol and M1 progress",
        paragraph="Adds docs/recording-protocol.md and updates PROGRESS.md and CLAUDE.md: M1 "
                  "is done offline, the first live washer run and the real clips are the next "
                  "steps, and spend is still $0.00.",
    ),
}

NEXT_M0_DONE = ("Milestone M2: Modal QLoRA training on the washer skill, export and serving, "
                "and the watch loop with smoothing, the Spark confirmation and spoken alerts "
                "on the glasses.")
