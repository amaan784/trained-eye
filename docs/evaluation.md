# Evaluation method

Every number in the README and in [../eval/results.md](../eval/results.md) comes from
`te eval`, which scores the frame logs that watch mode writes against human or annotation
labels. Nothing is typed in by hand.

## What is measured

Watch mode scores one frame a second. For each skill, over a set of test clips:

- **Frame precision, recall and F1:** a frame says yes when its score is at or over the
  run's threshold, and it is true when its time is inside a labelled range.
- **Events:** a labelled range that lasts at least the skill's hold (the fridge 20 s, the
  tap 20 s, the burner 10 s) is an event. Its alert is due when the hold ends and is on
  time within the skill's window after that. Shorter ranges expect no alert.
- **False alarms per hour:** alerts no event explains.
- **Time to alert, latency per frame, and dollars per hour:** the student's GPU, any Muse
  Spark checks, and a per-frame scorer such as Spark judging every frame.

`backend/eval/metrics.py` defines each of these exactly.

## The experiments

1. Real frames only, generated images only (Muse Image edits of the teaching frames), or
   both (mixed).
2. Generated images kept only when Muse Spark agrees with them, or all of them.
3. How many generated images a skill needs.
4. The fine-tuned Glimmer, base Glimmer with the same prompt, and Muse Spark judging
   every frame, on the same frames: accuracy against cost per hour.

## Why the fine-tuned model answers faster

Both Glimmers get the same request for each frame: the same prompt, the same image and the
low reasoning setting. Base Glimmer still writes a short passage of reasoning before its yes
or no, on every frame measured. The fine-tuned model answers at once, because every training
example's answer was the bare word yes or no, so it learned to skip the reasoning. A GPU
writes one token at a time, so the shorter reply comes back sooner. The GPU is paid for by
the hour while watching, so dollars per hour stay the same and only the wait for each
answer changes. The speed table under Experiment 4 in
[../eval/epic/results.md](../eval/epic/results.md) has the tokens and seconds, measured by
`python -m backend.eval.speed` (it sends the student's own request one frame at a time and
writes `eval/epic/speed.json`), with the morning's fridge-only model.

## Thresholds

Each student run's threshold is calibrated on its own calibration clips before any test
clip is scored (`te replay` runs `te calibrate --run`). By default it is the lowest
threshold with at most one false alarm an hour of alerts. Footage with too few events
that last past the hold cannot judge alerts, so a plan can calibrate for the best frame
F1 instead (`"threshold_objective": "frames"`), and the results say so. Muse Spark gives
yes or no, so only a yes counts for it.

## Footage, kept apart

Each plan names its footage, and its results say it in their heading. The product model
(`product`) is one model trained on the fridge and the burner together; the comparisons
are base Glimmer and Muse Spark on the same frames.

- **Other kitchens filmed on phones (the main result, `eval/plan.json`, results in
  [../eval/results.md](../eval/results.md)):** first-person, handheld phone clips of four
  fridges and two stoves, none of which the model was taught on. The fridge's threshold is
  calibrated for frame F1 on a fifth fridge's clip; the burner has no calibration clip of
  its own, so the student models use the default threshold, 0.5 (calibrating on EPIC's
  burner windows, which are mostly burner-on, chose "always yes"). The labels (when the
  door is open, when a burner is lit with no pan on it) were drafted by Claude, an AI
  assistant that is not one of the models compared, from one frame a second, at the
  author's request; each label file says so. A main door pulled only ajar and a freezer
  door open on its own count as not open, as the recording protocol's near-misses do.
- **Other kitchens (EPIC-KITCHENS-100, head-mounted camera, `eval/plan-epic.json`, results
  in [../eval/epic/results.md](../eval/epic/results.md)):** fridge windows from two
  kitchens to test and one to calibrate on, labelled from the dataset's public open and
  close annotations. Few fridge openings last past the hold, so this footage gives
  frame-level results, with thresholds calibrated for frames. Its Experiment 4 rows
  (fine-tuned, base, Spark) are the morning's fridge-only model, `exp1-mixed`; the product
  model's row is scored at the threshold calibrated on the phone clip, as it runs. A burner set was built from
  EPIC's hob annotations too, but it is not reported: they mark a hob as on whenever it is
  lit, which in EPIC is nearly always while cooking with a pan on it, while the burner skill
  watches for a burner on with nothing on it, so every model rightly said no to it. See
  [data-and-licenses.md](data-and-licenses.md).
- **The author's glasses:** the teaching look, and the live test in the app; the fridge's
  teaching clip also takes 12 s from each of two more fridges, so the skill is taught on
  three fridges.

## Reproducing

    te train --skills fridge_left_open,burner_left_on --run product
    te export --run product && te serve-student --run product   # TE_STUDENT_URL to its URL
    te replay --plan eval/plan.json --run product               # your clips: calibrate, then test
    te replay --plan eval/plan.json --run exp4-base --model muse-glimmer   # with the base's URL
    te replay --plan eval/plan.json --run exp4-spark            # paid per frame; asks
    te eval --plan eval/plan.json --out eval --readme README.md
    te replay --plan eval/plan-epic.json --run product          # and exp4-base, exp4-spark
    te eval --plan eval/plan-epic.json --out eval/epic
    python -m backend.eval.speed --url <endpoint>/v1 --model trained-eye --name finetuned \
        --clip <clip> --skill fridge_left_open --out eval/epic/speed.json   # and --name base

`te replay` needs `TE_STUDENT_URL` for the endpoint that serves the run (`te
serve-student` prints it). Frame logs land in `data/watch/<run>/<skill>/`. The morning's
fridge-only model (`exp1-mixed`) and its Experiment 1 comparison (`exp1-epic`) stay in
`eval/plan-epic.json`.
