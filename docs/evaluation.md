# Evaluation method

Every number in the README and in [../eval/results.md](../eval/results.md) comes from
`te eval`, which scores the frame logs that watch mode writes against human or annotation
labels. Nothing is typed in by hand.

## What is measured

Watch mode scores one frame a second. For each skill, over a set of test clips:

- **Frame precision, recall and F1:** a frame says yes when its score is at or over the
  run's threshold, and it is true when its time is inside a labelled range.
- **Events:** a labelled range that lasts at least the skill's hold (the fridge 30 s, the
  tap 20 s, the burner 10 s) is an event. Its alert is due when the hold ends and is on
  time within the skill's window after that. Shorter ranges expect no alert.
- **False alarms per hour:** alerts no event explains.
- **Time to alert, latency per frame, and dollars per hour:** the student's GPU, any Muse
  Spark checks, and a per-frame scorer such as Spark judging every frame.

`backend/eval/metrics.py` defines each of these exactly.

## The experiments

1. Real frames only, generated only, or both (mixed).
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
[../eval/results.md](../eval/results.md) has the tokens and seconds, measured by
`python -m backend.eval.speed` (it sends the student's own request one frame at a time and
writes `eval/speed.json`).

## Thresholds

Each student run's threshold is calibrated on its own calibration clips before any test
clip is scored (`te replay` runs `te calibrate --run`). By default it is the lowest
threshold with at most one false alarm an hour of alerts. Footage with too few events
that last past the hold cannot judge alerts, so a plan can calibrate for the best frame
F1 instead (`"threshold_objective": "frames"`), and the results say so. Muse Spark gives
yes or no, so only a yes counts for it.

## Footage, kept apart

Each plan names its footage, and its results say it in their heading.

- **The author's glasses:** the test session of `docs/recording-protocol.md`, never
  trained on, with session 2 clips for calibration (`eval/plan.json`).
- **Other kitchens (EPIC-KITCHENS-100, head-mounted camera):** fridge windows from two
  kitchens to test and one to calibrate on, labelled from the dataset's public open and
  close annotations (`eval/plan-epic.json`). Few of those openings last past the
  fridge's 30 s hold, so this footage gives frame-level results only, with thresholds
  calibrated for frames. See [data-and-licenses.md](data-and-licenses.md).

## Reproducing

    te replay --plan eval/plan-epic.json --run exp1-mixed      # the served fine-tuned model
    te replay --plan eval/plan-epic.json --run exp4-base --model muse-glimmer
    te replay --plan eval/plan-epic.json --run exp4-spark      # paid per frame; asks
    te replay --plan eval/plan-epic.json --run exp1-epic       # Experiment 1's comparison
    python -m backend.eval.speed --url <endpoint>/v1 --model trained-eye --name finetuned \
        --clip <clip> --skill fridge_left_open --out eval/speed.json   # and --name base
    te eval --plan eval/plan-epic.json --out eval --readme README.md

`te replay` needs `TE_STUDENT_URL` for the endpoint that serves the run (`te
serve-student` prints it). Frame logs land in `data/watch/<run>/<skill>/`.
