"""Per-stage metadata for stages v75 to v89: M3's code (labelling, evaluation, the four
experiments)."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("stages_late", Path(__file__).with_name("stages_late.py"))
_late = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_late)

INTRO, POLICY, PYTEST, RUFF = _late.INTRO, _late.POLICY, _late.PYTEST, _late.RUFF
COMPONENTS, VENV, DATE, stage = _late.COMPONENTS, _late.VENV, _late.DATE, _late.stage

STAGES = {
    75: stage(
        "the labelling tool for generated images, for Experiment 2.",
        "Labelling tool: images",
        "backend/labeltool/images.py shows sampled generated images one at a time with the "
        "question Spark was asked (never Spark's own answer, nor the file name, which says "
        "what the image was made to show), takes y, n, u and backspace, and saves after every "
        "answer, so labelling can resume. sample_images picks generated images from a skill's "
        "labels.jsonl, the same ones for the same seed. The page and the shared style sheet "
        "are in backend/labeltool/static/. Its tests come two stages later.",
        "Add the image labelling tool",
        "Adds the image labeller, its page and the shared style sheet."),
    76: stage(
        "the labelling tool for clips: where the condition is true.",
        "Labelling tool: clips",
        "backend/labeltool/clips.py plays a recorded clip (with range requests, so seeking "
        "works), marks where the condition becomes true and stops with [ and ], and saves "
        "sorted, merged, checked time ranges in the format te calibrate reads.",
        "Add the clip labelling tool",
        "Adds the clip labeller and its page."),
    77: stage(
        "the labelling tools' tests.",
        "Labelling tool tests",
        "Tests for both labellers: sampling, saving after every answer and resuming, Spark's "
        "answer kept out of what the page gets, range requests, and saved ranges that are "
        "sorted, merged and read back by te calibrate, with bad ranges (including numbers "
        "too large for a float) refused.",
        "Test the labelling tools",
        "Adds tests/test_labeltool_images.py and tests/test_labeltool_clips.py."),
    78: stage(
        "the evaluation metrics.",
        "Evaluation metrics",
        "backend/eval/metrics.py scores watch frame logs against labelled time ranges: frame "
        "precision, recall and F1, events caught within 5 s and at all, false alarms per "
        "hour, time to alert, latency per frame, and cost per hour (GPU, Spark checks, a "
        "per-frame scorer). Overlapping or touching ranges count as one event. Its tests "
        "come with the next stage.",
        "Add the evaluation metrics",
        "Adds the metrics."),
    79: stage(
        "the metrics' tests and the shared SVG pieces for the charts.",
        "Metric tests and chart building blocks",
        "The metrics' tests cover their edge cases and a log written by the watcher itself. "
        "backend/eval/svg.py holds the palette, the light and dark styles inside each SVG, "
        "escaping, axes and marks (matplotlib cannot be installed here, so the charts are "
        "written by hand).",
        "Test the metrics; add the SVG chart building blocks",
        "Adds tests/test_eval_metrics.py and svg.py."),
    80: stage(
        "the experiment charts.",
        "Experiment charts",
        "backend/eval/charts.py draws grouped bars (Experiments 1 and 2), F1 against the "
        "number of synthetic images with the cost of each point (Experiment 3), and accuracy "
        "against dollars per hour with the fine-tuned point highlighted (Experiment 4).",
        "Add the experiment charts",
        "Adds charts.py with tests, and the tests of svg.py."),
    81: stage(
        "the live timeline chart.",
        "Timeline chart",
        "backend/eval/timeline.py draws the student's probability over a clip, the threshold "
        "dashed, the labelled true ranges shaded and the alerts with their spoken text.",
        "Add the live timeline chart",
        "Adds timeline.py with tests."),
    82: stage(
        "results.md and the chart files.",
        "Results write-up",
        "backend/eval/writeup.py writes results.md (tables per experiment, a note where an "
        "experiment has no runs yet) and the chart files, every number from the metrics.",
        "Add the results write-up",
        "Adds writeup.py with tests."),
    83: stage(
        "the evaluation plan and te eval's engine.",
        "Evaluation runner",
        "backend/eval/report.py reads an evaluation plan (runs, clips, labels, prices), "
        "scores what exists, lists the rest as pending, writes results.json, results.md and "
        "the charts, and logs to W&B only when asked and installed.",
        "Add the evaluation runner",
        "Adds report.py with tests for the plan format, pending runs and the output."),
    84: stage(
        "Muse Spark as the per-frame scorer, for Experiment 4.",
        "Spark as a scorer",
        "backend/eval/spark_scorer.py lets watch mode use Muse Spark itself on every frame "
        "(yes 1.0, no 0.0, unsure 0.5), through the cache and the ledger, and records the "
        "latency of the frames it actually sent.",
        "Add Muse Spark as a per-frame scorer",
        "Adds SparkScorer with tests."),
    85: stage(
        "Spark's error rate and the unfiltered arm, for Experiment 2.",
        "Experiment 2",
        "backend/eval/spark_errors.py compares Spark's labels with the author's own on the "
        "hand-labelled images (error rate with a 95% interval, confusion matrix, the error of "
        "the generated-for labels and of the kept ones), and writes the unfiltered dataset "
        "that keeps every generated image with the label it was made for.",
        "Add Experiment 2's error rate and unfiltered arm",
        "Adds spark_errors.py with tests."),
    86: stage(
        "the experiment plan with fixed steps and prices.",
        "Experiment plan",
        "backend/eval/plan.py lists every training run of Experiments 1 to 3 with its te "
        "train arguments. Every run trains the same number of optimizer steps, so the "
        "image-count curve measures data, not steps; runs that would train the same bundle "
        "are marked. It prices them from the smoke timing and writes the evaluation plan.",
        "Add the experiment plan",
        "Adds plan.py with tests."),
    87: stage(
        "the regression check on general questions.",
        "Regression check",
        "backend/eval/regression.py asks a base and a fine-tuned endpoint the author's general "
        "questions, grades the answers (a number inside a longer one does not count, and a "
        "choice may be written in bold), and reports accuracy for each and the questions "
        "whose result changed.",
        "Add the regression check",
        "Adds regression.py with tests."),
    88: stage(
        "te eval, te label, fixed steps in te train and Spark in te watch.",
        "Commands: eval, label, steps, Spark",
        "te eval (and make eval) runs the evaluation plan; te label starts the labelling tool "
        "on a clip or on a skill's generated images; te train takes --max-steps, which the "
        "Modal app passes on; te watch takes --scorer spark (it prices the frames and asks "
        "first, and only Spark's yes can fire) and --log, which starts the log afresh.",
        "Add te eval, te label and the experiment options",
        "Adds the commands and options, the Makefile and the latency flag in the frame log, "
        "with tests."),
    89: stage(
        "te experiments, te unfiltered, te spark-errors and te regress.",
        "Experiment commands",
        "backend/cli_experiments.py adds te experiments (the plan and its price, and "
        "eval/plan.json), te unfiltered, te spark-errors and te regress, which uses the "
        "student's sampling for both models.",
        "Add the experiment commands",
        "Adds the four commands, registered by the te command line, with tests."),
}

NEXT_M0_DONE = ("Milestone M4: the release tools (manifests, regeneration, face blurring, the "
                "model card) and the web playground.")
