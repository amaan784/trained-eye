# Evaluation results: other kitchens filmed first-person on phones (handheld; four fridges and two stoves the model was not taught on; labels drafted by an AI assistant, see docs/evaluation.md)

Test footage: other kitchens filmed first-person on phones (handheld; four fridges and two stoves the model was not taught on; labels drafted by an AI assistant, see docs/evaluation.md). Results on other footage are reported apart.

Thresholds here were calibrated for frame F1 on the calibration clips, not for alerts: this footage has too few events that last past the skill's hold to judge alerts on, so read the frame columns.

Computed by `te eval` from the plan plan.json: the frame logs of `te watch` scored against the human labels of each test clip. Every number below comes from that computation, and results.json holds all of them. n/a means there was nothing to compute it from (for example, no alerts give no time to alert).

How to read the tables: a frame counts as yes when the student's probability is at or over the skill's threshold, shown with where it came from (run: the run's own calibration on its validation clips; skill: the skill-wide one; flag: given by hand; default: 0.5, not calibrated; spark: only a yes). A skill alerts only once its condition has stayed true for the skill's hold, so an event is one labelled true range that lasts at least the hold, and its alert is due when the hold ends. It is caught on time when its first alert after that comes within the skill's on-time window: burner_left_on, a 10 s hold and a 5 s window; fridge_left_open, a 20 s hold and a 10 s window. False alarms are alerts no event explains: on frames labelled false, or before the hold ended. Dollars per hour add the student's GPU (nothing on the Mac), the Muse Spark checks and any per-frame scorer, at the prices in the plan.

## Experiment 1: real frames only, generated images only, or both

No runs yet: the plan has no evaluations for this experiment.

## Experiment 2: labels filtered by Muse Spark, or not

No runs yet: the plan has no evaluations for this experiment.

Muse Spark's labels against your hand labels (te spark-errors, 95% Wilson intervals), and how often Spark agreed with the label each generated image was made for (te teach's label_stats.json; the filter keeps those):

| Skill | Hand-labelled | Spark wrong | Spark unsure | Generated-for label wrong (unfiltered arm) | Kept by the filter but wrong (filtered arm) | Spark agreed with the label |
|---|---|---|---|---|---|---|
| burner_left_on | not measured yet (te label --skill, then te spark-errors) | n/a | n/a | n/a | n/a | 287 of 289 kept (99.3%) |
| fridge_left_open | not measured yet (te label --skill, then te spark-errors) | n/a | n/a | n/a | n/a | 295 of 300 kept (98.3%) |

## Experiment 3: how many generated images

No runs yet: the plan has no evaluations for this experiment.

## Experiment 4: fine-tuned Glimmer, base Glimmer, Muse Spark on every frame

![Experiment 4: fine-tuned Glimmer, base Glimmer, Muse Spark on every frame](exp4_cost.svg)

| Run | Variant | Skill | Clips | Footage | Frames | Precision | Recall | F1 | Events caught on time | Events caught at all | False alarms per hour | Time to alert after the hold, median (mean) | Threshold (from) | Frame accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| phone-exp4-finetuned-fridge_left_open | finetuned | fridge_left_open | 4 | 7.9 min | 476 | 0.93 | 0.96 | 0.95 | 2 of 3 | 2 of 3 | 7.6 | 0.0 s (0.0 s) | 0.83 (run) | 0.95 |
| phone-exp4-base-fridge_left_open | base | fridge_left_open | 4 | 7.9 min | 464 | 0.86 | 0.97 | 0.91 | 2 of 3 | 2 of 3 | 7.6 | 0.0 s (0.0 s) | 0.50 (run) | 0.91 |
| phone-exp4-spark-fridge_left_open | spark | fridge_left_open | 4 | 7.9 min | 476 | 0.87 | 0.98 | 0.92 | 2 of 3 | 2 of 3 | 7.6 | 0.0 s (0.0 s) | 1.00 (spark) | 0.91 |
| phone-exp4-finetuned-burner_left_on | finetuned | burner_left_on | 2 | 3.6 min | 214 | 1.00 | 0.47 | 0.64 | 0 of 2 | 1 of 2 | 0.0 | 13.0 s (13.0 s) | 0.50 (default) | 0.70 |
| phone-exp4-base-burner_left_on | base | burner_left_on | 2 | 3.6 min | 214 | 1.00 | 0.65 | 0.79 | 1 of 2 | 1 of 2 | 0.0 | 0.0 s (0.0 s) | 0.50 (default) | 0.80 |
| phone-exp4-spark-burner_left_on | spark | burner_left_on | 2 | 3.6 min | 214 | 0.98 | 0.96 | 0.97 | 1 of 2 | 2 of 2 | 16.8 | 9.5 s (9.5 s) | 1.00 (spark) | 0.97 |

Speed and cost:

| Run | Latency median | Latency p90 | Student errors | Alerts | Spark checks per hour | Dollars per hour | Prices used |
|---|---|---|---|---|---|---|---|
| phone-exp4-finetuned-fridge_left_open | 1.58 s | 2.52 s | 0 | 5 | 0.0 | $3.95 | GPU $3.95 per hour, $0.005 per Spark check |
| phone-exp4-base-fridge_left_open | 2.34 s | 4.15 s | 12 | 5 | 0.0 | $3.95 | GPU $3.95 per hour, $0.005 per Spark check |
| phone-exp4-spark-fridge_left_open | 11.67 s | 17.30 s | 0 | 5 | 0.0 | $16.53 | GPU $0.00 per hour, $0.005 per frame scored |
| phone-exp4-finetuned-burner_left_on | 1.69 s | 2.19 s | 0 | 2 | 33.6 | $4.10 | GPU $3.95 per hour, $0.005 per Spark check |
| phone-exp4-base-burner_left_on | 3.22 s | 4.65 s | 0 | 4 | 67.3 | $4.26 | GPU $3.95 per hour, $0.005 per Spark check |
| phone-exp4-spark-burner_left_on | 11.93 s | 13.83 s | 0 | 5 | 84.1 | $16.53 | GPU $0.00 per hour, Spark checks not priced in the plan, $0.005 per frame scored |

Pooled over the skills (the cost chart's points; dollars per hour weighted by hours of footage):

| Judge | Skills | Frames | F1 | Frame accuracy | Events caught on time | False alarms per hour | Dollars per hour |
|---|---|---|---|---|---|---|---|
| finetuned | fridge_left_open, burner_left_on | 690 | 0.87 | 0.87 | 2 of 5 | 5.2 | $4.00 |
| base | fridge_left_open, burner_left_on | 678 | 0.88 | 0.87 | 3 of 5 | 5.2 | $4.05 |
| spark | fridge_left_open, burner_left_on | 690 | 0.94 | 0.93 | 3 of 5 | 10.4 | $16.53 |

## Regression check

Not run yet: te regress writes eval/regression.json.
