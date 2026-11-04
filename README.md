<p align="center"><img src="docs/img/banner.png" alt="Trained Eye: teach your glasses to notice anything" width="100%"></p>

# Trained Eye

**Look at something for a few seconds, say what to watch for, and your Meta glasses speak
up when it happens.**

Meta glasses answer when you ask them something, but they can't keep an eye on anything
for you. Trained Eye turns one sentence and a short look into a small model of your own,
fine-tuned from Meta's open-weights Muse Glimmer, that watches for it and tells you.

## How it works

1. **Teach.** Look at the thing and say what to watch for ("tell me when the fridge door
   is left open"). Muse Wonder turns the sentence into a skill, SAM 3.1 finds the target,
   Muse Image makes variations of what you showed, and Muse Spark labels every one.
2. **Train.** Muse Glimmer is fine-tuned on those examples with Meta's QLoRA recipe, on
   one GPU on Modal.
3. **Watch.** The glasses stream to your phone; the fine-tuned model scores a frame a
   second, and once the condition has lasted long enough, the glasses say so.

<p align="center"><img src="docs/img/strip.png" alt="How it learned this fridge: twelve teaching frames on film with SAM's boxes, and the four steps of the teach run with its numbers" width="100%"></p>

<!-- te eval: results start (written by te eval --readme; edits here are lost) -->
## Results

Written by `te eval --readme` from `eval/results.json` (SHA-256 `f40f0118fffc74620c4743009608774ca5f4481d65a711414b86248d967adc93`): 5 of 5 evaluations scored. Every number here is copied from that file; n/a means there was nothing to compute it from.

Test footage: other kitchens (EPIC-KITCHENS-100, head-mounted camera). Results on other footage are reported apart.

In short: on other kitchens, the fine-tuned model spotted the condition about as accurately as base Glimmer and Muse Spark (frame F1 0.88, 0.89 and 0.86), answering in 0.98 s a frame against 2.07 s and 11.16 s, at $3.95 an hour against $16.53 for Spark.

Thresholds were calibrated for frame F1 on calibration clips (this footage has too few events past the skill's hold to judge alerts), so read the frame columns.

An event is caught on time when its alert comes within the skill's window after the skill's hold ends (the condition must stay true that long before it alerts): fridge_left_open, a 30 s hold and a 10 s window.

### Experiment 1: real only, synthetic only, mixed

| Variant | Skill | Frame F1 | Events caught on time | False alarms per hour | Dollars per hour |
|---|---|---|---|---|---|
| mixed | fridge_left_open | 0.88 | 0.20 | 0.0 | $3.95 |
| real, other kitchens (EPIC, never published) | fridge_left_open | 0.89 | 0.20 | 0.0 | $3.95 |

### Experiment 4: fine-tuned Glimmer, base Glimmer, Muse Spark on every frame

| Judge (all skills) | Frame F1 | Frame accuracy | Events caught on time | False alarms per hour | Dollars per hour |
|---|---|---|---|---|---|
| finetuned | 0.88 | 0.96 | 0.20 | 0.0 | $3.95 |
| base | 0.89 | 0.96 | 0.20 | 0.0 | $3.95 |
| spark | 0.86 | 0.95 | 0.20 | 0.0 | $16.53 |

Charts: [Experiment 1](eval/exp1_bars.svg), [Experiment 4 (cost)](eval/exp4_cost.svg).

Regression check: not run yet (te regress).

Muse Spark's label error rate against hand labels (Experiment 2):

- fridge_left_open: not measured yet (te spark-errors)
<!-- te eval: results end -->

<!-- art: models start (written by python -m backend.art.build; edits here are lost) -->
## Meta models, each with a job

| Model | Its job | In this run |
|---|---|---|
| Muse Wonder (`muse-wonder-1.0`) | Turns the request into a skill: what to watch, what counts, what does not. | 1 skill from 1 sentence |
| SAM 3.1 (`sam-3.1`) | Finds the target in the teaching frames. | 22 of 24 teaching frames boxed |
| Muse Image (`muse-image-1.0`) | Makes the variations: the condition, near-misses, other light and angles. | 300 variations made |
| Muse Spark (`muse-spark-1.3`) | Labels every image and keeps the ones it agrees with. | 324 images labelled, 299 variations kept |
| **Muse Glimmer 30B** (`meta-models/Muse-Glimmer-30B`), fine-tuned | Fine-tuned, open weights: watches every frame. | 316 training examples, one epoch |
<!-- art: models end -->

## Quick start

    brew install python@3.12 ffmpeg
    python3.12 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev,train]"
    cp .env.example .env        # add your Meta API key
    te teach --skill fridge_left_open --clip teach.mp4 --request "Tell me when the fridge door is left open"
    te train --skills fridge_left_open --run exp1-mixed --smoke
    te train --skills fridge_left_open --run exp1-mixed
    te serve                    # then connect the iPhone app

Every paid step prints its price and asks first. [docs/developer-guide.md](docs/developer-guide.md)
walks the whole path, from a fresh Mac to an alert spoken on your glasses;
[docs/setup.md](docs/setup.md) and [SETUP.md](SETUP.md) have the accounts and keys.

## Try it

- **In a browser:** the playground replays recorded clips with the model's scores, shows
  how a skill was learned, and lets you teach one live.
- **On your own glasses:** the iPhone app is a developer build for now. It needs Xcode,
  your own keys and Modal, a Mac running `te serve`, and Developer Mode in the Meta AI app
  ([docs/developer-guide.md](docs/developer-guide.md)).

## Documentation

- [Architecture](docs/architecture.md): the teach flow, training and serving, watch mode,
  and every module
- [How it works, with diagrams](docs/how-it-works.html)
- [Setup](docs/setup.md) and [accounts and keys](SETUP.md)
- [Evaluation method](docs/evaluation.md) and the [full results](eval/results.md)
- [Privacy and policy](docs/privacy-and-policy.md)
- [Data and licenses](docs/data-and-licenses.md), with the EPIC-KITCHENS-100 credit
- [Recording protocol](docs/recording-protocol.md)
- [Glasses check](docs/glasses-check.md)
- [iOS app build](ios/README.md)
- [Credits](CONTRIBUTIONS.md), [progress](PROGRESS.md) and [handover notes](HANDOVER.md)

Built for the Meta Global AI Developer Hackathon (October 29 to November 4, 2026). All
code here was written during the event. MIT license, see [LICENSE](LICENSE); the base model is under its own (Apache-2.0).
