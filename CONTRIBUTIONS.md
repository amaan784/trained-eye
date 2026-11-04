# Contributions and credits

Trained Eye builds on the work below. Each item says what is used and under which terms.
Check each license again before anything is published.

## Meta OSS cookbook

github.com/meta-models/meta-oss-cookbook at commit b30ca2c, Apache-2.0 (its LICENSE file).

- Training recipe: Meta's QLoRA recipe for Muse Glimmer 30B,
  `muse-glimmer/post-training/sft/axolotl/configs/muse-glimmer-30b-qlora.yaml`.
  `backend/training/axolotl.py` keeps its values; only the paths, the W&B names and the
  gradient accumulation (32 on one H100, as the recipe's README says) differ.
- Training container: the recipe README's pins (Python 3.12, CUDA 13.0, torch 2.12.1,
  torchvision 0.27.1, Axolotl 0.19.0, flash-attn 2.8.3, Cut Cross Entropy at the pinned
  commit), in `backend/training/modal_train.py`.
- Dataset format: the recipe's multimodal chat format in `backend/dataengine/dataset.py`
  (messages, the image part first with an absolute PNG path, then the text; images padded
  to 512x512).
- Evaluation and export: how `backend/training/export.py` loads and prompts the model
  (model classes, chat template arguments, the answer after `to=user`) follows
  `post-training/sft/axolotl/scripts/evaluate.py`.
- Serving: the vLLM image, flags and sampling advice in `backend/inference/modal_serve.py`
  and `backend/inference/student.py` follow `inference-server/vllm.md`.
- Structured output through Chat Completions with a `json_schema` response format, as in
  the cookbook's recipes (`backend/meta/client.py`).

## Meta Wearables Device Access Toolkit (DAT)

github.com/facebook/meta-wearables-dat-ios, version 1.0.1, under the Meta Wearables
Developer Terms (wearables.developer.meta.com/terms) and the Acceptable Use Policy. This is
not an open-source license.

- The iOS app uses the SDK through Swift Package Manager: MWDATCore, MWDATCamera, and
  MWDATMockDevice in debug builds.
- Sample code copied without changes, with Meta's header kept and the SDK's LICENSE next
  to it in `ios/TrainedEye/Meta/`. Check that the terms allow publishing these files.

| File in this repository | Source in meta-wearables-dat-ios 1.0.1 | Changes |
|---|---|---|
| `ios/TrainedEye/Meta/VideoFrameDecoder.swift` | `samples/CameraAccess/CameraAccess/Media/VideoFrameDecoder.swift` | none |
| `ios/TrainedEye/Meta/HEVCKeyframe.swift` | the `isHEVCKeyframe()` extension in `samples/CameraAccess/CameraAccess/Media/VideoCaptureHandler.swift` | moved into its own file |

The rest of the app was written for this project, following how the CameraAccess and
BirdSpotter samples use the SDK (`ios/README.md` lists which sample each part follows).

## Meta SAM 3 protocol

github.com/meta-models/meta-sam at commit 9348a8a, under Meta's SAM License (not an
open-source license). `backend/dataengine/sam_text.py` is this project's own reader of the
segmentation text format that `protocol/sam3.md` defines; no code or test data from that
repository is copied here, and Meta's `meta-sam-parser` package is not used.

## Inter

github.com/rsms/inter, release 4.1, under the SIL Open Font License 1.1, which allows
bundling with software. The variable web font is in `backend/labeltool/static/fonts/`
with the license as `LICENSE.txt`; the app and the playground use Archivo.

## Archivo

github.com/Omnibus-Type/Archivo, copyright 2020 The Archivo Project Authors, under the SIL
Open Font License 1.1 with no Reserved Font Name. The variable web font (Latin, width 62 to
125, weight 100 to 900) is in `backend/playground/static/fonts/` and is drawn into the
README art. Four static instances cut from it with fontTools' instancer
(`python -m backend.art.fonts`: Display, Text, TextSemibold and Edge) are in
`ios/TrainedEye/Fonts/`. Each folder has the license as `Archivo-OFL.txt`.

## The contact-sheet design

The author's design kit: the colours, the type, the film strip and the hand-drawn ring.
`backend/art/film.py` ports its `make_banner.py` and `make_strip.py` previews, filled from
real runs instead of fixed frames.

## Axolotl

github.com/axolotl-ai-cloud/axolotl, version 0.19.0: the fine-tuning framework that runs
the recipe, with its Cut Cross Entropy plugin. Its license (Apache-2.0 as far as we know)
was not checked from the build machine.

## Meta models

Used through the Meta Model API; no `-contributor` variants.

| Model | Used for |
|---|---|
| Muse Voice Transcribe 1.0 (`muse-voice-transcribe-1.0`) | the spoken teach request to text |
| Muse Wonder 1.0 (`muse-wonder-1.0`) | a request and three keyframes to a skill spec, or a refusal |
| SAM 3.1 (`sam-3.1`) | the region of interest on each keyframe |
| Muse Image 1.0 (`muse-image-1.0`) | the synthetic variations, edits of the user's own keyframes; they keep their watermark and provenance metadata and are never published |
| Muse Spark 1.3 (`muse-spark-1.3`) | yes/no labels for the training images, and the cloud check before an alert |

The student is Muse Glimmer 30B (`meta-models/Muse-Glimmer-30B` on Hugging Face, revision
a4e59da52a7bc87ae7251dd5545c0dd437c44b68, as pinned by the cookbook recipe), fine-tuned
with QLoRA. Its model card at that revision gives the license as Apache-2.0 (read on
2026-11-02; the text is in `backend/release/BASE_LICENSE.txt` and ships as `LICENSE` with any
release), with Meta's Muse Glimmer Usage Policy beside it, which any use must follow.

## EPIC-KITCHENS-100

Test footage from other kitchens: EPIC-KITCHENS-100 (Damen et al., "Rescaling Egocentric
Vision: Collection, Pipeline and Challenges for EPIC-KITCHENS-100", IJCV 2022,
epic-kitchens.github.io), CC BY-NC 4.0. Its public annotations
(github.com/epic-kitchens/epic-kitchens-100-annotations) give the fridge open and close
times; its videos come from the dataset's own hosts (data.bris.ac.uk), the URLs its
download scripts (github.com/epic-kitchens/epic-kitchens-download-scripts) use. Used for
testing and unpublished comparisons only; no frame of it is committed or published, and
the published model is never trained on it.

## Other software

Modal (training and serving), Weights & Biases (experiment tracking), PyTorch and
flash-attn (in the training container), vLLM (`vllm/vllm-openai:muse-glimmer`, serving),
ffmpeg (video), OpenCV's Haar cascades through opencv-python-headless (face detection for
blurring published media), the Hugging Face Hub (hosting the release), and the Python
packages httpx, numpy, Pillow, pydantic, Starlette, uvicorn, pytest and ruff.
