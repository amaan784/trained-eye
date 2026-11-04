# Data and licenses

## The author's footage

The teaching clips and the glasses test footage are the author's own. Faces are blurred
in anything published.

## Generated images

Muse Image edits of the author's keyframes train the model. They are never committed or
published: `te manifests` keeps each skill's prompt manifest, and `te regenerate` rebuilds
the images from it and the face-blurred keyframes. Their watermark and provenance
metadata are kept as Muse Image returned them.

## Test footage from other kitchens: EPIC-KITCHENS-100

Besides the author's own glasses footage, the fridge skill is also tested on short
windows of EPIC-KITCHENS-100, head-mounted video of other people's kitchens, reported
apart in the results as "other kitchens (EPIC-KITCHENS-100, head-mounted camera)".
EPIC-KITCHENS-100 is by Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino
Furnari, Evangelos Kazakos, Jian Ma, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will
Price and Michael Wray ("Rescaling Egocentric Vision: Collection, Pipeline and Challenges
for EPIC-KITCHENS-100", IJCV 2022), epic-kitchens.github.io, under CC BY-NC 4.0. It is used
only to test and for comparisons that are not published: never to train the published
model, and none of its frames are published or committed.

The windows were picked from the dataset's public annotations
(github.com/epic-kitchens/epic-kitchens-100-annotations): kitchens with the most fridge
open and close actions, one kitchen to calibrate thresholds on and two to test on. Each
window runs from 60 s before an open to 60 s after its close, plus a few windows with no
event; the frames come from the dataset's own hosts, the URLs its download scripts
(github.com/epic-kitchens/epic-kitchens-download-scripts) use. See
[evaluation.md](evaluation.md).

## The model

Muse Glimmer 30B (`meta-models/Muse-Glimmer-30B`, Apache-2.0 at the pinned revision) is
fine-tuned with Meta's QLoRA recipe from the Meta OSS cookbook. Credits for the cookbook,
the Device Access Toolkit sample code, Inter and the other software are in
[../CONTRIBUTIONS.md](../CONTRIBUTIONS.md).
