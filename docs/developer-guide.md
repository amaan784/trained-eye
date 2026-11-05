# Developer guide: from a fresh Mac to an alert on your glasses

One path, in order, as it was run on Nov 4: install the backend, set the keys, teach and
train a skill, serve it, build the iPhone app onto a phone, connect the Meta glasses and
watch. Every step that costs money prints its price and asks first (`--yes` skips the
question). [setup.md](setup.md) and [../SETUP.md](../SETUP.md) have more on accounts.

## What you need before you start

Devices:

| Device | What for | Needs | Tested on Nov 4 |
|---|---|---|---|
| **A Mac** | the backend (`te serve`), teaching, training from the terminal, building the iPhone app | macOS with Xcode 26.4 or newer, Python 3.12, ffmpeg, XcodeGen | Xcode 27.2 |
| **An iPhone** | the app: the glasses' camera stream, Teach, Skills, Watch | iOS 17.2 or newer, Developer Mode on (step 7), a USB cable for the first install | the author's iPhone (and the iOS 27.2 simulator) |
| **Meta AI glasses** | the camera that watches, the microphone for the request, the speakers for the alert | glasses supported by Meta's Wearables Device Access Toolkit (DAT 1.0.1), paired with the Meta AI app, Developer Mode on (step 8) | Meta glasses, software version 122.0, Meta AI app 297.0 |
| **One Wi-Fi network** | the phone sends frames to the Mac | the iPhone and the Mac on the same network | |

No glasses? The app can watch with the iPhone's own camera (Connect, Camera, iPhone), and
Debug builds have mock glasses for the simulator.

Accounts: a Meta Model API key (dev.meta.ai), Modal (for training and serving on an H100),
Hugging Face (a token that can read `meta-models/Muse-Glimmer-30B`) and, optionally,
Weights and Biases. A free Apple ID (a Personal Team in Xcode) is enough to install on
your own iPhone.

Budget, roughly: a skill costs about $5 to $7 of Meta API calls to teach; training costs
$1 to $2 of H100 time; a served model costs about $4 an hour while a container is up (it
sleeps after 5 idle minutes).

## 1. The backend

    brew install python@3.12 ffmpeg xcodegen
    git clone https://github.com/amaan784/trained-eye && cd trained-eye
    python3.12 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev,train]"
    pytest && ruff check .          # offline and free

## 2. Keys

Copy `.env.example` to `.env` and fill it in. Two of the values are made up by you; make
each a long random string:

    python3 -c "import secrets; print(secrets.token_urlsafe(24))"

| In `.env` | What it is |
|---|---|
| `MODEL_API_KEY` | your Meta Model API key |
| `HF_TOKEN` | a Hugging Face token (write access only if you will publish) |
| `STUDENT_API_KEY` | made up: the key of your model's endpoint on Modal |
| `SERVER_TOKEN` | made up: the phone types it to teach (teaching spends money). Empty means teaching from the phone is off |
| `TE_STUDENT_URL` | set in step 5, once the model is served |

Then sign Modal in and give it the same values as Modal Secrets:

    modal token new
    modal secret create huggingface HF_TOKEN=<your token>
    modal secret create student STUDENT_API_KEY=<the same value as in .env>
    modal secret create wandb WANDB_API_KEY=<your key>          # optional
    modal secret create meta MODEL_API_KEY=<your key>           # only for the hosted playground

Check the Meta key for under a cent: `te models`, then `te api-smoke`.

## 3. Teach a skill

Film about 10 seconds of the thing in the state to watch for, moving your head a little
(a phone video or glasses footage; a folder of JPEG frames works too). Then:

    te teach --skill fridge_left_open --clip teach.mp4 \
        --request "Tell me when the fridge door is left open"

It asks Muse Wonder for the skill, SAM 3.1 for the target, Muse Image for 300 variations
and Muse Spark to check each one, and builds the dataset in `data/skills/<skill>/`. Each
skill has a $10 cap: raise it with `TE_SKILL_CAP_USD=25 te teach ...` if a run stops. Use
the project's names (`fridge_left_open`, `burner_left_on`, `tap_left_running`) to get
their holds and repeats; any other name works too.

## 4. Train one model for all your skills

Train every skill into one model, from the base each time a skill is added (one served
model then watches them all):

    te train --skills fridge_left_open,burner_left_on --run product --smoke   # two steps, cheap
    te train --skills fridge_left_open,burner_left_on --run product

The estimate printed before the real run comes from the two-step check and runs long; a
real run of about 600 examples took about 15 minutes on one H100.

## 5. Serve it

    te export --run product              # merges the adapter into the base (about 4 minutes)
    te serve-student --run product       # deploys it on Modal; prints its URL

Put that URL, plus `/v1`, in `.env` as `TE_STUDENT_URL`. The first request wakes the model,
which takes 2 to 4 minutes; the phone shows the wait. `te serve-student --stop` stops it.

## 6. Run the Mac backend

    te serve

It prints the address for the phone, for example `ws://192.168.1.230:8765/ws`, and says
whether teaching from the phone is on. Allow incoming connections if macOS asks. The phone
and the Mac must be on the same Wi-Fi network.

## 7. Build the app onto your iPhone

    xcodegen generate --spec ios/project.yml
    open ios/TrainedEye.xcodeproj

1. In Xcode, select the TrainedEye target, then Signing & Capabilities: tick Automatically
   manage signing, pick your Team (a free Personal Team works) and, if Xcode complains,
   change the Bundle Identifier to one of your own. Don't run XcodeGen again afterwards: it
   resets these.
2. Plug in the iPhone, choose it as the run destination and press Run.
3. If macOS asks for the "login" keychain password for `codesign`, type your Mac password
   and click Always Allow. If the prompt keeps coming back or sticks, stop the build and
   run this once (it asks for the password itself):

       security set-key-partition-list -S apple-tool:,apple:,codesign: -s ~/Library/Keychains/login.keychain-db

4. "Developer Mode disabled": on the iPhone, Settings, Privacy & Security, Developer Mode
   (at the bottom; it appears after the phone has been connected to Xcode once), turn it
   on, restart, and confirm Turn On. Then press Run again.
5. "Untrusted Developer" on the phone: Settings, General, VPN & Device Management, your
   Apple ID, Trust.

## 8. Prepare the glasses

In the Meta AI app: open Settings, then App Info, and tap the App version number five
times; turn on Developer Mode and confirm with Enable. The glasses reconnect. A firmware
update can turn it off again. Quit any other app that uses the glasses' camera.

The alert is spoken by the iPhone's text-to-speech through the glasses, which are an
ordinary Bluetooth audio output: in Control Center, make the glasses the audio output.
The app speaks only when the output is Bluetooth (it never speaks from the phone's
speaker) and says "Not spoken: no Bluetooth audio" otherwise.

## 9. Connect and watch

1. Connect: type `http://<the Mac's address>:8765` and the `SERVER_TOKEN`, tap Check the
   connection and allow Local Network.
2. Tap Connect my glasses: Meta AI opens, approve Trained Eye, come back. Put the glasses
   on, unfolded: the Camera row turns Ready.
3. Skills: switch your skills on. This also wakes the model; the screen shows the wait.
4. Watch: Start watching, and Allow in Meta AI for the camera. The film strip fills with
   the frames the model scored and their scores. When the condition has held for its hold
   (fridge 20 s, burner 10 s), the alert is spoken on the glasses and fills the screen;
   it repeats while the condition stays true (fridge every 30 s, burner every 20 s).

To teach from the phone: Teach tab, hold to talk, look for 10 seconds, name the skill,
send it to the Mac, then Teach it and That's right. Then train and serve again (steps 4
and 5).

## 10. Measure it

Label clips you recorded (`te label --clip <clip>`: mark when the condition is true), list
them in an evaluation plan, then:

    te replay --plan eval/plan.json --run product
    te eval --plan eval/plan.json --out eval --readme README.md

[evaluation.md](evaluation.md) explains the plans, the base Glimmer and Muse Spark
comparisons, and the numbers.

## When something is off

| You see | What it means, and what to do |
|---|---|
| Watch, Details: "Connects when you start watching" | Normal: the app connects to the Mac only while watching. |
| "Waking the model on Modal" | The endpoint was asleep; it answers within about 4 minutes. |
| "the model did not answer" | Check `te serve-student` and `TE_STUDENT_URL` on the Mac. |
| The alert shows but isn't heard | The glasses aren't the audio output (step 8). |
| Teaching from the phone answers 401 | `SERVER_TOKEN` is empty or differs between `.env` and the app; restart `te serve` after changing it. |
| `te teach` stops at "spend caps" | Raise `TE_SKILL_CAP_USD` for that run; finished calls are cached and free. |
| The Camera row never reaches Ready | Developer Mode in Meta AI (step 8); then `glasses-check.md` tells a glasses problem from an app problem. |
