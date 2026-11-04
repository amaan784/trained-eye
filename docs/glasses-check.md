# Glasses check (M0)

Before you build the Trained Eye iOS app, this confirms with Meta's own sample app that
your Meta Adventurer glasses and your iPhone can do the three things the project needs:

1. stream camera frames from the glasses to the phone (Meta's Device Access Toolkit, DAT)
2. get the glasses microphone onto the phone
3. play spoken audio from the phone on the glasses speakers

It takes about 45 minutes. Everything below comes from the DAT iOS repository
(github.com/facebook/meta-wearables-dat-ios, version 1.0.1 of October 9, 2026): its
README, CHANGELOG, `.cursor/rules` docs and the CameraAccess sample. The online docs at
wearables.developer.meta.com could not be reached from the build machine.

## What the SDK docs say, and what is still unknown

- The CameraAccess sample says a development build needs only Developer Mode in the
  Meta AI app, with no Wearables Developer Center account, app ID or client token. Two
  other SDK docs say the camera needs approval in the Developer Center, so if the camera
  is refused, that is the next step (see "If it gets stuck").
- Requirements: Xcode 26.4 or newer, iOS 17.2 or newer on the phone.
- The SDK supports a model it calls "Meta Glasses" (since DAT 0.8). Nothing in the SDK
  names Adventurer, so this check is what tells us whether Adventurer works.
- The firmware V128 and Meta AI app V292 numbers in SPEC.md are not in the repository
  (the version table is on the docs site). If something is too old, the sample shows an
  "Update firmware" or "Update app on glasses" button.
- The SDK has no speaker API. Audio reaches the glasses as ordinary Bluetooth audio from
  the phone, which is how Trained Eye will speak its alerts.
- Only one DAT app can use the glasses at a time.

## 1. On the Mac

1. In Terminal, run `xcodebuild -version`. It must say Xcode 26.4 or newer; if not,
   update Xcode from the App Store first.
2. Run `sysctl hw.memsize` and send me the number. It decides in M2 whether the model
   runs on the Mac or on Modal.
3. Clone Meta's repository somewhere outside the trained-eye folder:
   `git clone https://github.com/facebook/meta-wearables-dat-ios`

## 2. On the iPhone

1. Update the Meta AI app from the App Store. Make sure the glasses are paired with it,
   charged, and on the latest firmware the app offers.
2. In the Meta AI app, open Settings, select your glasses, and turn on Developer Mode.
   The glasses may restart. A firmware update turns Developer Mode off again, so check
   it after every update.
3. In the iPhone's own Settings, open Privacy & Security, then Developer Mode, and turn
   it on (the phone restarts). iOS needs this to run apps built in Xcode.

## 3. Build Meta's CameraAccess sample onto the phone

1. Open `meta-wearables-dat-ios/samples/CameraAccess/CameraAccess.xcodeproj` in Xcode.
   Wait until Package Dependencies shows meta-wearables-dat-ios 1.0.1 (Xcode fetches it
   from GitHub).
2. Select the CameraAccess target, open Signing & Capabilities, choose your Apple ID as
   the Team, and change the Bundle Identifier to one of your own, for example
   `com.yourname.cameraaccess`. If Xcode refuses the Hotspot Configuration or Access
   Wi-Fi Information capability for your Apple ID, stop and tell me the exact message.
3. Leave META_APP_ID and CLIENT_TOKEN as they are. Developer Mode does not need them.
4. Connect the iPhone with a cable, pick it as the run destination and press Run
   (Cmd+R). If the phone asks you to trust your developer certificate: Settings, General,
   VPN & Device Management, trust it, and press Run again.

## 4. Camera stream (the main check)

1. Allow the Bluetooth prompt when the app first opens.
2. Tap Connect my glasses. The Meta AI app opens: approve the app, and you come back to
   CameraAccess.
3. Put the glasses on, unfolded. The screen changes from "Waiting for an active device"
   to "Ready".
4. Tap Start session. The session chip at the top left shows that the session started.
5. Tap Preview. The first time, an "Allow camera access" alert appears: tap Continue and
   choose Allow once (or Allow always) in Meta AI. Allow the Local Network prompt too; the
   stream waits for it.
6. Success looks like this: a moving full-screen view from the glasses, and the stream
   chip shows it is streaming, with a green dot. Look at the stove's burner and knob, or
   printed text, and check it is readable. Keep the app in the foreground; leaving it ends the session.
7. Watch the capture LED on the glasses while the preview runs and tell me what it does.
   Trained Eye must never hide it, so I need to know how it behaves.
8. A single tap on the glasses' touchpad pauses the stream ("Paused", "Tap your glasses
   to resume."); tap again to resume.

If it gets stuck:

- "Waiting for an active device", or registration works but nothing connects: check
  Developer Mode is still on, the glasses are unfolded and near the phone, and Bluetooth
  is on.
- "Update required" or "needs an update": tap Update firmware, which opens Meta AI.
  If an error says the app on the glasses is outdated, update it from Meta AI.
- "Camera permission denied": grant camera access to the app in Meta AI. If it is still
  refused, the camera may need approval in the Wearables Developer Center: tell me and I
  will walk you through creating the project.
- The session starts but frames never arrive: in the target's Info tab, add
  `external-accessory` to "Required background modes" and a "Supported external accessory
  protocols" entry with the value `com.meta.ar.wearable`. The SDK docs call both
  required for Bluetooth Classic streaming, but the sample leaves them out. Rebuild and
  try again.

## 5. Glasses microphone

Record a short video in CameraAccess with the microphone toggle on while you say a
sentence, then play it back. Success: your voice is on the recording. The sample records
the glasses mic over Bluetooth (HFP); iOS asks for microphone permission the first time.

## 6. Speakers

CameraAccess plays no audio, so this uses iOS's own text-to-speech, the same system voices
the Trained Eye app will use for alerts.

1. With the glasses on, open Control Center, press and hold the music tile, tap the audio
   output button and check that the glasses are listed and selected.
2. In the iPhone's Settings, open Accessibility, then Spoken Content, and turn on Speak
   Selection.
3. In Notes, type "The burner is still on", select it and tap Speak.
4. Success: you hear it from the glasses' speakers, not the phone.

Optional: Meta's BirdSpotter sample (needs iOS 18) speaks through the glasses too. Build it
like CameraAccess (target `birdspotter`, your Team and bundle identifier), connect it from
Settings, Set up your glasses, then on the Identify tab tap Start listening: after about
8 seconds the glasses should say "I just heard an American Robin."

## 7. Can the phone still reach the Mac while streaming?

DAT can use a Wi-Fi link to the glasses. Trained Eye sends frames from the phone to the
Mac over your home Wi-Fi, so check both work at the same time. CameraAccess ends its
session as soon as it leaves the foreground, so do not switch apps while it streams.

1. Before opening CameraAccess: on the iPhone, open Settings, Wi-Fi, tap the (i) next to
   your home network, and note its IP Address (for example 192.168.1.42). Find the Mac's
   address in System Settings, Wi-Fi, Details (for example 192.168.1.20).
2. On the Mac, run `ping 192.168.1.42` (the iPhone's address) and leave it running. Keep
   the phone unlocked. Replies should arrive before you go on; if none ever do, note that
   and go to step 5.
3. Open CameraAccess, start the preview as in step 4, and leave it streaming in the
   foreground for one minute with the screen on. During that minute, swipe down from the
   top-right corner to open Control Center (this does not end the session), press and
   hold the network tile, note the Wi-Fi name it shows, then close Control Center.
4. Success: ping keeps getting replies for the whole minute (a few lost ones are normal;
   note the usual time in ms), and Control Center still shows your home network.
   Failure: "Request timeout" for more than about 10 seconds in a row, or another network
   name. Note whether replies stop when the preview starts and come back when it stops.
5. After the session, on the Mac in the trained-eye folder (venv active): `te serve`. If
   macOS asks whether Python may accept incoming connections, allow it. Then open
   `http://192.168.1.20:8765/health` (your address) in Safari on the phone. Success:
   `{"ok":true,"protocol":1}`. This shows the phone reaches the Mac after streaming, not
   during it; steps 2 to 4 cover that.

## 8. Without glasses: the Mock Device Kit

Use this for development whenever the glasses are not at hand.

1. Quickest check: choose an iPhone Simulator as the run destination and press Cmd+U.
   Meta's test `testVideoStreamingFlow` pairs a mock pair of glasses, feeds it the bundled
   video and checks that frames arrive.
2. By hand: press Run on the simulator, tap the ladybug button on the right edge, tap
   Enable MockDeviceKit, then Pair Ray-Ban Meta. On the device card, switch on Power,
   Unfolded and Donned.
3. Set a camera source before Preview, or the mock crashes: open "Camera source" and pick
   Video file. To get a video into the simulator's Photos, drag
   `samples/CameraAccess/CameraAccess/TestResources/plant.mp4` onto the simulator window.
   Mock videos must be HEVC (h.265); that file is.
4. Swipe the sheet down, then Connect my glasses (if shown), Start session and Preview. The
   video plays as if it came from glasses.

The CameraAccess mock only offers Ray-Ban Meta. Trained Eye's own app will pair the mock
as the "Meta Glasses" model instead.

## 9. Send me

- the output of `xcodebuild -version` and `sysctl hw.memsize`
- for steps 4 to 8: worked or not, plus the exact text of any error
- step 7: ping replies during the preview (yes, with gaps, or no) and their usual time,
  the Wi-Fi name Control Center showed during the preview, and whether /health loaded
  afterwards
- what the capture LED did during the preview
- anything the Meta AI app asked you to update, and its version numbers if you can see
  them (Meta AI app version, glasses firmware version)

Then end the session and quit CameraAccess before using any other glasses app.

If the camera stream does not work on Adventurer, the fallback from SPEC.md applies: the
iPhone camera watches and the glasses only speak the alerts.
