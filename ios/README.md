# Trained Eye for iPhone

The phone half of Trained Eye. It streams the glasses' camera to the backend (`te serve` on
the Mac), speaks alerts on the glasses, and teaches new skills: you say what to notice, show
it for 10 seconds, agree to the cost, and confirm the read-back.

**This code has not been compiled.** It was written on a machine with no Mac, Xcode or Swift
compiler, following Meta's DAT 1.0.1 samples (CameraAccess and BirdSpotter) as closely as
possible. Expect a first round of build fixes in Xcode. The open points are listed at the
end, under "TODOs and unverified points".

## What it does

- **Connect**: the backend address (local Wi-Fi or a tunnel), registration with the Meta AI
  app, the glasses' name, link and battery, and the camera choice (glasses, else iPhone).
  Debug builds add DAT's mock glasses.
- **Teach**: a skill name, hold to talk (or type) the request, a 10 second capture after a
  3 second countdown, then the Mac's cost estimate to agree to, the read-back to confirm or
  correct, and the run's progress.
- **Skills**: the taught skills from `GET /skills`, an on/off switch for each, and their
  status: Training (a `te train` run is going on Modal), Trained (a run finished), Ready (the
  model `te serve-student` deployed knows it and `te calibrate` set its threshold), else how
  far the teach flow got. Each skill also shows how long it waits before alerting (its
  hold: the fridge 30 s, the tap 20 s, the burner 10 s) and how long it stays quiet after.
- **Watch**: start and stop watching, the camera-on state, the frame the backend sees, the
  last alert (and whether it was spoken), each skill's score, the connection and the battery.
  It says when the glasses are folded or out of reach, and after a stay in the background
  (the screen locked) it shows what the camera gave and the socket sent and got acked meanwhile.
- **Clips**: the clips you saved with "Save the last 10 seconds", and nothing else.

While any camera is on, a bar above every screen says so. The glasses stream HEVC (`hvc1`)
at low resolution (360x640) and 7 fps while watching, which DAT keeps going in the
background, and at medium resolution (504x896) during a teach capture; one frame a second
goes to the backend as JPEG over the WebSocket (three a second during a teach capture), and
a keepalive ping goes every 20 seconds. Without the glasses, the iPhone's back camera does
the same at its own 640x480, with the app open.

While watching, the glasses' camera comes back by itself: after a fold or a link drop a new
session starts once they are available again (never while a session is paused), and a
stream that stops on its own is added again, a few times. A teach session still running on
the Mac when the app was closed is picked up again at the next launch.

## What you need

- A Mac with Xcode 26.4 or newer (DAT 1.0.1 needs it).
- An iPhone on iOS 17.2 or newer, with Developer Mode on (Settings, Privacy & Security,
  Developer Mode).
- The Meta AI app with the glasses paired and Developer Mode on there (Settings, your
  glasses, Developer Mode). `docs/glasses-check.md` walks through this with Meta's own
  CameraAccess sample; do that check first. Developer Mode turns itself off after a
  firmware update.
- The backend: `te serve` on the Mac (see the root README). Teaching also needs
  `MODEL_API_KEY` and `SERVER_TOKEN` in `.env`; watching needs the trained model's endpoint
  (`TE_STUDENT_URL`).

## 1. Create the Xcode project

The quick way: `ios/project.yml` describes the project with every setting below (sections
1 to 3), and XcodeGen writes it.

    brew install xcodegen
    xcodegen generate --spec ios/project.yml     # writes ios/TrainedEye.xcodeproj
    open ios/TrainedEye.xcodeproj

Then set your Team and your own bundle identifier (it starts as `com.example.trainedeye`)
in Signing & Capabilities, and go on with section 6. To check that it compiles without a
phone or signing:

    xcodebuild -project ios/TrainedEye.xcodeproj -scheme TrainedEye -sdk iphonesimulator \
      -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO build

By hand instead: Xcode would replace an existing `TrainedEye` folder, so make the project
somewhere else first and move it in:

1. In Xcode, choose File, New, Project, iOS, App. Product Name `TrainedEye`, Interface
   SwiftUI, Language Swift, Testing System None, Storage None. Save it on the Desktop and
   leave "Create Git repository" off.
2. Move two things into this folder, then delete the rest of the template (its
   `TrainedEyeApp.swift` and `ContentView.swift` are replaced by the files here):

       mv ~/Desktop/TrainedEye/TrainedEye.xcodeproj ios/
       mv ~/Desktop/TrainedEye/TrainedEye/Assets.xcassets ios/TrainedEye/

3. Open `ios/TrainedEye.xcodeproj`. Xcode 16 and newer projects follow the `TrainedEye`
   folder on disk, so every Swift file in it is part of the target. If the navigator shows
   red (missing) files instead, delete those references and add the `ios/TrainedEye` folder
   with File, Add Files, "Create folders", without copying.
4. In the target's General tab: Minimum Deployments iOS 17.2, iPhone only, Portrait only.
   In Signing & Capabilities: your Team and your own bundle identifier.

## 2. Add Meta's DAT package

File, Add Package Dependencies, enter `https://github.com/facebook/meta-wearables-dat-ios`,
Dependency Rule "Exact Version" `1.0.1`, and add these products to the TrainedEye target:
`MWDATCore`, `MWDATCamera` and `MWDATMockDevice`. The code imports `MWDATMockDevice` only
in Debug builds (`#if DEBUG`), as CameraAccess does; it can be dropped from the target's
frameworks for a release build.

## 3. Build settings

In the target's Build Settings:

- **Info.plist File**: `TrainedEye-Info.plist` (keep Generate Info.plist File on; Xcode
  merges the two).
- **Code Signing Entitlements**: `TrainedEye.entitlements`.
- **Swift Language Version**: Swift 5.
- **Default Actor Isolation**: `nonisolated`. New Xcode 26 projects default to MainActor,
  but CameraAccess (where `VideoFrameDecoder.swift` comes from) uses nonisolated, and the
  camera, speech and frame classes here run off the main thread. The view models are marked
  `@MainActor` themselves.
- Leave Approachable Concurrency on, as BirdSpotter's app target does (CameraAccess's app
  target does not set it).

## 4. Info.plist and entitlements

`TrainedEye-Info.plist` (beside the project, outside the source folder, as BirdSpotter
keeps its own) holds:

- `CFBundleURLTypes` with the scheme `trainedeye`, and `MWDAT` with `AppLinkURLScheme`
  `trainedeye://`, `MetaAppID` `$(META_APP_ID)`, `ClientToken` `$(CLIENT_TOKEN)` and
  `TeamID` `$(DEVELOPMENT_TEAM)`, copied from CameraAccess. With Developer Mode the two
  Meta values may stay empty.
- `MWDAT` `Analytics` `OptOut` and `CrashReporting` `OptOut`, both true: the SDK collects
  both by default.
- `UIBackgroundModes`: `processing`, `bluetooth-central`, `audio`, `bluetooth-peripheral`
  (CameraAccess) and `external-accessory` with `UISupportedExternalAccessoryProtocols`
  `com.meta.ar.wearable` (the SDK docs and BirdSpotter, for the Bluetooth Classic link).
- `NSBluetoothAlwaysUsageDescription`, `NSBluetoothPeripheralUsageDescription`,
  `NSLocalNetworkUsageDescription` and `NSBonjourServices` `_bonjour._tcp` (CameraAccess),
  plus `NSCameraUsageDescription` (the iPhone camera) and `NSMicrophoneUsageDescription`
  (the spoken request).
- `NSAppTransportSecurity` `NSAllowsLocalNetworking`, for plain `http` and `ws` to the Mac.
- `UIFileSharingEnabled` and `LSSupportsOpeningDocumentsInPlace`, so saved clips show in
  the Files app.
- `UIAppFonts`: Archivo's four static instances in `TrainedEye/Fonts/` (SIL Open Font
  License, `Fonts/Archivo-OFL.txt`), cut by `python -m backend.art.fonts`. The folder is
  added with the rest of `TrainedEye/`; check that the four `.ttf` files show under the
  target's Copy Bundle Resources. Without them the app falls back to the system font.

`TrainedEye.entitlements` has the keychain group both samples declare. CameraAccess also
has Hotspot Configuration and Access Wi-Fi Information; BirdSpotter streams without them,
so they are left out. Add them (Signing & Capabilities) if the Wi-Fi link to the glasses
fails.

## 5. Capabilities

The Info.plist above already lists the background modes; in Signing & Capabilities they
show as Background Modes with Audio, Uses Bluetooth LE accessories, Acts as a Bluetooth LE
accessory, External accessory communication and Background processing. Keychain Sharing
comes from the entitlements file. Camera, microphone, Bluetooth and local network need no
capability, only the usage strings, and iOS asks for each the first time it is used.

## 6. Run it

Connect the iPhone, pick it as the run destination and press Run. On the first launch:

1. Connect: type the backend address (next section) and tap "Check the connection". Allow
   the Local Network prompt.
2. Tap "Connect my glasses": Meta AI opens, approve Trained Eye, and you come back.
3. Put the glasses on, unfolded. The Camera row changes from "Waiting for an active device"
   to "Ready".
4. Watch, Start watching. The first time, the app asks before it opens Meta AI for the
   camera permission (Allow once or Allow always). Watch the capture LED on the glasses: it
   should stay on while the camera streams (not checked yet, see section 4, step 7 of
   `docs/glasses-check.md`), and the app never touches it.

A single tap on the glasses' touchpad pauses the stream; tap again to resume. Folding the
glasses ends the session; while watching, a new one starts once they are unfolded and
worn again. Only one DAT app can use the glasses at a time, so quit CameraAccess or
BirdSpotter first.

## 7. Point the app at `te serve`

On the Mac, `te serve` prints the address, for example
`Trained Eye backend: ws://192.168.1.20:8765/ws`. Type `http://192.168.1.20:8765` (or the
`ws://` address as printed; both work) on the Connect screen. The phone and the Mac must be
on the same Wi-Fi, and macOS must allow Python to accept incoming connections.

To teach from the phone, put a long random `SERVER_TOKEN` in the Mac's `.env` (te serve
says whether it is set) and type the same token on the Connect screen; the app keeps it in
the keychain. Without it the teach routes answer 401, since anyone on the network could
otherwise agree to a spend.

Away from home, use any HTTPS tunnel that forwards WebSockets to `localhost:8765` and type
its `https://` address; the app then uses `wss://`. Installing a tunnel tool needs the
author's yes (HANDOVER section 4).

The app uses `GET /health`, `GET /skills`, the WebSocket at `/ws` (protocol 1, with
`"watch"` in the hello), `POST /teach/start`, `GET /teach/status` and `POST /teach/answer`
(`backend/server.py`, `backend/teach_api.py`).

## 8. Without glasses: the mock device

In Debug builds, the Connect screen has "Use mock glasses". It pairs DAT's mock "Meta
Glasses", powers it on, unfolds and dons it, and sets its camera feed, which the mock needs
before any stream: an HEVC video named `mock_feed.mp4` if the target has one, else the
iPhone's back camera. `Resources/mock_feed.mp4` is a 16 second synthetic clip of a gas
burner, off for 3 seconds and then lit with no pan (HEVC, made by
`python ios/make_mock_feed.py`), so the simulator, which has no camera, has a feed. While the mock is on, Fold, Unfold and "Tap the
touchpad" run MockDeviceKit's `fold()`, `unfold()` and touchpad tap, to check that watching
comes back after a fold and is left alone while paused. To watch for something else,
replace the clip with one of your own, converted with `ffmpeg -i clip.mov -c:v libx265
-tag:v hvc1 mock_feed.mp4`, and do not commit that one: real footage can show people and
rooms.

## Privacy and policy in the app

- The glasses' capture LED is never touched; the camera-on bar shows on every screen.
- Frames are held in memory only: the last 10 for Save clip, the teach capture until it is
  sent. Nothing is written to the phone unless you tap "Save the last 10 seconds".
- Alerts are spoken only when the audio output is Bluetooth, never from the phone's
  speaker, and nothing is spoken while the request is being recorded.
- The Teach screen says that skills cover objects, displays, signs, text and vehicles,
  never a person; the backend refuses the rest and the app shows its one-sentence reason.
- Saved clips may show faces: blur them before publishing anything.

## Files

| Part | Files |
|------|-------|
| 1. Models and settings | `Models/Settings.swift`, `Models/Protocol.swift` |
| 2. Backend and WebSocket clients | `Network/BackendClient.swift`, `Network/WatchSocket.swift` |
| 3. Meta's HEVC decoder (copied) | `Meta/VideoFrameDecoder.swift`, `Meta/HEVCKeyframe.swift`, `Meta/LICENSE` |
| 4. Glasses connection and mock | `Glasses/GlassesConnection.swift`, `Glasses/MockGlasses.swift` |
| 5. Glasses camera | `Glasses/GlassesCamera.swift`, `Glasses/GlassesCamera+Stream.swift` |
| 6. iPhone camera, throttle, speech | `Camera/FrameThrottle.swift`, `Camera/PhoneCamera.swift`, `Audio/Speaker.swift` |
| 7. Teach client and recorder | `Models/TeachModels.swift`, `Network/TeachClient.swift`, `Audio/RequestRecorder.swift` |
| 8. Teach flow | `Teach/TeachFlow.swift` |
| 9. App model | `App/AppModel.swift` |
| 10. Theme and components | `UI/Theme.swift`, `UI/Components.swift`, `Fonts/` (Archivo) |
| 11. App entry and project files | `App/TrainedEyeApp.swift`, `../TrainedEye-Info.plist`, `../TrainedEye.entitlements` |
| 12. Tabs and Connect screen | `UI/RootView.swift`, `UI/ConnectView.swift` |
| 13. Teach screen | `UI/TeachView.swift` |
| 14. Skills and Watch screens | `UI/SkillsView.swift`, `UI/WatchView.swift` |
| 15. Clips | `Clips/ClipStore.swift`, `UI/ClipsView.swift` |
| 16. This README | `../README.md` |

Paths are relative to `ios/TrainedEye/`; `../` is `ios/`.

## Meta's code in this folder

Two files are Meta's sample code from github.com/facebook/meta-wearables-dat-ios 1.0.1,
copied unchanged with Meta's header comment kept:

- `TrainedEye/Meta/VideoFrameDecoder.swift`: `samples/CameraAccess/CameraAccess/Media/VideoFrameDecoder.swift`.
- `TrainedEye/Meta/HEVCKeyframe.swift`: the `isHEVCKeyframe()` extension from
  `samples/CameraAccess/CameraAccess/Media/VideoCaptureHandler.swift`, which the decoder calls.

They are under Meta's Wearables Developer Terms, not an open-source license, so Meta's
`LICENSE` sits next to them in `TrainedEye/Meta/`. CONTRIBUTIONS.md must list them, and the
terms must be checked before this folder is published. The rest of the code follows the
samples' calls and structure (named in each file's header) but was written for this app.

## TODOs and unverified points

No TODO is left in the code, but these behaviours were written without a compiler or the
glasses, and each needs one check on the glasses (and in the simulator where it says so):

- **A fold mid-watch** (`Glasses/GlassesCamera.swift`, `Glasses/GlassesCamera+Stream.swift`,
  `App/AppModel.swift`): fold the
  glasses while watching. The bar and the Watch screen should say they are folded or out of
  reach, and no frames go out while the socket stays connected. Unfold them and put them on:
  a new session should start by itself, frames come back, and the Glasses card's "Came back
  by itself" counts one. Then tap the touchpad: nothing may restart while the stream is
  paused. Walk out of Bluetooth range and back too. In the simulator, do the same with the
  mock glasses' Fold, Unfold and "Tap the touchpad" buttons.
- **The screen locked** (`App/AppModel.swift`, `Glasses/GlassesCamera+Stream.swift`): watch with the
  glasses, lock the phone for two minutes, unlock it. The Watch screen's "Last time in the
  background" card should show about one frame a second from the camera, sent and acked.
  If frames came from the camera but none went out, the socket stopped in the background; if
  many repeated frames were skipped, the decoder failed there (then render on the CPU,
  `VTCreateCGImageFromCVPixelBuffer` or a software `CIContext`, in a wrapper around
  `Meta/VideoFrameDecoder.swift`, which is Meta's copied file); if nothing came from the
  camera, the DAT stream stopped. Check that an alert is spoken with the phone locked; if
  not, give `Audio/Speaker.swift` a `.playback`, `.spokenAudio` session with `.duckOthers`.
  Record the numbers in PROGRESS.md.
- **The request microphone** (`Audio/RequestRecorder.swift`): hold to talk with the glasses
  on. The Teach screen should say "glasses mic", and the request should be the wearer's
  voice, not silence. Fold the glasses part way through a request: it should go on with the
  iPhone's microphone and say "glasses, then iPhone mic". Without the glasses it says
  "iPhone mic". The session mixes with other audio, as CameraAccess's does, so also check a
  request made while music plays on the glasses.
- **The keepalive** (`Network/WatchSocket.swift`): watch through an HTTPS tunnel, pause the
  stream with a tap and wait two minutes: the Connection card should stay Connected. Then
  stop `te serve` with the socket open, or turn the Mac's Wi-Fi off: within about 30 seconds
  the card should say no pong came and try again.
- **A teach session after a relaunch** (`Teach/TeachFlow.swift`, `Network/TeachClient.swift`):
  send a capture and, while the Mac asks "Spend it?", close the app from the app switcher
  and open it again. The Teach screen should show the same question (the server token must
  be set), and answering it should carry on. After a fresh `te serve` it starts empty.
- **Medium resolution for the teach capture** (`Glasses/GlassesCamera+Stream.swift`,
  `App/AppModel.swift`): during a capture with the glasses, the Teach screen's frame size
  should read 504x896 (less if the SDK lowers it for bandwidth), and the Watch screen's
  360x640 when watching again afterwards. Compare a skill taught at medium with one at low
  before keeping medium: the DAT repository's AGENTS.md says lower settings give better
  quality per frame over Bluetooth.
- **Skills status** (`Models/Protocol.swift`, `UI/SkillsView.swift`): while `te train` runs,
  the Skills screen should say "Training on Modal" with the run's name; after it, "Trained";
  after `te serve-student` and `te calibrate`, "Ready, threshold" and the number. A backend
  without the `training` and `served` fields shows the earlier lines.

Unverified until it runs on a phone with the glasses (it builds for the simulator with Xcode
27.2, Nov 4, so the code compiles against DAT 1.0.1):

- Every DAT call is copied from the samples. They compile against 1.0.1 now; how they behave
  with real glasses is known only from how the samples use them.
- `NSAllowsLocalNetworking` letting plain `http` and `ws` reach the Mac's IP address. If it
  does not, a Debug-only ATS exception for that address is the fallback.
- That `URLSessionWebSocketTask` and the spoken alerts keep working in the background while
  the glasses stream (the screen-locked check above).
- The two calls no sample makes: the `AVAudioEngineConfigurationChange` observer in
  `Audio/RequestRecorder.swift` and `URLSessionWebSocketTask.sendPing` in
  `Network/WatchSocket.swift`.
- How the capture LED behaves (section 4, step 7 of `docs/glasses-check.md` records it).
- That Meta Adventurer works as DAT's "Meta Glasses" model; the mock pairs `.metaGlasses`.
- That Info.plist merging (the file plus Xcode's generated keys) gives no duplicate keys.
