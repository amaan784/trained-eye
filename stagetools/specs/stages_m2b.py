"""Per-stage metadata for stages v50 to v74: the rest of M2 (teaching from the phone,
export and serving, the iOS app)."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("stages_late", Path(__file__).with_name("stages_late.py"))
_late = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_late)

INTRO, POLICY, PYTEST, RUFF = _late.INTRO, _late.POLICY, _late.PYTEST, _late.RUFF
COMPONENTS, VENV, DATE, stage = _late.COMPONENTS, _late.VENV, _late.DATE, _late.stage

IOS = ("The app is SwiftUI for iOS 17.2 and has not been compiled: there is no Xcode here. "
       "A file may use types from a later iOS stage (the app model and the teach flow need "
       "each other), so the app builds only as a whole, from v73 on. ")

STAGES = {
    50: stage(
        "a teach session for the phone: the spend question before any paid call, then the "
        "read-back.",
        "Phone teach session",
        "backend/teach_session.py runs one teach for the iPhone app in a background thread. "
        "It asks the spend question first and calls nothing paid until the phone says yes, "
        "then runs the teach flow, whose read-back the phone answers too (a correction "
        "recompiles). It keeps the progress lines, times out an unanswered question after "
        "10 minutes as a no and sums up the dataset at the end. When the server stops, it "
        "refuses the next paid call, cuts a retry's backoff short and waits for the call in "
        "flight, so the ledger records it; the Meta client now checks the caps before every "
        "retry, not only after one that may have been billed. Its tests come two stages "
        "later.",
        "Add the phone teach session",
        "Adds TeachSession: the spend and read-back questions, the background run, timeouts, "
        "corrections, a stop for server shutdown and a summary."),
    51: stage(
        "the phone's teach upload: the capture, its checks and the estimate.",
        "Phone teach upload",
        "backend/teach_upload.py takes the phone's multipart upload (a hand-written parser, "
        "since python-multipart is not a dependency), checks the skill name, the request, "
        "the audio, the counts and every frame before anything is spent (a frame longer than "
        "4:1 is refused: scaled for keyframes it would take gigabytes), saves the capture under "
        "data/teach/<skill>/<time>/ so it can be repeated from the Mac, and builds the "
        "estimate and an unstarted session. A run whose worst case is over $15 is refused: "
        "it has to be started from the Mac.",
        "Add the phone teach upload and its checks",
        "Adds the multipart parser, the input checks, the saved capture and the estimate."),
    52: stage(
        "the tests of the teach session and the upload.",
        "Teach session and upload tests",
        "Tests for the session (both questions and their timeouts, answers that do not fit, "
        "a refused request, a failed run, a run that fails before the spend question, and a "
        "stop that answers no, refuses the next call or cuts a rate-limit backoff short) "
        "and for the parser and the upload's refusals, against the fake Meta API. The full "
        "run, a correction and the $15 and frame-shape limits are tested over HTTP with the "
        "routes.",
        "Test the phone teach session and upload",
        "Adds tests/test_teach_session.py and tests/test_teach_upload.py."),
    53: stage(
        "the teach routes the phone calls: start, status and answer, behind a token.",
        "Phone teach routes",
        "backend/teach_api.py adds POST /teach/start, GET /teach/status and POST "
        "/teach/answer. Every route needs the SERVER_TOKEN from .env as a Bearer header: "
        "these routes spend money and te serve listens on the whole network. One session "
        "runs at a time; the status carries the state, the progress lines, the estimate, the "
        "read-back and the result. Both bodies have size limits, and stop() ends every "
        "running session for a shutdown. Its tests come two stages later.",
        "Add the teach routes for the phone",
        "Adds the three routes, the token check, the body limits and stop()."),
    54: stage(
        "the server mounts the teach routes and lists taught skills; the student gets its "
        "key and Meta's sampling.",
        "Server, keys and sampling",
        "The server now mounts the teach routes, stops them when it shuts down, and answers "
        "GET /skills for the app's Skills screen. Two new secrets: STUDENT_API_KEY guards the "
        "trained model's endpoint, which watch mode sends, and SERVER_TOKEN guards the teach "
        "routes. The scorer stops sending temperature 0: Meta's serving notes advise against "
        "greedy decoding for Glimmer, so the server's published sampling applies, with "
        "reasoning_strength low. The training bundle can read the unfiltered dataset folder "
        "for Experiment 2.",
        "Mount the teach routes; add the student key, the server token and sampling",
        "Mounts the teach API with a shutdown hook, adds GET /skills, the STUDENT_API_KEY and "
        "SERVER_TOKEN secrets, the cookbook's sampling for the scorer and a dataset_dir "
        "option for the bundle, with tests."),
    55: stage(
        "the teach routes' tests over HTTP.",
        "Phone teach routes tests",
        "Tests over HTTP with the fake Meta API: a full run, a no to the spend, a corrected "
        "read-back, a second start while one runs, the $15 limit, bad inputs, the token "
        "(missing, wrong, unset or unusable), odd answers, frames that are not camera frames "
        "and an endless audio length. None of them sends or bills anything.",
        "Test the teach routes",
        "Adds tests/test_teach_api.py."),
    56: stage(
        "the export: merge the trained adapter on Modal and check the merged model.",
        "Export on Modal",
        "backend/training/export.py merges a run's QLoRA adapter into the base model on one "
        "H100 and saves it to the volume, then reloads it and checks that it gives the same "
        "yes/no answers as base plus adapter on up to 50 examples. Loading and prompting "
        "follow the cookbook's evaluate.py. backend/training/export_check.py holds the plain "
        "helpers (the fixed example order, reading the answer, the comparison).",
        "Add the export of a trained adapter",
        "Adds the Modal export app and its plain helpers. Its tests come with the next stage."),
    57: stage(
        "serving the merged model with vLLM on Modal, and the export's tests.",
        "Student endpoint on Modal",
        "backend/inference/modal_serve.py serves the merged model with the vllm-openai "
        "muse-glimmer image on one H100, with the flags from the cookbook's vllm.md, logprobs "
        "for the scorer, and an API key from the Modal Secret named student, passed to vLLM "
        "in its environment rather than on the command line. The tests read both Modal apps "
        "(modal is not installed here) and test the export helpers.",
        "Add the student endpoint and the export tests",
        "Adds the vLLM serving app and static tests for it and for the export."),
    58: stage(
        "te export and te serve-student.",
        "te export and te serve-student",
        "te export runs the merge on Modal, asking first (at most 3 hours, not timed yet), "
        "and reports the agreement and the file checks. te serve-student deploys or stops "
        "the endpoint, asking first, and says how to set TE_STUDENT_URL. te watch sends the "
        "student key, te train takes --dataset, te serve says whether phone teaching is on "
        "(and why not), and te teach now asks above $15 even with --yes.",
        "Add te export and te serve-student",
        "Adds both commands with their questions and messages, the student key in te watch, "
        "--dataset in te train, the phone teaching line in te serve and the $15 question in "
        "te teach, with tests."),
    59: stage(
        "the iPhone app starts: its settings and the server's messages.",
        "iOS: settings and protocol",
        IOS + "Settings (the backend URL, stored on the phone, and the server token, in the "
        "keychain) and the WebSocket messages and the /health and /skills replies as Swift "
        "types.",
        "Add the iOS app's settings and protocol types",
        "Adds Settings.swift and Protocol.swift."),
    60: stage(
        "the app's clients for the backend and the WebSocket.",
        "iOS: backend and WebSocket clients",
        IOS + "BackendClient fetches /skills and sends the server token; WatchSocket speaks "
        "the watch protocol (hello, a frame a second with one in flight at a time, acks with "
        "scores, alerts) and reconnects, also when the backend stops answering frames.",
        "Add the iOS backend and WebSocket clients",
        "Adds BackendClient.swift and WatchSocket.swift."),
    61: stage(
        "Meta's HEVC frame decoder, copied from the CameraAccess sample.",
        "iOS: Meta's HEVC decoder",
        IOS + "VideoFrameDecoder.swift and the keyframe check are copied unchanged from "
        "Meta's CameraAccess sample, with Meta's header and the SDK's LICENSE next to them "
        "(Meta's Wearables Developer Terms).",
        "Add Meta's HEVC decoder from the DAT sample",
        "Adds the two copied files and Meta's LICENSE under ios/TrainedEye/Meta/."),
    62: stage(
        "the glasses connection and the mock device.",
        "iOS: glasses connection and mock",
        IOS + "GlassesConnection registers with the Meta AI app and follows the device, link "
        "and battery states as the DAT samples do; MockGlasses drives DAT's mock device in "
        "debug builds, choosing its feed before the pair is worn.",
        "Add the iOS glasses connection and mock device",
        "Adds GlassesConnection.swift and MockGlasses.swift."),
    63: stage(
        "the glasses camera: HEVC at 7 fps, one JPEG a second to the backend.",
        "iOS: glasses camera",
        IOS + "GlassesCamera starts a DAT camera session (hvc1, low resolution, 7 fps, as "
        "BirdSpotter does), decodes every frame and hands one a second on. It never touches "
        "the capture LED.",
        "Add the iOS glasses camera",
        "Adds GlassesCamera.swift."),
    64: stage(
        "the iPhone camera fallback, the frame throttle and spoken alerts.",
        "iOS: phone camera and speech",
        IOS + "PhoneCamera is the fallback when the glasses are not there; FrameThrottle keeps "
        "one frame a second (three during a teach capture, on an even schedule); Speaker "
        "says alerts only when the audio goes to Bluetooth.",
        "Add the iOS phone camera, throttle and speech",
        "Adds FrameThrottle.swift, PhoneCamera.swift and Speaker.swift."),
    65: stage(
        "the teach client and the request recorder.",
        "iOS: teach client and recorder",
        IOS + "TeachClient talks to /teach/start, /teach/status and /teach/answer; "
        "RequestRecorder records the spoken request.",
        "Add the iOS teach client and recorder",
        "Adds TeachModels.swift, TeachClient.swift and RequestRecorder.swift."),
    66: stage(
        "the teach flow on the phone.",
        "iOS: teach flow",
        IOS + "TeachFlow runs hold to talk, the 10 second capture with a countdown, the "
        "upload, the spend question, the read-back and the progress.",
        "Add the iOS teach flow",
        "Adds TeachFlow.swift."),
    67: stage(
        "the app model that ties the pieces together.",
        "iOS: app model",
        IOS + "AppModel holds the connection, the camera source, the watch socket, the "
        "skills and the last alert; watch and teach never run at once.",
        "Add the iOS app model",
        "Adds AppModel.swift."),
    68: stage(
        "the app's theme and shared components.",
        "iOS: theme and components",
        IOS + "A dark, minimal theme with one accent colour and the shared views, including "
        "the camera-on bar shown on every screen.",
        "Add the iOS theme and components",
        "Adds Theme.swift and Components.swift."),
    69: stage(
        "the app entry, the Info.plist and the entitlements.",
        "iOS: app entry and project files",
        IOS + "The app entry configures the SDK; the Info.plist has CameraAccess's keys with "
        "the analytics and crash reporting turned off.",
        "Add the iOS app entry and project files",
        "Adds TrainedEyeApp.swift, the Info.plist and the entitlements."),
    70: stage(
        "the tabs and the Connect screen.",
        "iOS: tabs and Connect",
        IOS + "The tab layout and the Connect screen (glasses, backend URL, status).",
        "Add the iOS tabs and Connect screen",
        "Adds RootView.swift and ConnectView.swift."),
    71: stage(
        "the Teach screen.",
        "iOS: Teach screen",
        IOS + "The Teach screen for the teach flow.",
        "Add the iOS Teach screen",
        "Adds TeachView.swift."),
    72: stage(
        "the Skills and Watch screens.",
        "iOS: Skills and Watch",
        IOS + "Skills lists the taught skills with on and off toggles; Watch shows that the "
        "camera is on, the last alert and the battery.",
        "Add the iOS Skills and Watch screens",
        "Adds SkillsView.swift and WatchView.swift."),
    73: stage(
        "clips the user saves explicitly.",
        "iOS: Clips",
        IOS + "Frames are held in memory only; Save clip keeps the last ten as a folder that "
        "te watch can replay. Nothing is stored without that tap.",
        "Add the iOS clips",
        "Adds ClipStore.swift and ClipsView.swift."),
    74: stage(
        "how to build the app in Xcode.",
        "iOS: README",
        "ios/README.md says how to create the Xcode project, add the DAT package at 1.0.1, "
        "the Info.plist keys and capabilities, Developer Mode, and how to point the app at "
        "te serve. It lists the TODOs and says plainly that the code has not been compiled.",
        "Add the iOS build instructions",
        "Adds ios/README.md."),
}

NEXT_M0_DONE = ("Milestone M3: the labelling tool, the evaluation harness and the code for "
                "the four experiments.")
