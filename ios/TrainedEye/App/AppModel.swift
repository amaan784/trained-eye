//
// AppModel.swift
//
// Ties the parts together: which camera runs (the glasses, the DAT mock in Debug builds,
// or the iPhone), where its frames go (the WebSocket while watching, the teach flow while
// capturing), what an alert does (speak it on the glasses, show it), and saving a clip.
// The last 10 frames are held in memory only, for Save clip; nothing is written to disk
// unless the user taps it. The glasses stream at low resolution (360x640) while watching
// and at medium (504x896) during the teach capture; the iPhone camera keeps its own size.
// While watching, each stay in the background (the screen locked, or another app in front)
// is counted, as CameraAccess observes didEnterBackgroundNotification, so the screen-locked
// check on the glasses can read what went out and what came back acked. At launch, a teach
// session still running on the Mac is picked up again.
//

import Foundation
import MWDATCore
import Observation
import UIKit

enum FrameSourceKind: String {
  case glasses
  case mock
  case phone

  var label: String {
    switch self {
    case .glasses: return "glasses"
    case .mock: return "mock glasses"
    case .phone: return "iPhone camera"
    }
  }
}

/// One stay in the background while watching, for the screen-locked check on the glasses:
/// did frames keep coming from the camera, going out and coming back acked?
struct BackgroundStay: Equatable {
  var seconds: TimeInterval
  /// Frames the camera delivered (one a second while watching).
  var framesFromCamera: Int
  var framesSent: Int
  var framesAcked: Int
  /// Images the glasses' decoder handed back unchanged, skipped rather than sent again.
  var repeatsSkipped: Int
}

@MainActor
@Observable
final class AppModel {
  let settings = AppSettings()
  let glasses: GlassesConnection
  let glassesCamera: GlassesCamera
  let phoneCamera: PhoneCamera
  let socket = WatchSocket()
  let teach = TeachFlow()
  let clips = ClipStore()
  @ObservationIgnored let speaker = Speaker()
  #if DEBUG
  let mock = MockGlassesController()
  #endif

  private(set) var source: FrameSourceKind?
  private(set) var isWatching = false
  private(set) var lastAlert: AlertEvent?
  /// The last five frames the backend scored, each with its highest score, for the
  /// Watch screen's film strip (in memory only, like every frame).
  private(set) var scoredFrames: [ScoredFrame] = []
  /// The newest frame (JPEG, in memory only), for the preview on the Watch and Teach screens.
  private(set) var lastFrame: Data?
  /// The newest frame's size in pixels, as "504x896".
  private(set) var lastFrameSize: String?
  private(set) var skills: [SkillInfo] = []
  private(set) var skillsError: String?
  /// A one-line notice for the user.
  var message: String?
  /// The latest stay in the background since watching started, and how many there were.
  private(set) var lastBackgroundStay: BackgroundStay?
  private(set) var backgroundStays = 0

  static let clipFrames = 10

  @ObservationIgnored private let throttle: FrameThrottle
  @ObservationIgnored private var recentFrames: [Data] = []
  /// Frames sent and not yet acked, oldest first: each ack belongs to the oldest.
  @ObservationIgnored private var unacked: [Data] = []
  @ObservationIgnored private var watchStarted = Date()
  /// Frames that reached the app from any camera, for the background counts.
  @ObservationIgnored private var framesReceived = 0
  /// The counters when the app went to the background while watching.
  @ObservationIgnored private var backgroundStart:
    (date: Date, received: Int, sent: Int, acked: Int, skipped: Int)?
  @ObservationIgnored private var lifecycleTasks: [Task<Void, Never>] = []

  init() {
    let wearables = Wearables.shared
    let throttle = FrameThrottle()
    self.throttle = throttle
    glasses = GlassesConnection(wearables: wearables)
    glassesCamera = GlassesCamera(wearables: wearables, throttle: throttle)
    phoneCamera = PhoneCamera(throttle: throttle)
    teach.app = self
    throttle.setSink { [weak self] frame in
      Task { @MainActor in self?.receive(frame) }
    }
    socket.onAlert = { [weak self] message in
      self?.handleAlert(message)
    }
    socket.onScores = { [weak self] scores in
      self?.scored(scores)
    }
    startLifecycleMonitoring()
    // Once at launch: a session waiting on its spend or read-back question would otherwise
    // sit until the server's answer timeout counts it as a no.
    Task { [weak self] in await self?.teach.resumeIfRunning() }
  }

  var backend: BackendClient? {
    settings.baseURL.map { BackendClient(base: $0, token: settings.serverToken) }
  }

  /// True while any camera is on: the watch-mode indicator shows on every screen.
  var isCameraOn: Bool { source != nil }

  /// The glasses' camera is on but sends nothing: folded, out of reach, or the link
  /// dropped. It comes back by itself when the glasses are back.
  var isGlassesInterrupted: Bool {
    switch source {
    case .glasses, .mock: return glassesCamera.isInterrupted
    case .phone, nil: return false
    }
  }

  var isMockActive: Bool {
    #if DEBUG
    return mock.isActive
    #else
    return false
    #endif
  }

  // MARK: - Watching

  func startWatching() async {
    guard !isWatching else { return }
    guard let url = settings.webSocketURL else {
      message = BackendError.noAddress.localizedDescription
      return
    }
    speaker.prepare()  // the audio session is set before the stream, not at the first alert
    guard let source = await startCamera() else { return }
    throttle.setInterval(1)
    isWatching = true
    message = nil
    scoredFrames = []
    unacked = []
    watchStarted = Date()
    lastBackgroundStay = nil
    backgroundStays = 0
    socket.connect(to: url, hello: Hello(source: source.rawValue, watch: watchList()))
  }

  func stopWatching() {
    isWatching = false
    backgroundStart = nil
    socket.disconnect()
    recentFrames.removeAll()
    unacked.removeAll()
    if !teach.isCapturing { stopCamera() }
  }

  /// The Skills screen changed the switches: say hello again with the new list.
  func watchedSkillsChanged() {
    guard isWatching, let url = settings.webSocketURL, let source else { return }
    socket.connect(to: url, hello: Hello(source: source.rawValue, watch: watchList()))
  }

  /// The switched-on skills the backend still lists, or nil for none (frames only acked).
  private func watchList() -> [String]? {
    let known = Set(skills.map(\.name))
    let names = settings.watchedSkills.filter { known.isEmpty || known.contains($0) }.sorted()
    return names.isEmpty ? nil : names
  }

  func saveClip() {
    guard !recentFrames.isEmpty else {
      message = "No frames to save yet."
      return
    }
    let info = ClipInfo(
      savedAt: Date(), source: source?.rawValue ?? "unknown", fps: 1,
      skills: socket.watching, lastAlert: lastAlert?.text)
    do {
      try clips.save(frames: recentFrames, info: info)
      message = "Saved the last \(recentFrames.count) seconds to Clips."
    } catch {
      message = error.localizedDescription
    }
  }

  // MARK: - Teach capture

  /// Starts a camera for the teach capture, the glasses at medium resolution (504x896),
  /// switches to three frames a second, and waits (up to 20 seconds, not counting the
  /// camera permission step) for the first frame.
  func startCameraForTeach() async -> Bool {
    guard !isWatching else {
      message = "Stop watching before teaching."
      return false
    }
    // Before the start, so the first camera is added at that size; a camera already on is
    // stopped and added again at it.
    glassesCamera.setResolution(.medium)
    let started = await startTeachCamera()
    if !started { glassesCamera.setResolution(.low) }  // the camera is off by now
    return started
  }

  private func startTeachCamera() async -> Bool {
    lastFrame = nil
    guard await startCamera() != nil else { return false }
    throttle.setInterval(TeachFlow.captureInterval)
    var waited = 0
    // "Not now" at the permission prompt stops the camera (source becomes nil).
    while lastFrame == nil, waited < 80, source != nil, !glassesCamera.permissionRefused {
      try? await Task.sleep(for: .milliseconds(250))
      if !glassesCamera.permissionPending { waited += 1 }
    }
    guard lastFrame != nil else {
      if source == nil {
        message = "The capture was cancelled: the camera is off."
      } else if glassesCamera.permissionRefused {
        message = glassesCamera.errorMessage ?? "Camera permission denied"
        stopCamera()
      } else {
        message = "No frames from the \(source?.label ?? "camera") yet."
        stopCamera()
      }
      return false
    }
    return true
  }

  /// Back to one frame a second and, for watching, the low resolution.
  func endTeachCapture() {
    throttle.setInterval(1)
    if !isWatching { stopCamera() }
    glassesCamera.setResolution(.low)  // after a stop it only sets the next camera's size
  }

  // MARK: - Cameras

  /// Starts the camera the setting asks for: the glasses when they are available (or
  /// the setting says glasses), else the iPhone.
  private func startCamera() async -> FrameSourceKind? {
    if let source { return source }
    let useGlasses: Bool
    switch settings.cameraChoice {
    case .glasses: useGlasses = true
    case .phone: useGlasses = false
    case .auto: useGlasses = glassesCamera.hasActiveDevice
    }
    if useGlasses {
      // start() reports a refused start or a session it could not create (no eligible
      // device) at once; isActive alone is also true for a session that is still ending.
      guard glassesCamera.start(), glassesCamera.isActive else {
        message = glassesCamera.errorMessage ?? "The glasses are not available."
        return nil
      }
      source = isMockActive ? .mock : .glasses
    } else {
      guard await phoneCamera.start() else {
        message = phoneCamera.errorMessage
        return nil
      }
      source = .phone
    }
    return source
  }

  private func stopCamera() {
    switch source {
    case .glasses, .mock: glassesCamera.stop()
    case .phone: phoneCamera.stop()
    case nil: break
    }
    source = nil
    lastFrame = nil
    lastFrameSize = nil
  }

  private func receive(_ frame: CapturedFrame) {
    guard source != nil else { return }
    framesReceived += 1
    lastFrame = frame.jpeg
    lastFrameSize = "\(frame.width)x\(frame.height)"
    if teach.isCapturing {
      teach.add(frame)
      return
    }
    guard isWatching else { return }
    recentFrames.append(frame.jpeg)
    if recentFrames.count > Self.clipFrames {
      recentFrames.removeFirst(recentFrames.count - Self.clipFrames)
    }
    socket.send(jpeg: frame.jpeg)
    unacked.append(frame.jpeg)
    if unacked.count > 10 { unacked.removeFirst(unacked.count - 10) }
  }

  /// An ack came back: its frame is the oldest one not yet acked.
  private func scored(_ scores: [String: Double?]) {
    guard !unacked.isEmpty else { return }
    let jpeg = unacked.removeFirst()
    let best = scores.values.compactMap { $0 }.max()
    scoredFrames.append(
      ScoredFrame(id: (scoredFrames.last?.id ?? 0) + 1, jpeg: jpeg,
                  seconds: Date().timeIntervalSince(watchStarted), score: best))
    if scoredFrames.count > 5 { scoredFrames.removeFirst(scoredFrames.count - 5) }
  }

  private func handleAlert(_ alert: ServerMessage) {
    let text = alert.text ?? alert.skill ?? "Alert"
    let spoken = speaker.speak(text)
    lastAlert = AlertEvent(
      skill: alert.skill ?? "", text: text, probability: alert.p, confirmed: alert.confirmed,
      receivedAt: Date(), spoken: spoken)
  }

  // MARK: - In the background

  private func startLifecycleMonitoring() {
    lifecycleTasks = [
      Task { [weak self] in
        for await _ in NotificationCenter.default.notifications(
          named: UIApplication.didEnterBackgroundNotification)
        {
          self?.didEnterBackground()
        }
      },
      Task { [weak self] in
        for await _ in NotificationCenter.default.notifications(
          named: UIApplication.willEnterForegroundNotification)
        {
          self?.willEnterForeground()
        }
      },
    ]
  }

  private func didEnterBackground() {
    guard isWatching else { return }
    backgroundStart = (
      date: Date(), received: framesReceived, sent: socket.framesSent,
      acked: socket.framesAcked, skipped: glassesCamera.repeatedFramesSkipped
    )
  }

  /// The counts since the app went to the background. The socket's counters start again
  /// on a new hello, so a count is never taken below zero.
  private func willEnterForeground() {
    guard let start = backgroundStart, isWatching else { return }
    backgroundStart = nil
    lastBackgroundStay = BackgroundStay(
      seconds: Date().timeIntervalSince(start.date),
      framesFromCamera: framesReceived - start.received,
      framesSent: max(0, socket.framesSent - start.sent),
      framesAcked: max(0, socket.framesAcked - start.acked),
      repeatsSkipped: max(0, glassesCamera.repeatedFramesSkipped - start.skipped))
    backgroundStays += 1
  }

  // MARK: - Skills

  /// The model behind watch mode, as the backend last said, and when that was.
  private(set) var modelStatus: ModelStatus?
  private(set) var modelStatusAt = Date()
  @ObservationIgnored private var modelPoll: Task<Void, Never>?

  /// Wakes the model on Modal ahead of watching (the Skills screen calls this), then asks
  /// how it is going every 2 s until it is ready or has failed.
  func wakeModel() {
    guard let backend, modelPoll == nil else { return }
    modelPoll = Task { [weak self] in
      var status = try? await backend.wakeModel()
      while !Task.isCancelled {
        await MainActor.run {
          self?.modelStatus = status
          self?.modelStatusAt = Date()
        }
        guard let current = status, current.state == "waking" else { break }
        try? await Task.sleep(for: .seconds(2))
        status = try? await backend.model()
      }
      await MainActor.run { self?.modelPoll = nil }
    }
  }

  func refreshSkills() async {
    guard let backend else {
      skillsError = BackendError.noAddress.localizedDescription
      return
    }
    do {
      skills = try await backend.skills()
      skillsError = nil
    } catch {
      skillsError = error.localizedDescription
    }
  }
}

/// A frame the backend scored: its time since watching started and its highest score.
struct ScoredFrame: Identifiable, Equatable {
  let id: Int
  let jpeg: Data
  let seconds: TimeInterval
  let score: Double?
}
