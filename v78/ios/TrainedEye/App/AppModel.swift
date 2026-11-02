//
// AppModel.swift
//
// Ties the parts together: which camera runs (the glasses, the DAT mock in Debug builds,
// or the iPhone), where its frames go (the WebSocket while watching, the teach flow while
// capturing), what an alert does (speak it on the glasses, show it), and saving a clip.
// The last 10 frames are held in memory only, for Save clip; nothing is written to disk
// unless the user taps it.
//

import Foundation
import MWDATCore
import Observation

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
  /// The newest frame (JPEG, in memory only), for the preview on the Watch and Teach screens.
  private(set) var lastFrame: Data?
  private(set) var skills: [SkillInfo] = []
  private(set) var skillsError: String?
  /// A one-line notice for the user.
  var message: String?

  static let clipFrames = 10

  @ObservationIgnored private let throttle: FrameThrottle
  @ObservationIgnored private var recentFrames: [Data] = []

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
  }

  var backend: BackendClient? {
    settings.baseURL.map { BackendClient(base: $0, token: settings.serverToken) }
  }

  /// True while any camera is on: the watch-mode indicator shows on every screen.
  var isCameraOn: Bool { source != nil }

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
    guard let source = await startCamera() else { return }
    throttle.setInterval(1)
    isWatching = true
    message = nil
    socket.connect(to: url, hello: Hello(source: source.rawValue, watch: watchList()))
  }

  func stopWatching() {
    isWatching = false
    socket.disconnect()
    recentFrames.removeAll()
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

  /// Starts a camera for the teach capture, switches to three frames a second, and waits
  /// (up to 20 seconds, not counting the camera permission step) for the first frame.
  func startCameraForTeach() async -> Bool {
    guard !isWatching else {
      message = "Stop watching before teaching."
      return false
    }
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

  func endTeachCapture() {
    throttle.setInterval(1)
    if !isWatching { stopCamera() }
  }

  // MARK: - Cameras

  // TODO: no sample shows starting a new session after the glasses fold or the link
  // drops mid-watch (AGENTS.md: start a new session once the device is available again).
  // Today the Watch screen shows the state, and the user stops and starts watching.

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
  }

  private func receive(_ frame: CapturedFrame) {
    guard source != nil else { return }
    lastFrame = frame.jpeg
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
  }

  private func handleAlert(_ alert: ServerMessage) {
    let text = alert.text ?? alert.skill ?? "Alert"
    let spoken = speaker.speak(text)
    lastAlert = AlertEvent(
      skill: alert.skill ?? "", text: text, probability: alert.p, confirmed: alert.confirmed,
      receivedAt: Date(), spoken: spoken)
  }

  // MARK: - Skills

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
