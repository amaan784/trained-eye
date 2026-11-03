//
// GlassesCamera.swift
//
// The glasses' camera through Meta's DAT SDK, with the lifecycle of CameraViewModel.swift
// in Meta's CameraAccess sample (session, started, camera permission, Camera, stream) and
// BirdSpotter's stream settings: HEVC (hvc1), low resolution (360x640), 7 fps, which DAT
// keeps streaming in the background. Unlike CameraAccess, the session is not ended when
// the app goes to the background. Every frame is decoded (each HEVC frame builds on the
// ones before) and the throttle keeps one a second. Nothing here touches the capture LED.
//
// While the camera is wanted (from start() until stop()) it comes back by itself:
// - After a fold or a link drop the session stops. AGENTS.md: "Opening hinges restores
//   Bluetooth but does not restart sessions. Start a new session after the device becomes
//   available again." So a new session starts when the device selector reports the glasses
//   again; never while a session is paused, and at most 3 times without a frame between.
// - A stream that stops on its own while the session stays up is added again, up to 3
//   times 750 ms apart, as bringCameraBack in BirdSpotter's DatGlassesSessionRepository.
// The teach capture streams at medium (504x896). A Camera's configuration cannot change in
// place, so the camera is stopped, which detaches it from the session, and added again
// with a new StreamConfiguration once its stream reports stopped (AGENTS.md, CameraAccess's
// stopStreaming). The session stays up.
//
// The decoder hands back its last good image when a frame fails or while it waits for a
// keyframe (Meta/VideoFrameDecoder.swift), so an image already delivered is skipped here
// rather than sent again as a new frame. Whether the stream and the WebSocket keep going
// with the screen locked is still to be checked on the glasses: the Watch screen counts
// what went out and came back acked while the app was in the background.
//

import MWDATCamera
import MWDATCore
import Observation
import UIKit

@MainActor
@Observable
final class GlassesCamera {
  /// The stream sizes the app asks for (AGENTS.md: low 360x640, medium 504x896). The SDK
  /// lowers the size on its own when Bluetooth bandwidth is short.
  enum Resolution: Equatable {
    case low
    case medium

    var streaming: StreamingResolution {
      switch self {
      case .low: return StreamingResolution.low
      case .medium: return StreamingResolution.medium
      }
    }

    var label: String {
      switch self {
      case .low: return "360x640"
      case .medium: return "504x896"
      }
    }
  }

  /// BirdSpotter's sensorRevivalAttempts and sensorRevivalDelay: how often, and how far
  /// apart, a stream that stopped on its own is added again.
  static let streamRevivalAttempts = 3
  static let streamRevivalDelay = Duration.milliseconds(750)
  /// New sessions started after drops, without a frame arriving in between, before the
  /// app stops trying and asks the user to start watching again.
  static let sessionRestartAttempts = 3

  /// An eligible pair of glasses is available, from the auto device selector.
  private(set) var hasActiveDevice = false
  private(set) var sessionState: DeviceSessionState = .idle
  private(set) var streamState: StreamState = .stopped
  /// Camera permission is not granted yet: the UI asks before Meta AI opens.
  var needsPermissionConfirm = false
  /// The Meta AI app is open for the camera permission.
  private(set) var isAskingPermission = false
  /// The camera permission was refused (or the request failed): no frames will come.
  private(set) var permissionRefused = false
  /// Waiting on the user for the camera permission, so no frames can come yet.
  var permissionPending: Bool { needsPermissionConfirm || isAskingPermission }
  private(set) var needsGlassesAppUpdate = false
  var errorMessage: String?
  /// The session or the stream ended without stop() while the camera is still wanted: the
  /// glasses were folded or went out of reach, the link dropped, or another app took them.
  /// Cleared once frames flow again, or by stop().
  private(set) var isInterrupted = false
  /// How often the camera came back by itself after an interruption, since start().
  private(set) var recoveries = 0
  /// The size the next camera is added at.
  private(set) var resolution: Resolution = .low

  var isStreaming: Bool { streamState == .streaming }
  var isPaused: Bool { streamState == .paused }

  /// A session or a stream exists, so the camera is on or coming on.
  var isActive: Bool {
    switch sessionState {
    case .starting, .started, .paused, .stopping: return true
    default: return streamState != .stopped
    }
  }

  var statusText: String {
    switch streamState {
    case .streaming: return "Streaming"
    case .paused: return "Paused: tap the glasses to resume"
    case .waitingForDevice: return "Waiting for the glasses"
    case .starting: return "Starting the camera"
    case .stopping: return "Stopping the camera"
    default: break
    }
    switch sessionState {
    case .starting: return "Connecting to the glasses"
    case .started: return needsPermissionConfirm ? "Waiting for camera permission" : "Connected"
    case .paused: return "Session paused by the glasses"
    case .stopping: return "Ending the session"
    default:
      if isInterrupted { return hasActiveDevice ? "Session ended" : "Folded or out of reach" }
      return hasActiveDevice ? "Ready" : "Waiting for an active device"
    }
  }

  /// Why the wanted camera sends no frames, for the Watch screen; nil when nothing is wrong.
  var interruptionText: String? {
    guard isInterrupted else { return nil }
    switch sessionState {
    case .idle, .stopped:
      return hasActiveDevice
        ? "The glasses' session ended. If watching does not go on by itself, stop and start it."
        : "The glasses are folded or out of reach. Watching goes on when they are back."
    default:
      return "Bringing the glasses' camera back."
    }
  }

  /// Images the decoder handed back unchanged (a frame that failed to decode, or one while
  /// it waits for a keyframe), skipped rather than sent again.
  var repeatedFramesSkipped: Int { repeatFilter.skippedCount }

  @ObservationIgnored private let wearables: WearablesInterface
  @ObservationIgnored private let deviceSelector: AutoDeviceSelector
  @ObservationIgnored private let throttle: FrameThrottle
  // Thread safe: called from the SDK's frame callback, off the main actor.
  @ObservationIgnored private let videoFrameDecoder = VideoFrameDecoder()
  // Thread safe too: the frame callback asks it about every decoded image.
  @ObservationIgnored private let repeatFilter = RepeatedFrameFilter()
  @ObservationIgnored private var deviceSession: DeviceSession?
  @ObservationIgnored private var camera: MWDATCamera.Camera?
  @ObservationIgnored private var wantsStream = false
  /// The camera was stopped to be added again at a new resolution.
  @ObservationIgnored private var isSwitchingResolution = false
  @ObservationIgnored private var sessionRestarts = 0
  @ObservationIgnored private var streamRevivals = 0
  @ObservationIgnored private var streamRevival: Task<Void, Never>?
  @ObservationIgnored private var deviceMonitorTask: Task<Void, Never>?
  @ObservationIgnored private let sessionTokenBag = ListenerTokenBag()
  @ObservationIgnored private let streamTokenBag = ListenerTokenBag()

  init(wearables: WearablesInterface, throttle: FrameThrottle) {
    self.wearables = wearables
    self.deviceSelector = AutoDeviceSelector(wearables: wearables)
    self.throttle = throttle
    startDeviceMonitoring()
  }

  // MARK: - Start and stop

  /// Starts the session; the stream follows once the session reports started. False when
  /// the start was refused (the last session is still ending) or failed.
  @discardableResult
  func start() -> Bool {
    sessionRestarts = 0
    streamRevivals = 0
    recoveries = 0
    return openSession()
  }

  /// Ends the stream and the session; stopping the session cascades to its camera.
  func stop() {
    wantsStream = false
    needsPermissionConfirm = false
    isInterrupted = false
    isSwitchingResolution = false
    streamRevival?.cancel()
    streamRevival = nil
    guard let session = deviceSession else { return }
    sessionState = .stopping
    session.stop()
  }

  /// The size for the stream. A running camera is stopped and added again at the new size
  /// once its stream reports stopped; without one, the next camera is added at this size.
  func setResolution(_ newResolution: Resolution) {
    guard newResolution != resolution else { return }
    resolution = newResolution
    guard wantsStream, let camera, streamState != .stopping else { return }
    isSwitchingResolution = true
    streamState = .stopping
    // Stopping the camera cascades to its stream and detaches it from the session, so a
    // later addCamera() can register a new one (AGENTS.md).
    camera.stop()
  }

  /// The user confirmed: ask for camera access in Meta AI, then start the stream.
  func confirmCameraPermission() async {
    needsPermissionConfirm = false
    isAskingPermission = true
    defer { isAskingPermission = false }
    guard let session = deviceSession, session.state == .started, camera == nil else { return }
    do {
      guard try await wearables.requestPermission(.camera) == .granted else {
        errorMessage = "Camera permission denied"
        permissionRefused = true
        return
      }
    } catch {
      errorMessage = error.localizedDescription
      permissionRefused = true
      return
    }
    beginStream(on: session)
  }

  /// Creates and starts a session, or starts the stream on the one that exists.
  @discardableResult
  private func openSession() -> Bool {
    if deviceSession != nil, sessionState == .stopping {
      errorMessage = "The last session is still ending. Try again in a moment."
      return false
    }
    wantsStream = true
    errorMessage = nil
    permissionRefused = false
    if deviceSession != nil {
      beginStreamIfReady()
      return true
    }
    do {
      let session = try wearables.createSession(deviceSelector: deviceSelector)
      deviceSession = session
      // Subscribe before start() so no initial state transitions are missed.
      observeSession(session)
      sessionState = .starting
      try session.start()
      return true
    } catch {
      if let sessionError = error as? DeviceSessionError {
        handleSessionError(sessionError)
      } else {
        errorMessage = error.localizedDescription
      }
      sessionState = .idle
      cleanupSession()
      return false
    }
  }

  // MARK: - Stream

  private func beginStreamIfReady() {
    guard wantsStream, camera == nil, let session = deviceSession, session.state == .started
    else { return }
    Task {
      do {
        if try await wearables.checkPermissionStatus(.camera) == .granted {
          beginStream(on: session)
        } else {
          needsPermissionConfirm = true
        }
      } catch {
        errorMessage = error.localizedDescription
      }
    }
  }

  private func beginStream(on session: DeviceSession) {
    guard wantsStream, camera == nil else { return }
    let config = StreamConfiguration(
      videoCodec: VideoCodec.hvc1,
      resolution: resolution.streaming,
      frameRate: 7
    )
    do {
      guard let newCamera = try session.addCamera(config: config) else {
        errorMessage = "Couldn't create the camera. Try again."
        return
      }
      camera = newCamera
      // Subscribe before start() so no initial state transitions are missed.
      setupStreamListeners(for: newCamera.stream)
      streamState = .starting
      newCamera.stream.start()
    } catch {
      camera = nil
      errorMessage = error.localizedDescription
    }
  }

  private func setupStreamListeners(for stream: MWDATCamera.Stream) {
    stream.statePublisher.listen { [weak self] state in
      Task { @MainActor in self?.handleStreamStateChange(state) }
    }.store(in: streamTokenBag)

    stream.videoFramePublisher.listen { [weak self] frame in
      guard let self else { return }
      self.handleFrame(frame)
    }.store(in: streamTokenBag)

    stream.errorPublisher.listen { [weak self] error in
      Task { @MainActor in self?.errorMessage = error.localizedDescription }
    }.store(in: streamTokenBag)
  }

  /// Runs on the SDK's callback thread, as CameraAccess's frame handling does. An image the
  /// decoder handed back unchanged went out already (or a newer one did), so it is skipped:
  /// the backend would score it as a new frame.
  nonisolated private func handleFrame(_ frame: VideoFrame) {
    guard let image = videoFrameDecoder.decode(frame.sampleBuffer),
      !repeatFilter.isRepeat(image), throttle.isDue()
    else { return }
    repeatFilter.delivered(image)
    throttle.deliver(image)
  }

  private func handleStreamStateChange(_ state: StreamState) {
    streamState = state
    if state == .streaming {
      // Frames flow again: whatever stopped them is over.
      if isInterrupted { recoveries += 1 }
      isInterrupted = false
      sessionRestarts = 0
      streamRevivals = 0
    } else if state == .stopped {
      clearStreamResources()
      streamStopped()
    }
  }

  /// The one place the stream is torn down, whether the app or the SDK stopped it.
  private func clearStreamResources() {
    streamTokenBag.clear()
    // Detach the camera (idempotent) so a later addCamera() can register a new one.
    camera?.stop()
    camera = nil
    streamState = .stopped
  }

  /// A stream that stopped while the camera is still wanted: added again at the new size
  /// after setResolution(), else revived while its session is up. When the session ended
  /// too (a fold), the device monitor starts a new one once the glasses are back.
  private func streamStopped() {
    guard wantsStream else { return }
    if isSwitchingResolution {
      isSwitchingResolution = false
      beginStreamIfReady()  // a paused session adds it once it is started again
      return
    }
    guard let session = deviceSession, session.state == .started else { return }
    isInterrupted = true
    reviveStream()
  }

  /// BirdSpotter's bringCameraBack: wait a beat, add the camera again, and look again a
  /// beat later, up to streamRevivalAttempts times. A stream that reaches streaming resets
  /// the count; a paused or ended session ends the attempts (AGENTS.md: never restart while
  /// paused).
  private func reviveStream() {
    guard streamRevival == nil else { return }
    streamRevival = Task { [weak self] in
      while true {
        try? await Task.sleep(for: GlassesCamera.streamRevivalDelay)
        guard !Task.isCancelled, let self else { return }
        guard self.needsStreamBack else { break }
        guard self.streamRevivals < GlassesCamera.streamRevivalAttempts else {
          self.errorMessage =
            "The glasses' camera stopped and did not start again. Stop watching and start again."
          break
        }
        self.streamRevivals += 1
        self.beginStreamIfReady()
      }
      self?.streamRevival = nil
    }
  }

  /// The camera is wanted, none is attached, nothing waits on the user, and the session
  /// is started.
  private var needsStreamBack: Bool {
    guard wantsStream, camera == nil, !permissionPending, !permissionRefused,
      let session = deviceSession
    else { return false }
    return session.state == .started
  }

  // MARK: - Session

  private func observeSession(_ session: DeviceSession) {
    session.statePublisher.listen { [weak self] state in
      Task { @MainActor in self?.handleSessionStateChange(state) }
    }.store(in: sessionTokenBag)
    session.errorPublisher.listen { [weak self] error in
      Task { @MainActor in self?.handleSessionError(error) }
    }.store(in: sessionTokenBag)
  }

  private func handleSessionStateChange(_ state: DeviceSessionState) {
    sessionState = state
    if state == .started {
      // Also after a pause: the stream exists already then, and nothing is added twice.
      beginStreamIfReady()
    } else if state == .stopped {
      // A stop that stop() did not ask for (it clears wantsStream first).
      if wantsStream { isInterrupted = true }
      streamRevival?.cancel()
      streamRevival = nil
      cleanupSession()
    }
  }

  private func handleSessionError(_ error: DeviceSessionError) {
    if error == .datAppOnTheGlassesUpdateRequired {
      needsGlassesAppUpdate = true
    }
    errorMessage = error.localizedDescription
  }

  /// Terminal: drop the session so the next start creates a fresh one.
  private func cleanupSession() {
    sessionTokenBag.clear()
    deviceSession = nil
  }

  /// Monitors device availability, as CameraAccess does, and starts a new session when the
  /// glasses come back while the camera is still wanted.
  private func startDeviceMonitoring() {
    deviceMonitorTask = Task { [weak self] in
      guard let self else { return }
      for await deviceId in deviceSelector.activeDeviceStream() {
        hasActiveDevice = deviceId != nil
        if deviceId != nil { restartSessionIfNeeded() }
      }
    }
  }

  /// A new session after a fold or a link drop: only while the camera is wanted and no
  /// session exists, never while one is paused, and at most sessionRestartAttempts times
  /// before a frame arrives. A session that ended while the glasses stayed available (a
  /// system gesture, another app) is not fought over: nothing restarts it until the device
  /// selector reports the glasses again.
  private func restartSessionIfNeeded() {
    guard wantsStream, deviceSession == nil, sessionState != .paused else { return }
    guard sessionRestarts < Self.sessionRestartAttempts else {
      errorMessage =
        "The glasses are back, but no new session started. Stop watching and start again."
      return
    }
    sessionRestarts += 1
    openSession()
  }
}

/// Remembers the last image handed to the throttle, so a repeat of that same object by the
/// decoder is skipped and counted. Thread safe, like FrameThrottle: the SDK's frame
/// callback calls it off the main actor.
final class RepeatedFrameFilter: @unchecked Sendable {
  private let lock = NSLock()
  /// Held, not only compared, so its address cannot be reused by a new image meanwhile.
  private var lastDelivered: UIImage?
  private var skipped = 0

  /// True, and counted, for the very image object delivered last.
  func isRepeat(_ image: UIImage) -> Bool {
    lock.withLock {
      guard lastDelivered === image else { return false }
      skipped += 1
      return true
    }
  }

  func delivered(_ image: UIImage) {
    lock.withLock { lastDelivered = image }
  }

  var skippedCount: Int { lock.withLock { skipped } }
}
