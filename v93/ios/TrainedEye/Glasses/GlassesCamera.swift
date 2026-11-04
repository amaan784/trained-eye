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
// The stream itself is in GlassesCamera+Stream.swift; the stored properties it uses stay here.
//

import MWDATCamera
import MWDATCore
import Observation
import UIKit

@MainActor
@Observable
final class GlassesCamera {
  /// New sessions started after drops, without a frame arriving in between, before the
  /// app stops trying and asks the user to start watching again.
  static let sessionRestartAttempts = 3

  /// An eligible pair of glasses is available, from the auto device selector.
  private(set) var hasActiveDevice = false
  private(set) var sessionState: DeviceSessionState = .idle
  var streamState: StreamState = .stopped
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
  var isInterrupted = false
  /// How often the camera came back by itself after an interruption, since start().
  var recoveries = 0
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

  @ObservationIgnored let wearables: WearablesInterface
  @ObservationIgnored private let deviceSelector: AutoDeviceSelector
  @ObservationIgnored let throttle: FrameThrottle
  // Thread safe: called from the SDK's frame callback, off the main actor.
  @ObservationIgnored let videoFrameDecoder = VideoFrameDecoder()
  // Thread safe too: the frame callback asks it about every decoded image.
  @ObservationIgnored let repeatFilter = RepeatedFrameFilter()
  @ObservationIgnored var deviceSession: DeviceSession?
  @ObservationIgnored var camera: MWDATCamera.Camera?
  @ObservationIgnored var wantsStream = false
  /// The camera was stopped to be added again at a new resolution.
  @ObservationIgnored var isSwitchingResolution = false
  @ObservationIgnored var sessionRestarts = 0
  @ObservationIgnored var streamRevivals = 0
  @ObservationIgnored var streamRevival: Task<Void, Never>?
  @ObservationIgnored private var deviceMonitorTask: Task<Void, Never>?
  @ObservationIgnored private let sessionTokenBag = ListenerTokenBag()
  @ObservationIgnored let streamTokenBag = ListenerTokenBag()

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
