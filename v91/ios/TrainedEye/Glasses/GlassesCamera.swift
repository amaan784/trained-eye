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
// TODO: verify on the glasses that the stream, and the app with its WebSocket, keep
// running with the screen locked. BirdSpotter shows hvc1 streaming in the background, but
// no sample sends frames over the network from there.
//

import MWDATCamera
import MWDATCore
import Observation
import UIKit

@MainActor
@Observable
final class GlassesCamera {
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
    default: return hasActiveDevice ? "Ready" : "Waiting for an active device"
    }
  }

  @ObservationIgnored private let wearables: WearablesInterface
  @ObservationIgnored private let deviceSelector: AutoDeviceSelector
  @ObservationIgnored private let throttle: FrameThrottle
  // Thread safe: called from the SDK's frame callback, off the main actor.
  @ObservationIgnored private let videoFrameDecoder = VideoFrameDecoder()
  @ObservationIgnored private var deviceSession: DeviceSession?
  @ObservationIgnored private var camera: MWDATCamera.Camera?
  @ObservationIgnored private var wantsStream = false
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

  /// Ends the stream and the session; stopping the session cascades to its camera.
  func stop() {
    wantsStream = false
    needsPermissionConfirm = false
    guard let session = deviceSession else { return }
    sessionState = .stopping
    session.stop()
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
      resolution: StreamingResolution.low,
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

  /// Runs on the SDK's callback thread, as CameraAccess's frame handling does.
  nonisolated private func handleFrame(_ frame: VideoFrame) {
    guard let image = videoFrameDecoder.decode(frame.sampleBuffer) else { return }
    guard throttle.isDue() else { return }
    throttle.deliver(image)
  }

  private func handleStreamStateChange(_ state: StreamState) {
    streamState = state
    if state == .stopped {
      clearStreamResources()
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
      cleanupSession()
    }
  }

  private func handleSessionError(_ error: DeviceSessionError) {
    if error == .datAppOnTheGlassesUpdateRequired {
      needsGlassesAppUpdate = true
    }
    errorMessage = error.localizedDescription
  }

  /// Terminal: drop the session so the next start() creates a fresh one.
  private func cleanupSession() {
    sessionTokenBag.clear()
    deviceSession = nil
  }

  private func startDeviceMonitoring() {
    deviceMonitorTask = Task { [weak self] in
      guard let self else { return }
      for await deviceId in deviceSelector.activeDeviceStream() {
        hasActiveDevice = deviceId != nil
      }
    }
  }
}
