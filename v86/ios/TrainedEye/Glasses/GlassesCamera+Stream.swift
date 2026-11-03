//
// GlassesCamera+Stream.swift
//
// The stream half of GlassesCamera (GlassesCamera.swift): the sizes it asks for, the camera
// added to a started session, the stream's state, frame and error listeners, the frames on
// the SDK's callback thread, the one teardown, and a stream that stopped on its own brought
// back. The stored properties these use stay in the class: an extension cannot hold them.
//

import MWDATCamera
import MWDATCore
import UIKit

extension GlassesCamera {
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

  // MARK: - Stream

  func beginStreamIfReady() {
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

  func beginStream(on session: DeviceSession) {
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
