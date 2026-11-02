//
// FrameThrottle.swift
//
// Turns a camera's frames into JPEGs at a fixed pace: one a second while watching (the
// backend scores frames at 1 fps) and three a second during the teach capture (the pace
// the backend reads a folder of frames at). Thread safe: the cameras call it from their
// own queues, and only frames that are due are encoded.
//

import Foundation
import UIKit

final class FrameThrottle: @unchecked Sendable {
  /// JPEG quality for frames sent over Wi-Fi; the backend pads them to 512x512 anyway.
  static let jpegQuality: CGFloat = 0.7

  private let lock = NSLock()
  private var interval: TimeInterval = 1
  private var lastDelivered: TimeInterval = 0
  private var sink: (@Sendable (CapturedFrame) -> Void)?

  /// Where encoded frames go. The closure is called on the camera's queue.
  func setSink(_ sink: (@Sendable (CapturedFrame) -> Void)?) {
    lock.withLock { self.sink = sink }
  }

  /// Seconds between delivered frames.
  func setInterval(_ seconds: TimeInterval) {
    lock.withLock {
      interval = seconds
      lastDelivered = 0
    }
  }

  /// True, and the start of a new interval, when the next frame is due. A camera calls
  /// this first so it skips converting frames that would be dropped. The schedule moves
  /// on by one interval each time, so 7 fps thinned to 3 a second gives 3 a second, not
  /// every third frame; after a gap of two intervals or more it starts again from now.
  func isDue() -> Bool {
    let now = ProcessInfo.processInfo.systemUptime
    return lock.withLock {
      guard sink != nil, now - lastDelivered >= interval else { return false }
      lastDelivered = now - lastDelivered >= 2 * interval ? now : lastDelivered + interval
      return true
    }
  }

  /// Encodes a due frame as JPEG and hands it to the sink.
  func deliver(_ image: UIImage) {
    let time = ProcessInfo.processInfo.systemUptime
    guard let sink = lock.withLock({ self.sink }),
      let jpeg = image.jpegData(compressionQuality: Self.jpegQuality)
    else { return }
    sink(CapturedFrame(jpeg: jpeg, time: time))
  }
}
