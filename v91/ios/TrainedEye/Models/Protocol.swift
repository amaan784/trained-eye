//
// Protocol.swift
//
// The messages of the backend's WebSocket protocol (version 1, backend/server.py) and of
// GET /health and GET /skills.
//

import Foundation

/// The first message on the WebSocket.
struct Hello: Encodable {
  let type = "hello"
  let client = "ios"
  /// "glasses", "mock" (the DAT mock device in Debug builds) or "phone".
  let source: String
  let fps = 1
  /// The skills to watch. Nil leaves the key out, so frames are only acked; the server
  /// refuses an empty list.
  let watch: [String]?
}

/// Any message from the server. Which fields are set depends on `type`:
/// welcome (session, watching), ack (seq, bytes, width, height, scores),
/// alert (skill, text, t, p, confirmed) and error (error, sometimes seq).
struct ServerMessage: Decodable {
  let type: String
  let session: String?
  let watching: [String]?
  let seq: Int?
  let bytes: Int?
  let width: Int?
  let height: Int?
  /// skill name to yes-probability, or nil when the student failed on that frame.
  let scores: [String: Double?]?
  let error: String?
  let skill: String?
  let text: String?
  let t: Double?
  let p: Double?
  /// Whether Muse Spark agreed with the alert; nil when there was no cloud check or it failed.
  let confirmed: Bool?
}

/// GET /health
struct Health: Decodable {
  let ok: Bool
  let protocolVersion: Int

  enum CodingKeys: String, CodingKey {
    case ok
    case protocolVersion = "protocol"
  }
}

/// One entry of GET /skills.
struct SkillInfo: Decodable, Identifiable, Hashable {
  let name: String
  let alert: String?
  let cooldownSeconds: Int?
  /// True once the teach flow has written the skill's dataset.
  let dataset: Bool
  /// True once `te calibrate` has set a threshold from labelled validation clips.
  let calibrated: Bool
  let threshold: Double?

  var id: String { name }

  enum CodingKeys: String, CodingKey {
    case name, alert, dataset, calibrated, threshold
    case cooldownSeconds = "cooldown_s"
  }

  /// The status line on the Skills screen.
  var statusText: String {
    if calibrated, let threshold {
      return String(format: "Ready, threshold %.2f", threshold)
    }
    if dataset { return "Dataset built; train and calibrate on the Mac" }
    return "Taught, no dataset yet"
  }
}

/// GET /skills
struct SkillList: Decodable {
  let skills: [SkillInfo]
}

/// An alert as the Watch screen shows it.
struct AlertEvent: Identifiable {
  let id = UUID()
  let skill: String
  let text: String
  let probability: Double?
  let confirmed: Bool?
  let receivedAt: Date
  /// False when no Bluetooth output was connected, so nothing was said aloud.
  let spoken: Bool
}

/// One camera frame as JPEG, with the time it was taken (seconds since boot), so frames
/// keep their order even if they reach the main actor out of order.
struct CapturedFrame: Sendable {
  let jpeg: Data
  let time: TimeInterval
}
