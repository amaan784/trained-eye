//
// Protocol.swift
//
// The messages of the backend's WebSocket protocol (version 1, backend/server.py) and of
// GET /health and GET /skills. Each skill in GET /skills may carry "training" (null, or
// {"run", "state": "running" | "done" | "failed", "started", "finished"}: the newest te
// train run whose bundle lists it) and "served" (whether the run te serve-student last
// deployed lists it). Both are optional, so an older backend still decodes.
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

/// The newest `te train` run whose bundle lists a skill (GET /skills "training"), from
/// data/train/<run>.status.json on the Mac.
struct TrainingRun: Decodable, Hashable {
  enum State: String {
    case running
    case done
    case failed
    /// A state this app does not know yet.
    case unknown
  }

  let run: String
  let state: State
  /// ISO times as the Mac wrote them; `finished` is nil while the run is going.
  let started: String?
  let finished: String?

  enum CodingKeys: String, CodingKey {
    case run, state, started, finished
  }

  /// Read leniently, like TeachStatus: a field the app does not know never hides the rest.
  init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    run = (try? c.decodeIfPresent(String.self, forKey: .run)) ?? ""
    let stateText: String = (try? c.decodeIfPresent(String.self, forKey: .state)) ?? ""
    state = State(rawValue: stateText) ?? .unknown
    started = try? c.decodeIfPresent(String.self, forKey: .started)
    finished = try? c.decodeIfPresent(String.self, forKey: .finished)
  }

  /// "Nov 2, 2026 at 14:03" for an ISO time, or the text itself if it does not parse.
  static func display(_ isoTime: String) -> String {
    let choices: [ISO8601DateFormatter.Options] = [
      [.withInternetDateTime, .withFractionalSeconds], [.withInternetDateTime],
    ]
    for options in choices {
      let formatter = ISO8601DateFormatter()
      formatter.formatOptions = options
      if let date = formatter.date(from: isoTime) {
        return date.formatted(date: .abbreviated, time: .shortened)
      }
    }
    return isoTime
  }
}

/// One entry of GET /skills.
struct SkillInfo: Decodable, Identifiable, Hashable {
  let name: String
  let alert: String?
  let cooldownSeconds: Int?
  /// How long the condition must stay true before it alerts; nil from a backend older
  /// than the field.
  let holdSeconds: Int?
  /// True once the teach flow has written the skill's dataset.
  let dataset: Bool
  /// True once `te calibrate` has set a threshold from labelled validation clips.
  let calibrated: Bool
  let threshold: Double?
  /// The newest training run whose bundle lists the skill; nil when none does, or from a
  /// backend older than the field.
  let training: TrainingRun?
  /// Whether the run `te serve-student` last deployed lists the skill; nil from a backend
  /// older than the field.
  let served: Bool?

  var id: String { name }

  enum CodingKeys: String, CodingKey {
    case name, alert, dataset, calibrated, threshold, training, served
    case cooldownSeconds = "cooldown_s"
    case holdSeconds = "hold_s"
  }

  init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    name = try c.decode(String.self, forKey: .name)
    alert = try c.decodeIfPresent(String.self, forKey: .alert)
    cooldownSeconds = try c.decodeIfPresent(Int.self, forKey: .cooldownSeconds)
    holdSeconds = try? c.decodeIfPresent(Int.self, forKey: .holdSeconds)
    dataset = try c.decode(Bool.self, forKey: .dataset)
    calibrated = try c.decode(Bool.self, forKey: .calibrated)
    threshold = try c.decodeIfPresent(Double.self, forKey: .threshold)
    // Newer backends only, and read leniently: an odd value never hides the skill.
    training = try? c.decodeIfPresent(TrainingRun.self, forKey: .training)
    served = try? c.decodeIfPresent(Bool.self, forKey: .served)
  }

  /// Watch mode can score it: calibrated, and served by the deployed model (an older
  /// backend does not say, and then calibrated is enough, as before).
  var isReady: Bool { calibrated && threshold != nil && served != false }

  /// The status line on the Skills screen: Training, Trained or Ready, else how far the
  /// teach flow got.
  var statusText: String {
    if let training, training.state == .running {
      return "Training on Modal (\(training.run))"
    }
    if isReady, let threshold {
      return String(format: "Ready, threshold %.2f", threshold)
    }
    if served == true {
      return "Trained and served; calibrate on the Mac (te calibrate)"
    }
    if let training, training.state == .done {
      return "Trained (\(training.run)); serve it and calibrate on the Mac"
    }
    if let training, training.state == .failed {
      return "The last training run (\(training.run)) failed"
    }
    if dataset { return "Dataset built; train and calibrate on the Mac" }
    return "Taught, no dataset yet"
  }

  /// When the newest run started, or finished, for the Skills screen.
  var trainingTimeText: String? {
    guard let training else { return nil }
    if let finished = training.finished {
      return "Run \(training.run) ended \(TrainingRun.display(finished))"
    }
    return training.started.map { "Run \(training.run) started \(TrainingRun.display($0))" }
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
/// keep their order even if they reach the main actor out of order, and its size in pixels.
struct CapturedFrame: Sendable {
  let jpeg: Data
  let time: TimeInterval
  let width: Int
  let height: Int
}
