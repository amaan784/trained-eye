//
// TeachModels.swift
//
// The teach API's messages (backend/teach_api.py): the status that POST /teach/start,
// GET /teach/status and POST /teach/answer all return, and the answer the app sends.
//

import Foundation

/// The states of a teach session (backend/teach_session.py).
enum TeachState: String {
  case waitingSpend = "waiting_spend"
  case running
  case waitingReadBack = "waiting_read_back"
  case done
  case stopped
  case failed
  case unknown
}

/// What the phone shows at the end of a run that wrote a dataset.
struct TeachSummary: Decodable {
  let examples: Int?
  let yes: Int?
  let no: Int?
  let generated: Int?
  let generatedKept: Int?
  let agreement: Double?
  let warning: String?

  enum CodingKeys: String, CodingKey {
    case examples, yes, no, generated, agreement, warning
    case generatedKept = "generated_kept"
  }
}

/// The status of one teach session. Every field is read leniently, so a field the app
/// does not know, or a null, never hides the rest.
struct TeachStatus: Decodable {
  let skill: String?
  let state: TeachState
  /// "spend" or "read_back" while the server waits for that answer, else nil.
  let question: String?
  let lines: [String]
  /// Cost parts in US dollars: transcribe, compile, sam, images, labels, total, at_most.
  let estimate: [String: Double]
  let warning: String?
  let readBack: String?
  let stopped: String?
  let error: String?
  let summary: TeachSummary?
  let spentUSD: Double?

  enum CodingKeys: String, CodingKey {
    case skill, state, question, lines, estimate, warning, stopped, error, summary
    case readBack = "read_back"
    case spentUSD = "spent_usd"
  }

  init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    skill = try? c.decodeIfPresent(String.self, forKey: .skill)
    let stateText: String = (try? c.decodeIfPresent(String.self, forKey: .state)) ?? ""
    state = TeachState(rawValue: stateText) ?? .unknown
    question = try? c.decodeIfPresent(String.self, forKey: .question)
    lines = (try? c.decodeIfPresent([String].self, forKey: .lines)) ?? []
    estimate = (try? c.decodeIfPresent([String: Double].self, forKey: .estimate)) ?? [:]
    warning = try? c.decodeIfPresent(String.self, forKey: .warning)
    readBack = try? c.decodeIfPresent(String.self, forKey: .readBack)
    stopped = try? c.decodeIfPresent(String.self, forKey: .stopped)
    error = try? c.decodeIfPresent(String.self, forKey: .error)
    summary = try? c.decodeIfPresent(TeachSummary.self, forKey: .summary)
    spentUSD = try? c.decodeIfPresent(Double.self, forKey: .spentUSD)
  }

  var isFinished: Bool { state == .done || state == .stopped || state == .failed }
  var asksSpend: Bool { question == "spend" }
  var asksReadBack: Bool { question == "read_back" }
  var estimateTotal: Double? { estimate["total"] }
  var estimateAtMost: Double? { estimate["at_most"] }
}

/// POST /teach/answer. `correction` goes only with step "read_back" and yes false.
struct TeachAnswer: Encodable {
  let skill: String
  let step: String
  let yes: Bool
  let correction: String?
}
