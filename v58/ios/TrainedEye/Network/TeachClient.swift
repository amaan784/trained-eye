//
// TeachClient.swift
//
// The teach API (backend/teach_api.py): POST /teach/start with the request and the
// captured frames as multipart/form-data, GET /teach/status (for one skill, or the latest
// session), POST /teach/answer.
//

import Foundation

extension BackendClient {
  /// Starts a teach run. Send exactly one of `audio` (a WAV recording) and `request`
  /// (typed text). `frames` are the capture's JPEGs in order. Returns the first status,
  /// which asks whether to spend the estimate.
  func startTeach(skill: String, audio: Data?, request: String?, frames: [Data]) async throws
    -> TeachStatus
  {
    var form = MultipartForm()
    form.addField("skill", skill)
    if let audio {
      form.addFile("audio", filename: "request.wav", contentType: "audio/wav", data: audio)
    } else if let request {
      form.addField("request", request)
    }
    for (i, frame) in frames.enumerated() {
      form.addFile(
        "frames", filename: String(format: "f%03d.jpg", i), contentType: "image/jpeg",
        data: frame)
    }
    var urlRequest = self.request("teach/start")  // `request` here is the typed text
    urlRequest.httpMethod = "POST"
    urlRequest.setValue(form.contentType, forHTTPHeaderField: "Content-Type")
    // The server checks every frame and picks keyframes before it answers.
    urlRequest.timeoutInterval = 120
    let (data, response) = try await session.upload(for: urlRequest, from: form.finished())
    try Self.check(response, data)
    return try JSONDecoder().decode(TeachStatus.self, from: data)
  }

  /// GET /teach/status for `skill`; for nil, the query is left out and the backend returns
  /// the latest session since it started (404 when there is none).
  func teachStatus(skill: String?) async throws -> TeachStatus {
    let query = skill.map { [URLQueryItem(name: "skill", value: $0)] } ?? []
    return try await get("teach/status", query: query)
  }

  /// Answers the open question ("spend" or "read_back") and returns the new status.
  func answerTeach(_ answer: TeachAnswer) async throws -> TeachStatus {
    var urlRequest = self.request("teach/answer")
    urlRequest.httpMethod = "POST"
    urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
    urlRequest.httpBody = try JSONEncoder().encode(answer)
    urlRequest.timeoutInterval = 15
    let (data, response) = try await session.data(for: urlRequest)
    try Self.check(response, data)
    return try JSONDecoder().decode(TeachStatus.self, from: data)
  }
}

/// A multipart/form-data body, built in memory (a 10 second capture is about 30 small
/// JPEGs and a few hundred kilobytes of audio).
struct MultipartForm {
  let boundary = "TrainedEye-\(UUID().uuidString)"
  private var body = Data()

  var contentType: String { "multipart/form-data; boundary=\(boundary)" }

  mutating func addField(_ name: String, _ value: String) {
    append("--\(boundary)\r\n")
    append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n")
    append("\(value)\r\n")
  }

  mutating func addFile(_ name: String, filename: String, contentType: String, data: Data) {
    append("--\(boundary)\r\n")
    append("Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(filename)\"\r\n")
    append("Content-Type: \(contentType)\r\n\r\n")
    body.append(data)
    append("\r\n")
  }

  /// The body with its closing boundary.
  func finished() -> Data {
    var out = body
    out.append(Data("--\(boundary)--\r\n".utf8))
    return out
  }

  private mutating func append(_ text: String) {
    body.append(Data(text.utf8))
  }
}
