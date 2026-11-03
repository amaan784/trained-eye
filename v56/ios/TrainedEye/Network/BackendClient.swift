//
// BackendClient.swift
//
// Plain HTTP calls to the backend (`te serve`): GET /health, GET /skills, and the shared
// helpers the teach calls use (TeachClient.swift). Every request carries the server token
// when one is set; the teach routes need it.
//

import Foundation

enum BackendError: LocalizedError {
  case noAddress
  case http(status: Int, message: String)

  var errorDescription: String? {
    switch self {
    case .noAddress:
      return "Set the backend address on the Connect screen first."
    case .http(let status, let message):
      return message.isEmpty ? "The backend answered \(status)." : message
    }
  }
}

/// The error body the backend sends with a 4xx or 5xx: {"error": "...", ...}.
struct BackendErrorBody: Decodable {
  let error: String?
  let detail: String?
}

struct BackendClient {
  let base: URL
  var token = ""
  var session: URLSession = .shared

  /// GET /health: {"ok": true, "protocol": 1}
  func health() async throws -> Health {
    try await get("health")
  }

  /// GET /skills: every taught skill that was not refused.
  func skills() async throws -> [SkillInfo] {
    let list: SkillList = try await get("skills")
    return list.skills
  }

  func get<T: Decodable>(_ path: String, query: [URLQueryItem] = []) async throws -> T {
    var urlRequest = request(path, query: query)
    urlRequest.timeoutInterval = 10
    let (data, response) = try await session.data(for: urlRequest)
    try Self.check(response, data)
    return try JSONDecoder().decode(T.self, from: data)
  }

  /// A request to `path` with the server token, if any.
  func request(_ path: String, query: [URLQueryItem] = []) -> URLRequest {
    var urlRequest = URLRequest(url: url(path, query: query))
    if !token.isEmpty {
      urlRequest.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
    }
    return urlRequest
  }

  func url(_ path: String, query: [URLQueryItem] = []) -> URL {
    let full = base.appending(path: path)
    guard !query.isEmpty,
      var parts = URLComponents(url: full, resolvingAgainstBaseURL: false)
    else { return full }
    parts.queryItems = query
    return parts.url ?? full
  }

  /// Throws BackendError.http for a status outside 200 to 299, with the backend's own
  /// {"error": "..."} message when there is one.
  static func check(_ response: URLResponse, _ data: Data) throws {
    guard let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode)
    else { return }
    let body = try? JSONDecoder().decode(BackendErrorBody.self, from: data)
    let text = body?.error ?? body?.detail ?? String(decoding: data.prefix(300), as: UTF8.self)
    throw BackendError.http(status: http.statusCode, message: text)
  }
}
