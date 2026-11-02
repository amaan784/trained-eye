//
// WatchSocket.swift
//
// The WebSocket to the backend (protocol 1, backend/server.py): a hello naming the camera
// and the watched skills, then one JPEG frame a second as a binary message. It reads the
// welcome, the acks (with each watched skill's score) and the alerts. One frame is in
// flight at a time: the next goes only after the server acked the last, so a server that
// takes longer than a second to score drops frames here instead of queueing them. It
// reconnects with
// a growing pause (1, 2, 4, 8, then 15 seconds) whenever the connection drops. Each
// reconnect is a new session on the server, so its 3-of-4 frame count starts again.
//

import Foundation
import Observation

@MainActor
@Observable
final class WatchSocket {
  enum Status: Equatable {
    case off
    case connecting
    case live
    case retrying(seconds: Int)

    var label: String {
      switch self {
      case .off: return "Not connected"
      case .connecting: return "Connecting"
      case .live: return "Connected"
      case .retrying(let seconds): return "Connection lost, trying again in \(seconds) s"
      }
    }
  }

  private(set) var status: Status = .off
  private(set) var session: String?
  /// The skills the server is scoring, from its welcome.
  private(set) var watching: [String] = []
  /// The latest score per skill, from the latest ack.
  private(set) var scores: [String: Double?] = [:]
  private(set) var framesSent = 0
  private(set) var framesAcked = 0
  private(set) var lastError: String?

  /// Called on the main actor for every alert message.
  @ObservationIgnored var onAlert: ((ServerMessage) -> Void)?

  // TODO: no keepalive ping yet. A tunnel may close a socket that is idle while the
  // glasses' stream is paused; URLSessionWebSocketTask.sendPing could keep it open.
  @ObservationIgnored private var task: URLSessionWebSocketTask?
  @ObservationIgnored private var runner: Task<Void, Never>?
  @ObservationIgnored private var isSending = false
  /// When the frame in flight was sent, until its ack (or an error) arrives.
  @ObservationIgnored private var awaitingAckSince: Date?
  /// After this long without an ack the server is taken to be stuck (each watched skill
  /// can take the student's 60 s timeout): the socket is dropped and opened again, which
  /// starts a new server session with nothing queued.
  private static let ackTimeout: TimeInterval = 90
  @ObservationIgnored private var retryDelay = 1
  @ObservationIgnored private let urlSession = URLSession(configuration: .default)

  /// Opens the socket, closing any earlier one, and keeps it open until disconnect().
  func connect(to url: URL, hello: Hello) {
    disconnect()
    lastError = nil
    framesSent = 0
    framesAcked = 0
    runner = Task { [weak self] in
      await self?.run(url: url, hello: hello)
    }
  }

  /// Says bye and closes the socket.
  func disconnect() {
    runner?.cancel()
    runner = nil
    if let task {
      self.task = nil
      Task {
        try? await task.send(.string(#"{"type": "bye"}"#))
        task.cancel(with: .normalClosure, reason: nil)
      }
    }
    status = .off
    session = nil
    watching = []
    scores = [:]
    isSending = false
    awaitingAckSince = nil
  }

  /// Sends one frame if the socket is live and the server acked the previous one. A frame
  /// that would have to wait is dropped: a newer one follows a second later.
  func send(jpeg: Data) {
    guard status == .live, let task, !isSending else { return }
    if let since = awaitingAckSince {
      if Date().timeIntervalSince(since) >= Self.ackTimeout {
        lastError = "No answer from the backend for a frame: reconnecting."
        task.cancel(with: .goingAway, reason: nil)  // receive() fails and run() reconnects
      }
      return
    }
    isSending = true
    awaitingAckSince = Date()
    framesSent += 1
    Task { [weak self] in
      do {
        try await task.send(.data(jpeg))
      } catch {
        self?.lastError = error.localizedDescription
        self?.awaitingAckSince = nil
      }
      self?.isSending = false
    }
  }

  private func run(url: URL, hello: Hello) async {
    retryDelay = 1
    while !Task.isCancelled {
      status = .connecting
      let task = urlSession.webSocketTask(with: url)
      self.task = task
      task.resume()
      do {
        let helloText = String(decoding: try JSONEncoder().encode(hello), as: UTF8.self)
        try await task.send(.string(helloText))
        while true {
          let message = try await task.receive()
          // A message that lands after disconnect() belongs to nobody.
          guard !Task.isCancelled else { return }
          switch message {
          case .string(let text): handle(Data(text.utf8))
          case .data(let data): handle(data)
          @unknown default: break
          }
        }
      } catch {
        // disconnect() closes the task, which ends receive() with an error: not a failure.
        guard !Task.isCancelled else { return }
        lastError = error.localizedDescription
      }
      task.cancel(with: .goingAway, reason: nil)
      if self.task === task { self.task = nil }
      session = nil
      isSending = false
      awaitingAckSince = nil
      let delay = retryDelay
      status = .retrying(seconds: delay)
      try? await Task.sleep(for: .seconds(delay))
      retryDelay = min(retryDelay * 2, 15)
    }
  }

  private func handle(_ data: Data) {
    guard let message = try? JSONDecoder().decode(ServerMessage.self, from: data) else { return }
    switch message.type {
    case "welcome":
      status = .live
      retryDelay = 1
      session = message.session
      watching = message.watching ?? []
    case "ack":
      framesAcked += 1
      awaitingAckSince = nil
      if let scores = message.scores { self.scores = scores }
    case "alert":
      onAlert?(message)
    case "error":
      // For example "cannot watch: ..." before the welcome, or a frame the server could
      // not read. The session goes on either way.
      lastError = message.error
      awaitingAckSince = nil
    default:
      break
    }
  }
}
