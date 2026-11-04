//
// WatchSocket.swift
//
// The WebSocket to the backend (protocol 1, backend/server.py): a hello naming the camera
// and the watched skills, then one JPEG frame a second as a binary message. It reads the
// welcome, the acks (with each watched skill's score) and the alerts. One frame is in
// flight at a time: the next goes only after the server acked the last, so a server that
// takes longer than a second to score drops frames here instead of queueing them. It
// reconnects with a growing pause (1, 2, 4, 8, then 15 seconds) whenever the connection
// drops. Each reconnect is a new session on the server, so its 3-of-4 frame count starts
// again. After the welcome, a keepalive pings every 20 seconds: a tunnel may close a socket
// left idle while the glasses' stream is paused, and a link that went quiet is noticed only
// when a ping gets no pong within 10 seconds (te serve's uvicorn pings the phone every 20 s
// too, and URLSessionWebSocketTask answers those by itself).
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
  /// How long each skill's condition has held so far, from the latest ack.
  private(set) var held: [String: Double?] = [:]
  /// The model behind the backend: "waking", "ready" or "failed" (nil before the first ack).
  private(set) var modelState: String?
  /// When the backend first said it was waking the model, and how long that usually takes.
  private(set) var wakingSince: Date?
  private(set) var wakeExpected: Double = 180
  private(set) var framesSent = 0
  private(set) var framesAcked = 0
  private(set) var lastError: String?

  /// Called on the main actor for every alert message.
  @ObservationIgnored var onAlert: ((ServerMessage) -> Void)?
  /// Each ack's scores, in the order the frames were sent (the Watch screen's strip).
  @ObservationIgnored var onScores: (([String: Double?]) -> Void)?

  @ObservationIgnored private var task: URLSessionWebSocketTask?
  @ObservationIgnored private var runner: Task<Void, Never>?
  /// Pings while the socket is live; cancelled when its loop iteration ends.
  @ObservationIgnored private var keepalive: Task<Void, Never>?
  /// Why the app dropped the socket itself (no ack, no pong), shown instead of the error
  /// receive() then ends with.
  @ObservationIgnored private var dropReason: String?
  static let pingInterval = Duration.seconds(20)
  static let pongTimeout: TimeInterval = 10
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
    stopKeepalive()
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
    held = [:]
    modelState = nil
    wakingSince = nil
    isSending = false
    awaitingAckSince = nil
  }

  /// Sends one frame if the socket is live and the server acked the previous one. A frame
  /// that would have to wait is dropped: a newer one follows a second later.
  func send(jpeg: Data) {
    guard status == .live, let task, !isSending else { return }
    if let since = awaitingAckSince {
      if Date().timeIntervalSince(since) >= Self.ackTimeout {
        dropReason = "No answer from the backend for a frame: reconnecting."
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
      dropReason = nil
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
        lastError = dropReason ?? error.localizedDescription
      }
      stopKeepalive()
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
      if let task { startKeepalive(on: task) }
    case "ack":
      framesAcked += 1
      awaitingAckSince = nil
      if let scores = message.scores { self.scores = scores }
      if let held = message.held { self.held = held }
      if let model = message.model {
        if model == "waking" {
          // from the backend's own clock, so a wake started on Skills keeps its time
          wakingSince = Date().addingTimeInterval(-(message.elapsed ?? 0))
          wakeExpected = message.expected ?? wakeExpected
        } else {
          wakingSince = nil
        }
        modelState = model
      }
      onScores?(message.scores ?? [:])
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

  // MARK: - Keepalive

  /// Every pingInterval, a ping. A failed ping, or no pong within pongTimeout, drops the
  /// socket, so receive() throws and run() reconnects with its pause.
  private func startKeepalive(on task: URLSessionWebSocketTask) {
    keepalive?.cancel()
    keepalive = Task { [weak self] in
      while true {
        try? await Task.sleep(for: WatchSocket.pingInterval)
        guard !Task.isCancelled else { return }
        let failure = await WatchSocket.ping(task, timeout: WatchSocket.pongTimeout)
        guard !Task.isCancelled else { return }
        guard let failure else { continue }
        self?.dropReason = failure
        task.cancel(with: .goingAway, reason: nil)  // receive() fails and run() reconnects
        return
      }
    }
  }

  private func stopKeepalive() {
    keepalive?.cancel()
    keepalive = nil
  }

  /// Sends a ping and waits for its pong, at most `timeout` seconds: nil for a pong, else
  /// what went wrong. The pong handler runs on URLSession's queue, or never, so whichever of
  /// it and the timeout comes first answers.
  private static func ping(_ task: URLSessionWebSocketTask, timeout: TimeInterval) async
    -> String?
  {
    await withCheckedContinuation { continuation in
      let answer = FirstAnswer(continuation)
      task.sendPing { error in
        answer.give(error.map { "The keepalive ping failed: \($0.localizedDescription)" })
      }
      DispatchQueue.main.asyncAfter(deadline: .now() + timeout) {
        answer.give("No pong from the backend in \(Int(timeout)) s: reconnecting.")
      }
    }
  }
}

/// Resumes a continuation once, with whichever answer comes first. Thread safe.
private final class FirstAnswer: @unchecked Sendable {
  private let lock = NSLock()
  private var continuation: CheckedContinuation<String?, Never>?

  init(_ continuation: CheckedContinuation<String?, Never>) {
    self.continuation = continuation
  }

  func give(_ answer: String?) {
    let waiting = lock.withLock { () -> CheckedContinuation<String?, Never>? in
      let first = continuation
      continuation = nil
      return first
    }
    waiting?.resume(returning: answer)
  }
}
