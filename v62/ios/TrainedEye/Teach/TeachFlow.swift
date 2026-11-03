//
// TeachFlow.swift
//
// The Teach screen's steps: name the skill, hold to talk (or type) the request, a 10
// second capture after a 3 second countdown that keeps three frames a second in memory,
// the upload to /teach/start, then the server's two questions (spend the estimate? is the
// read-back right?) and its progress lines until the run ends. Frames are only held in
// memory: they are dropped once the Mac has them, or on Start over. The glasses stream at
// medium resolution (504x896) for the capture (AppModel). After a relaunch, a session still
// running on the Mac is picked up again from GET /teach/status, which without a skill
// returns the latest one.
//

import Foundation
import Observation

@MainActor
@Observable
final class TeachFlow {
  enum Phase: Equatable {
    case compose
    case startingCamera
    case countdown(Int)
    case capturing(secondsLeft: Int)
    case uploading
    case onServer
  }

  static let captureSeconds = 10
  /// The pace the backend reads a folder of frames at (keyframes.load_frames).
  static let captureInterval: TimeInterval = 1.0 / 3.0

  var skillName = ""
  var typedRequest = ""
  var correction = ""
  private(set) var phase: Phase = .compose
  private(set) var isRecording = false
  private(set) var requestSeconds: Double?
  /// Which microphone recorded the request: "glasses", "iPhone", or both in order when the
  /// route changed part way ("glasses, then iPhone").
  private(set) var requestMicrophone = "iPhone"
  private(set) var frameCount = 0
  private(set) var activeSkill: String?
  private(set) var status: TeachStatus?
  private(set) var isAnswering = false
  var errorMessage: String?

  /// Set by AppModel, which starts the camera and hands frames over while capturing.
  @ObservationIgnored weak var app: AppModel?
  @ObservationIgnored private let recorder = RequestRecorder()
  @ObservationIgnored private var requestAudio: Data?
  @ObservationIgnored private var frames: [CapturedFrame] = []
  @ObservationIgnored private var wantsRecording = false
  @ObservationIgnored private var captureTask: Task<Void, Never>?
  @ObservationIgnored private var pollTask: Task<Void, Never>?
  @ObservationIgnored private var didTryResume = false

  var isCapturing: Bool {
    if case .capturing = phase { return true }
    return false
  }

  /// The backend's rule: snake_case, 2 to 40 characters, starting with a letter.
  var isNameValid: Bool {
    skillName.range(of: "^[a-z][a-z0-9_]{1,39}$", options: .regularExpression) != nil
  }

  var hasRequest: Bool {
    requestSeconds != nil || !typedRequest.trimmingCharacters(in: .whitespaces).isEmpty
  }

  var canSend: Bool { phase == .compose && isNameValid && hasRequest && frameCount > 0 }

  /// Start over is offered before sending and once the server's run has ended.
  var canStartOver: Bool { phase == .compose || status?.isFinished == true }

  // MARK: - The spoken request

  func pressToTalk() {
    guard !isRecording, !wantsRecording, phase == .compose else { return }
    wantsRecording = true
    app?.speaker.setListening(true)  // never listen and speak at once
    Task {
      do {
        try await recorder.start()
        isRecording = true
        if !wantsRecording { releaseToTalk() }  // let go before the recorder started
      } catch {
        wantsRecording = false
        errorMessage = error.localizedDescription
        app?.speaker.setListening(false)
      }
    }
  }

  func releaseToTalk() {
    wantsRecording = false
    guard isRecording else { return }
    isRecording = false
    if let result = recorder.stop() {
      requestAudio = result.wav
      requestSeconds = result.seconds
      requestMicrophone = recorder.microphoneText
      typedRequest = ""
    }
    app?.speaker.setListening(false)
  }

  func clearRecording() {
    requestAudio = nil
    requestSeconds = nil
  }

  // MARK: - The 10 second capture

  func startCapture() {
    guard phase == .compose, let app else { return }
    errorMessage = nil
    phase = .startingCamera
    captureTask = Task {
      guard await app.startCameraForTeach() else {
        phase = .compose
        errorMessage = app.message ?? "The camera did not start."
        return
      }
      for n in stride(from: 3, through: 1, by: -1) {
        phase = .countdown(n)
        try? await Task.sleep(for: .seconds(1))
        guard !Task.isCancelled else { return }
      }
      frames = []
      frameCount = 0
      for left in stride(from: Self.captureSeconds, through: 1, by: -1) {
        phase = .capturing(secondsLeft: left)
        try? await Task.sleep(for: .seconds(1))
        guard !Task.isCancelled else { return }
      }
      phase = .compose
      app.endTeachCapture()
    }
  }

  /// Called by AppModel for every frame while capturing.
  func add(_ frame: CapturedFrame) {
    guard isCapturing else { return }
    frames.append(frame)
    frameCount = frames.count
  }

  // MARK: - On the Mac

  /// Run once at launch: picks up the newest session on the Mac if it is still running, so
  /// its spend or read-back question can be answered here. Only on a fresh Teach screen,
  /// with an address and the server token set (the teach routes need it); a 404 (no session
  /// since the server started) or a network error leaves the screen as it is.
  func resumeIfRunning() async {
    guard !didTryResume else { return }
    didTryResume = true
    guard phase == .compose, status == nil, let backend = app?.backend, !backend.token.isEmpty
    else { return }
    guard let latest = try? await backend.teachStatus(skill: nil), !latest.isFinished,
      let skill = latest.skill
    else { return }
    // The user may have started something while the request was out.
    guard phase == .compose, status == nil else { return }
    activeSkill = skill
    status = latest
    phase = .onServer
    startPolling(skill: skill)
  }

  func send() {
    guard canSend else { return }
    guard let backend = app?.backend else {
      errorMessage = BackendError.noAddress.localizedDescription
      return
    }
    let skill = skillName
    let audio = requestAudio
    let text = audio == nil ? typedRequest.trimmingCharacters(in: .whitespacesAndNewlines) : nil
    let jpegs = frames.sorted { $0.time < $1.time }.map(\.jpeg)
    phase = .uploading
    errorMessage = nil
    Task {
      do {
        status = try await backend.startTeach(
          skill: skill, audio: audio, request: text, frames: jpegs)
        frames = []  // the Mac has them now
        frameCount = 0
        requestAudio = nil
        activeSkill = skill
        phase = .onServer
        startPolling(skill: skill)
      } catch {
        errorMessage = error.localizedDescription  // the capture is kept for another try
        phase = .compose
      }
    }
  }

  func answerSpend(_ yes: Bool) { answer(step: "spend", yes: yes, correction: nil) }
  func confirmReadBack() { answer(step: "read_back", yes: true, correction: nil) }
  func rejectReadBack() { answer(step: "read_back", yes: false, correction: nil) }

  /// A "no" that carries the fix: the server compiles the skill again with it.
  func sendCorrection() {
    let text = correction.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !text.isEmpty else { return }
    answer(step: "read_back", yes: false, correction: text)
    correction = ""
  }

  func startOver() {
    guard canStartOver else { return }
    captureTask?.cancel()
    pollTask?.cancel()
    phase = .compose
    status = nil
    activeSkill = nil
    frames = []
    frameCount = 0
    clearRecording()
    typedRequest = ""
    correction = ""
    errorMessage = nil
  }

  private func answer(step: String, yes: Bool, correction: String?) {
    guard let skill = activeSkill, let backend = app?.backend, !isAnswering else { return }
    isAnswering = true
    let answer = TeachAnswer(skill: skill, step: step, yes: yes, correction: correction)
    Task {
      do {
        status = try await backend.answerTeach(answer)
        errorMessage = nil
      } catch {
        errorMessage = error.localizedDescription
      }
      isAnswering = false
    }
  }

  /// Polls GET /teach/status once a second until the run ends. A failed poll (a Wi-Fi
  /// blip) is shown and the polling goes on.
  private func startPolling(skill: String) {
    pollTask?.cancel()
    pollTask = Task {
      while !Task.isCancelled {
        try? await Task.sleep(for: .seconds(1))
        guard !Task.isCancelled, let backend = app?.backend else { return }
        do {
          let latest = try await backend.teachStatus(skill: skill)
          guard !Task.isCancelled else { return }
          status = latest
          errorMessage = nil
          if latest.isFinished { return }
        } catch {
          errorMessage = error.localizedDescription
        }
      }
    }
  }
}
