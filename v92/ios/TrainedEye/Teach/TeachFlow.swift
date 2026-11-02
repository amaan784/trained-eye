//
// TeachFlow.swift
//
// The Teach screen's steps: name the skill, hold to talk (or type) the request, a 10
// second capture after a 3 second countdown that keeps three frames a second in memory,
// the upload to /teach/start, then the server's two questions (spend the estimate? is the
// read-back right?) and its progress lines until the run ends. Frames are only held in
// memory: they are dropped once the Mac has them, or on Start over.
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

  // TODO: the capture uses the watch stream's low resolution (360x640). Medium
  // (504x896) may teach better, but needs the camera added again with a new
  // StreamConfiguration. And after a relaunch, a running session could be picked up
  // again from GET /teach/status (without a skill it returns the latest).
  static let captureSeconds = 10
  /// The pace the backend reads a folder of frames at (keyframes.load_frames).
  static let captureInterval: TimeInterval = 1.0 / 3.0

  var skillName = ""
  var typedRequest = ""
  var correction = ""
  private(set) var phase: Phase = .compose
  private(set) var isRecording = false
  private(set) var requestSeconds: Double?
  private(set) var requestFromGlasses = false
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
      requestFromGlasses = recorder.usesGlassesMicrophone
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
