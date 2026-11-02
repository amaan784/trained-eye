//
// RequestRecorder.swift
//
// Records the spoken teach request ("tell me when the washer is done") with
// AVAudioRecorder, as 16 kHz mono 16-bit WAV: the backend reads a WAV file's length for
// its cost estimate. The glasses' microphone is preferred over Bluetooth HFP, set up as
// AudioInputHandler.swift in Meta's CameraAccess sample does; without it the iPhone's
// microphone records. The file is read into memory and deleted at once.
//

import AVFoundation
import Foundation

enum RecorderError: LocalizedError {
  case microphoneDenied
  case couldNotRecord

  var errorDescription: String? {
    switch self {
    case .microphoneDenied:
      return "Microphone access is off. Turn it on in Settings, Trained Eye, or type the request."
    case .couldNotRecord:
      return "The recording could not start."
    }
  }
}

// TODO: verify on the glasses that AVAudioRecorder records from the HFP input
// (CameraAccess records the glasses' microphone with AVAudioEngine instead).
@MainActor
final class RequestRecorder {
  private var recorder: AVAudioRecorder?

  var isRecording: Bool { recorder?.isRecording ?? false }

  /// True while the glasses' microphone is the input.
  private(set) var usesGlassesMicrophone = false

  func start() async throws {
    guard await AVAudioApplication.requestRecordPermission() else {
      throw RecorderError.microphoneDenied
    }
    let session = AVAudioSession.sharedInstance()
    try session.setCategory(.playAndRecord, mode: .default, options: [.allowBluetoothHFP])
    try session.setActive(true)
    if let glasses = session.availableInputs?.first(where: { $0.portType == .bluetoothHFP }) {
      try? session.setPreferredInput(glasses)
    }
    usesGlassesMicrophone = session.currentRoute.inputs.contains { $0.portType == .bluetoothHFP }

    let url = FileManager.default.temporaryDirectory.appending(path: "request.wav")
    try? FileManager.default.removeItem(at: url)
    let settings: [String: Any] = [
      AVFormatIDKey: Int(kAudioFormatLinearPCM),
      AVSampleRateKey: 16_000.0,
      AVNumberOfChannelsKey: 1,
      AVLinearPCMBitDepthKey: 16,
      AVLinearPCMIsFloatKey: false,
      AVLinearPCMIsBigEndianKey: false,
    ]
    let recorder = try AVAudioRecorder(url: url, settings: settings)
    guard recorder.record() else { throw RecorderError.couldNotRecord }
    self.recorder = recorder
  }

  /// Stops and returns the WAV and its length in seconds; nil for a recording under half
  /// a second (a tap rather than a hold).
  func stop() -> (wav: Data, seconds: Double)? {
    guard let recorder else { return nil }
    let seconds = recorder.currentTime
    recorder.stop()
    self.recorder = nil
    let session = AVAudioSession.sharedInstance()
    try? session.setActive(false, options: .notifyOthersOnDeactivation)
    // Back to playback, so alerts reach the glasses over A2DP again.
    try? session.setCategory(.playback, mode: .spokenAudio)
    let data = try? Data(contentsOf: recorder.url)
    try? FileManager.default.removeItem(at: recorder.url)
    guard let data, seconds >= 0.5 else { return nil }
    return (data, seconds)
  }
}
