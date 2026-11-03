//
// Speaker.swift
//
// Speaks alerts on the glasses. DAT has no speaker API: the glasses are an ordinary
// Bluetooth audio output, so, as SystemSpokenOutput.swift in Meta's BirdSpotter sample
// does, a line is spoken only when the current output route is Bluetooth and dropped
// otherwise, never played from the iPhone's speaker. The app never listens while it speaks:
// holding to talk silences the speaker first, and no alert is spoken while the teach
// request is being recorded (the alert still shows on screen).
//

import AVFoundation
import Foundation

final class Speaker: NSObject, AVSpeechSynthesizerDelegate, @unchecked Sendable {
  private let synthesizer = AVSpeechSynthesizer()
  private let lock = NSLock()
  private var linesInFlight = 0
  private var isListening = false

  /// True from the moment a line is handed over until its last word.
  var isSpeaking: Bool { lock.withLock { linesInFlight > 0 } }

  override init() {
    super.init()
    synthesizer.delegate = self
  }

  /// Says `text` if a Bluetooth output is connected and nothing is being recorded.
  /// Returns whether it was spoken. Lines queue rather than interrupt each other.
  @discardableResult
  func speak(_ text: String) -> Bool {
    let line = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !line.isEmpty, Self.outputIsBluetooth, !lock.withLock({ isListening }) else {
      return false
    }
    // As BirdSpotter: move a session still on the system default (which the silent
    // switch mutes) to playback; leave a category something else chose.
    let session = AVAudioSession.sharedInstance()
    if session.category == .soloAmbient {
      try? session.setCategory(.playback, mode: .spokenAudio)
    }
    let utterance = AVSpeechUtterance(string: line)
    utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
    lock.withLock { linesInFlight += 1 }
    synthesizer.speak(utterance)
    return true
  }

  /// Cuts the current line short and drops the queued ones.
  func silence() {
    synthesizer.stopSpeaking(at: .immediate)
  }

  /// Marks the microphone as in use (or not). Starting to listen silences the speaker.
  func setListening(_ listening: Bool) {
    lock.withLock { isListening = listening }
    if listening { silence() }
  }

  /// Whether a line said now would reach an ear (the glasses or another Bluetooth output).
  static var outputIsBluetooth: Bool {
    AVAudioSession.sharedInstance().currentRoute.outputs.contains { output in
      switch output.portType {
      case .bluetoothA2DP, .bluetoothHFP, .bluetoothLE: return true
      default: return false
      }
    }
  }

  func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
    finishLine()
  }

  func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
    finishLine()
  }

  private func finishLine() {
    lock.withLock { linesInFlight = max(0, linesInFlight - 1) }
  }
}
