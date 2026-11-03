//
// RequestRecorder.swift
//
// Records the spoken teach request ("tell me when the washer is done") as 16 kHz mono
// 16-bit WAV: the backend reads a WAV file's length for its cost estimate. Set up the way
// AudioInputHandler.swift in Meta's CameraAccess sample records the glasses' microphone
// over Bluetooth HFP: a playAndRecord session in videoRecording mode with allowBluetoothHFP
// and mixWithOthers, the glasses' HFP input preferred, and an AVAudioEngine tap on the
// input node in the input's own format (HFP gives 8 or 16 kHz), set up again when the
// route gains or loses a device, when an interruption ends, or when the engine's
// configuration changes. Unlike CameraAccess, the iPhone's microphone records when the
// glasses' is not the input, and the Teach screen says which one did: the route is read at
// the start, on every route change and once more at the stop (setPreferredInput can switch
// it after it returns). Each tap buffer is converted to 16 kHz as BirdSpotter's
// MicrophoneCapture.swift does, and the WAV is written in memory with the 44-byte header of
// its WavCodec.swift. Nothing goes to disk.
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

@MainActor
final class RequestRecorder {
  /// The two microphones, as the Teach screen names them.
  enum Microphone: String {
    case glasses
    case phone = "iPhone"
  }

  private var engine: AVAudioEngine?
  private var samples: RequestSamples?
  private var observers: [NSObjectProtocol] = []

  var isRecording: Bool { engine != nil }

  /// True while the glasses' microphone is the input.
  private(set) var usesGlassesMicrophone = false
  /// The microphones that recorded the latest request, in order: two when the route
  /// changed part way (the glasses dropped, or came back).
  private(set) var microphones: [Microphone] = []

  /// "glasses", "iPhone", or "glasses, then iPhone" after a route change.
  var microphoneText: String {
    guard !microphones.isEmpty else { return Microphone.phone.rawValue }
    return microphones.map(\.rawValue).joined(separator: ", then ")
  }

  func start() async throws {
    guard !isRecording else { return }
    guard await AVAudioApplication.requestRecordPermission() else {
      throw RecorderError.microphoneDenied
    }
    guard !isRecording else { return }  // a second start while the first one waited
    do {
      try Self.configureSession()
      guard let samples = RequestSamples() else { throw RecorderError.couldNotRecord }
      let engine = AVAudioEngine()
      guard samples.install(on: engine.inputNode) else { throw RecorderError.couldNotRecord }
      engine.prepare()
      do {
        try engine.start()
      } catch {
        engine.inputNode.removeTap(onBus: 0)
        throw error
      }
      self.engine = engine
      self.samples = samples
    } catch {
      Self.releaseSession()
      throw error
    }
    microphones = []
    noteRoute()
    observeChanges()
  }

  /// Stops and returns the WAV and its length in seconds; nil for a recording under half
  /// a second (a tap rather than a hold).
  func stop() -> (wav: Data, seconds: Double)? {
    guard let engine, let samples else { return nil }
    // Once more: setPreferredInput can switch the route after it returns.
    noteRoute()
    for observer in observers {
      NotificationCenter.default.removeObserver(observer)
    }
    observers = []
    engine.stop()
    engine.inputNode.removeTap(onBus: 0)
    self.engine = nil
    self.samples = nil
    Self.releaseSession()
    let recorded = samples.take()
    let seconds = Double(recorded.count) / Double(RequestSamples.sampleRate)
    guard seconds >= 0.5 else { return nil }
    return (RequestSamples.wav(recorded), seconds)
  }

  // MARK: - The audio session

  /// CameraAccess's configureBluetooth(): `.allowBluetoothHFP` is what surfaces the glasses'
  /// HFP input, which is then preferred. If it cannot be, the iPhone's microphone records.
  private static func configureSession() throws {
    let session = AVAudioSession.sharedInstance()
    try session.setCategory(
      .playAndRecord, mode: .videoRecording, options: [.allowBluetoothHFP, .mixWithOthers])
    try session.setActive(true)
    if let glasses = session.availableInputs?.first(where: { $0.portType == .bluetoothHFP }) {
      try? session.setPreferredInput(glasses)
    }
  }

  /// Lets other audio resume, and goes back to playback so alerts reach the glasses over
  /// A2DP again.
  private static func releaseSession() {
    let session = AVAudioSession.sharedInstance()
    try? session.setActive(false, options: .notifyOthersOnDeactivation)
    try? session.setCategory(.playback, mode: .spokenAudio)
  }

  /// Reads which microphone is the input now and adds it to the list when it changed.
  private func noteRoute() {
    usesGlassesMicrophone = AVAudioSession.sharedInstance().currentRoute.inputs.contains {
      $0.portType == .bluetoothHFP
    }
    let current: Microphone = usesGlassesMicrophone ? .glasses : .phone
    if microphones.last != current { microphones.append(current) }
  }

  // MARK: - Route changes and interruptions

  /// As CameraAccess's AudioInputHandler, on the main queue: route changes, interruptions,
  /// and the engine's own configuration change (it stops itself when the input's format
  /// changes under it).
  private func observeChanges() {
    guard let engine else { return }
    let center = NotificationCenter.default
    let session = AVAudioSession.sharedInstance()
    observers = [
      center.addObserver(
        forName: AVAudioSession.routeChangeNotification, object: session, queue: .main
      ) { [weak self] notification in
        guard let value = notification.userInfo?[AVAudioSessionRouteChangeReasonKey] as? UInt,
          let reason = AVAudioSession.RouteChangeReason(rawValue: value)
        else { return }
        Task { @MainActor in self?.routeChanged(reason) }
      },
      center.addObserver(
        forName: AVAudioSession.interruptionNotification, object: session, queue: .main
      ) { [weak self] notification in
        guard let value = notification.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt,
          let type = AVAudioSession.InterruptionType(rawValue: value)
        else { return }
        Task { @MainActor in self?.interruptionChanged(type) }
      },
      center.addObserver(
        forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main
      ) { [weak self] _ in
        Task { @MainActor in self?.restartTap(configuringSession: false) }
      },
    ]
  }

  /// CameraAccess's handleRouteChange: the glasses connecting or dropping sets the input up
  /// again (the glasses' microphone preferred, else the iPhone's); every change is noted.
  private func routeChanged(_ reason: AVAudioSession.RouteChangeReason) {
    guard isRecording else { return }
    switch reason {
    case .newDeviceAvailable, .oldDeviceUnavailable:
      restartTap(configuringSession: true)
    default:
      break
    }
    noteRoute()
  }

  /// A call or Siri stops the engine; when the interruption ends the tap is set up again
  /// and the request goes on, with what was said before kept.
  private func interruptionChanged(_ type: AVAudioSession.InterruptionType) {
    guard isRecording, type == .ended else { return }
    restartTap(configuringSession: true)
    noteRoute()
  }

  /// CameraAccess's attemptAudioReset: stop the engine, remove the tap, set the session up
  /// again if asked, and install the tap in the input's format now. Without a usable input
  /// the engine stays stopped and what was recorded so far is kept.
  private func restartTap(configuringSession: Bool) {
    guard let engine, let samples else { return }
    if engine.isRunning { engine.stop() }
    engine.inputNode.removeTap(onBus: 0)
    if configuringSession { try? Self.configureSession() }
    guard samples.install(on: engine.inputNode) else { return }
    engine.prepare()
    try? engine.start()
  }
}

/// The request's samples at 16 kHz mono, whatever the input gives (HFP 8 or 16 kHz, the
/// iPhone's microphone 48 kHz), gathered on the engine's tap thread. Thread safe: the tap
/// thread appends under the lock; the main actor installs the tap and takes the samples,
/// never while holding the lock (installTap and engine.stop() can call the tap at once).
final class RequestSamples: @unchecked Sendable {
  static let sampleRate = 16_000
  /// CameraAccess's tap buffer size, in frames.
  private static let tapBufferSize: AVAudioFrameCount = 1024
  /// Hold to talk ends after 30 seconds; anything past a minute is dropped.
  private static let maxSamples = sampleRate * 60

  private let lock = NSLock()
  private let targetFormat: AVAudioFormat
  private var converter: AVAudioConverter?
  private var samples: [Float] = []

  init?() {
    guard
      let format = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: Double(Self.sampleRate), channels: 1,
        interleaved: false)
    else { return nil }
    targetFormat = format
  }

  /// Installs the tap on `input` in the input's own format, with a converter from it.
  /// False for a zero format, which installTap would crash on (CameraAccess: a simulator
  /// without an input, or the microphone refused).
  func install(on input: AVAudioInputNode) -> Bool {
    let format = input.inputFormat(forBus: 0)
    guard format.sampleRate > 0, format.channelCount > 0,
      let newConverter = AVAudioConverter(from: format, to: targetFormat)
    else { return false }
    lock.withLock { converter = newConverter }
    input.installTap(
      onBus: 0, bufferSize: Self.tapBufferSize, format: format
    ) { [weak self] buffer, _ in
      self?.append(buffer)
    }
    return true
  }

  /// Everything recorded so far; the tap is removed before this is called.
  func take() -> [Float] {
    lock.withLock {
      let out = samples
      samples = []
      converter = nil
      return out
    }
  }

  /// Converts one tap buffer to 16 kHz mono and keeps it (BirdSpotter's deliver(_:)).
  private func append(_ buffer: AVAudioPCMBuffer) {
    lock.withLock {
      guard let converter, samples.count < Self.maxSamples else { return }
      let ratio = targetFormat.sampleRate / buffer.format.sampleRate
      let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 1
      guard let converted = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity)
      else { return }
      // The converter pulls rather than takes: one buffer in, then .noDataNow to say that
      // is all there is this time round.
      var supplied = false
      var error: NSError?
      converter.convert(to: converted, error: &error) { _, status in
        if supplied {
          status.pointee = .noDataNow
          return nil
        }
        supplied = true
        status.pointee = .haveData
        return buffer
      }
      guard error == nil, let channel = converted.floatChannelData?[0] else { return }
      samples.append(
        contentsOf: UnsafeBufferPointer(start: channel, count: Int(converted.frameLength)))
    }
  }

  /// 16-bit PCM mono WAV at 16 kHz: the 44-byte RIFF header of BirdSpotter's
  /// WavCodec.encode, then each sample clamped to -1...1 and scaled by 32767.
  static func wav(_ samples: [Float]) -> Data {
    let dataBytes = samples.count * 2
    var out = Data()
    out.reserveCapacity(44 + dataBytes)
    out.append(contentsOf: Array("RIFF".utf8))
    out.append(contentsOf: littleEndian(UInt32(36 + dataBytes)))
    out.append(contentsOf: Array("WAVE".utf8))
    out.append(contentsOf: Array("fmt ".utf8))
    out.append(contentsOf: littleEndian(UInt32(16)))  // fmt chunk size
    out.append(contentsOf: littleEndian(UInt16(1)))  // PCM
    out.append(contentsOf: littleEndian(UInt16(1)))  // mono
    out.append(contentsOf: littleEndian(UInt32(sampleRate)))
    out.append(contentsOf: littleEndian(UInt32(sampleRate * 2)))  // bytes per second
    out.append(contentsOf: littleEndian(UInt16(2)))  // block align
    out.append(contentsOf: littleEndian(UInt16(16)))  // bits per sample
    out.append(contentsOf: Array("data".utf8))
    out.append(contentsOf: littleEndian(UInt32(dataBytes)))
    let pcm = samples.map { sample -> Int16 in
      let clamped = sample.isNaN ? 0 : min(max(sample, -1), 1)
      return Int16(clamped * Float(Int16.max)).littleEndian
    }
    pcm.withUnsafeBytes { out.append(contentsOf: $0) }
    return out
  }

  private static func littleEndian<T: FixedWidthInteger>(_ value: T) -> [UInt8] {
    withUnsafeBytes(of: value.littleEndian) { Array($0) }
  }
}
