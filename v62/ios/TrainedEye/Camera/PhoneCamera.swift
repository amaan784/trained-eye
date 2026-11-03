//
// PhoneCamera.swift
//
// The iPhone's back camera, used when the glasses are unavailable. Built like
// PhoneCameraPreviewSource.swift in Meta's BirdSpotter sample: one AVCaptureSession on its
// own queue, BGRA frames turned upright (a quarter turn), and only the frames that are due
// converted. iOS stops the camera while the app is in the background, so this source only
// watches with the app open; the green camera dot iOS shows stays on while it runs.
//

import AVFoundation
import CoreImage
import Foundation
import Observation
import UIKit

@MainActor
@Observable
final class PhoneCamera {
  private(set) var isRunning = false
  var errorMessage: String?

  @ObservationIgnored private let capture: PhoneCapture

  init(throttle: FrameThrottle) {
    capture = PhoneCapture(throttle: throttle)
  }

  /// Asks for camera access the first time, then starts the camera. False if it could not.
  func start() async -> Bool {
    guard await AVCaptureDevice.requestAccess(for: .video) else {
      errorMessage = "Camera access is off. Turn it on in Settings, Trained Eye."
      return false
    }
    let started: Bool = await withCheckedContinuation { continuation in
      capture.start { ok in continuation.resume(returning: ok) }
    }
    errorMessage = started ? nil : "The iPhone camera could not start."
    isRunning = started
    return started
  }

  func stop() {
    capture.stop()
    isRunning = false
  }
}

/// The capture graph, confined to its own queue.
final class PhoneCapture: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate,
  @unchecked Sendable
{
  private let queue = DispatchQueue(label: "trained-eye.phone-camera")
  private let session = AVCaptureSession()
  private let context = CIContext()
  private let throttle: FrameThrottle
  private var isConfigured = false

  /// The back camera's sensor sits a quarter turn from portrait.
  private let portraitRotationAngle: CGFloat = 90

  init(throttle: FrameThrottle) {
    self.throttle = throttle
    super.init()
  }

  func start(completion: @escaping @Sendable (Bool) -> Void) {
    queue.async { [self] in
      guard isConfigured || configure() else {
        completion(false)
        return
      }
      if !session.isRunning { session.startRunning() }
      completion(true)
    }
  }

  func stop() {
    queue.async { [self] in
      if session.isRunning { session.stopRunning() }
    }
  }

  private func configure() -> Bool {
    guard
      let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
      let input = try? AVCaptureDeviceInput(device: device)
    else { return false }

    session.beginConfiguration()
    defer { session.commitConfiguration() }
    // 640x480, close to the glasses' low resolution: the backend pads frames to 512x512.
    session.sessionPreset = .vga640x480
    guard session.canAddInput(input) else { return false }
    session.addInput(input)

    let output = AVCaptureVideoDataOutput()
    output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
    output.alwaysDiscardsLateVideoFrames = true
    output.setSampleBufferDelegate(self, queue: queue)
    guard session.canAddOutput(output) else { return false }
    session.addOutput(output)

    if let connection = output.connection(with: .video),
      connection.isVideoRotationAngleSupported(portraitRotationAngle)
    {
      connection.videoRotationAngle = portraitRotationAngle
    }
    isConfigured = true
    return true
  }

  func captureOutput(
    _ output: AVCaptureOutput,
    didOutput sampleBuffer: CMSampleBuffer,
    from connection: AVCaptureConnection
  ) {
    guard throttle.isDue(), let pixelBuffer = sampleBuffer.imageBuffer else { return }
    let image = CIImage(cvPixelBuffer: pixelBuffer)
    guard let cgImage = context.createCGImage(image, from: image.extent) else { return }
    throttle.deliver(UIImage(cgImage: cgImage))
  }
}
