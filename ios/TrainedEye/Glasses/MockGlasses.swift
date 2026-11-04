//
// MockGlasses.swift
//
// Debug builds only: DAT's Mock Device Kit, for working without the glasses. Pairs the
// "Meta Glasses" model, powers it on, unfolds and dons it, and sets its camera feed before
// any stream starts (the mock crashes without one). The feed is an HEVC video named
// mock_feed.mp4 if the app bundle has one (the simulator has no camera), else the iPhone's
// back camera. Calls as in MockDeviceKitViewModel.swift and MockDeviceCardViewModel.swift
// in Meta's CameraAccess sample. Fold, unfold and a touchpad tap are there to check, in the
// simulator, that watching comes back after a fold and is not restarted while paused.
//

// Resources/mock_feed.mp4 (made by ios/make_mock_feed.py: a drawn gas burner, off and then
// lit with no pan, HEVC tagged hvc1) is the feed in the simulator, which has no camera.

#if DEBUG

import AVFoundation
import Foundation
import MWDATMockDevice
import Observation

@MainActor
@Observable
final class MockGlassesController {
  private(set) var isActive = false
  private(set) var isFolded = false
  var errorMessage: String?

  @ObservationIgnored private let mockDeviceKit: MockDeviceKitInterface = MockDeviceKit.shared
  @ObservationIgnored private var device: MockGlasses?

  func start() async {
    // Decide the feed first: a mock pair that is worn without one would look active to
    // the device selector and stream nothing (BirdSpotter sets the feed right after don()).
    let video = Bundle.main.url(forResource: "mock_feed", withExtension: "mp4")
    if video == nil {
      let granted = await AVCaptureDevice.requestAccess(for: .video)
      guard granted else {
        errorMessage = "Camera access is off, and the app has no mock_feed.mp4 to play."
        return
      }
    }
    if !mockDeviceKit.isEnabled {
      mockDeviceKit.enable()  // starts registered, with permissions granted
    }
    let glasses: MockGlasses
    if let paired = mockDeviceKit.pairedDevices.compactMap({ $0 as? MockGlasses }).first {
      glasses = paired
    } else {
      do {
        glasses = try mockDeviceKit.pairGlasses(model: .metaGlasses)
      } catch {
        errorMessage = error.localizedDescription
        return
      }
    }
    glasses.powerOn()
    glasses.unfold()
    glasses.don()
    device = glasses

    if let video {
      glasses.services.camera.setCameraFeed(fileURL: video)
    } else {
      glasses.services.camera.setCameraFeed(cameraFacing: .back)
    }
    errorMessage = nil
    isFolded = false
    isActive = true
  }

  /// Folding ends the glasses' session (AGENTS.md); unfolding makes them available again,
  /// and puts them back on, as MockDeviceCardViewModel's don() does after an unfold.
  func fold() {
    device?.fold()
    isFolded = true
  }

  func unfold() {
    device?.unfold()
    device?.don()
    isFolded = false
  }

  /// A single tap pauses an active stream; a second tap resumes it.
  func tapTouchpad() {
    device?.services.captouch.tap()
  }

  func stop() {
    if let device {
      mockDeviceKit.unpairDevice(device)
    }
    mockDeviceKit.disable()
    device = nil
    isActive = false
    isFolded = false
  }
}

#endif
