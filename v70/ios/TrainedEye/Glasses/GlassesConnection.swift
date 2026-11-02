//
// GlassesConnection.swift
//
// Registration with the Meta AI app and the paired glasses: their name, link, battery and
// whether they need a firmware update. Follows WearablesViewModel.swift in Meta's
// CameraAccess sample, with the device-state reading from BirdSpotter's
// DatGlassesSessionRepository.swift.
//

import MWDATCore
import Observation
import SwiftUI

@MainActor
@Observable
final class GlassesConnection {
  var registrationState: RegistrationState
  var devices: [DeviceIdentifier]
  /// The glasses this app reports on: the first connected pair, else the first paired one.
  private(set) var deviceName: String?
  private(set) var isLinked = false
  /// Battery percent, or nil while unknown (DAT reports 0 until the control link is up).
  private(set) var batteryLevel: Int?
  private(set) var requiresFirmwareUpdate = false
  var errorMessage: String?

  @ObservationIgnored private let wearables: WearablesInterface
  @ObservationIgnored private var registrationTask: Task<Void, Never>?
  @ObservationIgnored private var deviceStreamTask: Task<Void, Never>?
  @ObservationIgnored private var deviceTokens: [any AnyListenerToken] = []

  init(wearables: WearablesInterface) {
    self.wearables = wearables
    self.devices = wearables.devices
    self.registrationState = wearables.registrationState

    registrationTask = Task { [weak self] in
      for await registrationState in wearables.registrationStateStream() {
        self?.registrationState = registrationState
      }
    }
    deviceStreamTask = Task { [weak self] in
      for await devices in wearables.devicesStream() {
        self?.update(devices: devices)
      }
    }
  }

  var isRegistered: Bool { registrationState == .registered }

  var registrationText: String {
    switch registrationState {
    case .registered: return "Connected to Meta AI"
    case .registering: return "Waiting for Meta AI"
    case .available: return "Not connected yet"
    case .unavailable: return "Meta AI is not available"
    default: return "Unknown"
    }
  }

  /// Opens the Meta AI app to approve this app; it calls back through the URL scheme.
  func connectGlasses() {
    guard registrationState != .registering else { return }
    Task { @MainActor in
      do {
        try await wearables.startRegistration()
      } catch let error as RegistrationError {
        errorMessage = error.description
      } catch {
        errorMessage = error.localizedDescription
      }
    }
  }

  func disconnectGlasses() {
    Task { @MainActor in
      do {
        try await wearables.startUnregistration()
      } catch let error as UnregistrationError {
        errorMessage = error.description
      } catch {
        errorMessage = error.localizedDescription
      }
    }
  }

  func openFirmwareUpdate() async {
    do {
      try await wearables.openFirmwareUpdate()
    } catch {
      errorMessage = error.localizedDescription
    }
  }

  func openDATGlassesAppUpdate() async {
    do {
      try await wearables.openDATGlassesAppUpdate()
    } catch {
      errorMessage = error.localizedDescription
    }
  }

  /// A new device list: listen to each device's link, compatibility and state (battery),
  /// as BirdSpotter does, and read everything again whenever one of them changes.
  private func update(devices: [DeviceIdentifier]) {
    self.devices = devices
    let oldTokens = deviceTokens
    deviceTokens = []
    Task {
      for token in oldTokens { await token.cancel() }
    }
    for id in devices {
      guard let device = wearables.deviceForIdentifier(id) else { continue }
      deviceTokens.append(device.addLinkStateListener { [weak self] _ in
        Task { @MainActor in self?.refresh() }
      })
      deviceTokens.append(device.addCompatibilityListener { [weak self] _ in
        Task { @MainActor in self?.refresh() }
      })
      deviceTokens.append(device.addDeviceStateListener { [weak self] _ in
        Task { @MainActor in self?.refresh() }
      })
    }
    refresh()
  }

  private func refresh() {
    let found = devices.compactMap { wearables.deviceForIdentifier($0) }
    requiresFirmwareUpdate = found.contains { $0.compatibility() == .deviceUpdateRequired }
    guard let chosen = found.first(where: { $0.linkState == .connected }) ?? found.first else {
      deviceName = nil
      isLinked = false
      batteryLevel = nil
      return
    }
    deviceName = chosen.name.isEmpty ? "Meta glasses" : chosen.name
    isLinked = chosen.linkState == .connected
    batteryLevel = chosen.batteryLevel.flatMap { $0 > 0 ? $0 : nil }
  }
}
