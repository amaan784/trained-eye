//
// ConnectView.swift
//
// Where the app finds the Mac (the address `te serve` prints, on the home Wi-Fi or through
// a tunnel) and the glasses (registration with Meta AI, the paired pair, its battery), and
// which camera to use. Debug builds add the DAT mock glasses.
//

import MWDATCore
import SwiftUI

struct ConnectView: View {
  let app: AppModel
  @State private var healthText: String?
  @State private var isChecking = false

  var body: some View {
    @Bindable var settings = app.settings
    let glasses = app.glasses
    Screen(title: "Connect") {
      Card(title: "Backend") {
        TextField("http://192.168.1.20:8765", text: $settings.backendURL)
          .keyboardType(.URL)
          .textInputAutocapitalization(.never)
          .autocorrectionDisabled()
          .font(Theme.data)
          .padding(10)
          .background(Theme.ground, in: RoundedRectangle(cornerRadius: 8))
        Notice(text: settings.addressKind)
        SecureField("Server token (SERVER_TOKEN)", text: $settings.serverToken)
          .textInputAutocapitalization(.never)
          .autocorrectionDisabled()
          .font(Theme.data)
          .padding(10)
          .background(Theme.ground, in: RoundedRectangle(cornerRadius: 8))
        Notice(text: "Teaching needs the SERVER_TOKEN from the Mac's .env. It is kept in the "
          + "iPhone's keychain.")
        Button(isChecking ? "Checking" : "Check the connection") {
          Task { await checkHealth() }
        }
        .buttonStyle(SecondaryButtonStyle())
        .disabled(isChecking)
        if let healthText { Notice(text: healthText) }
        Notice(text: "On the Mac, run te serve: it prints the address to type here.")
      }

      Card(title: "Glasses") {
        InfoRow(label: "Meta AI", value: glasses.registrationText)
        if let name = glasses.deviceName {
          InfoRow(label: name, value: glasses.isLinked ? "Connected" : "Not connected")
        }
        if let battery = glasses.batteryLevel {
          InfoRow(label: "Battery", value: "\(battery)%")
        }
        InfoRow(label: "Camera", value: app.glassesCamera.statusText)
        if glasses.isRegistered {
          Button("Disconnect my glasses") { glasses.disconnectGlasses() }
            .buttonStyle(SecondaryButtonStyle())
        } else {
          Button(glasses.registrationState == .registering ? "Connecting" : "Connect my glasses") {
            glasses.connectGlasses()
          }
          .buttonStyle(PrimaryButtonStyle())
          .disabled(glasses.registrationState == .registering)
          Notice(text: "Meta AI opens to approve Trained Eye. Developer Mode must be on there.")
        }
        if glasses.requiresFirmwareUpdate {
          Button("Update the glasses") { Task { await glasses.openFirmwareUpdate() } }
            .buttonStyle(SecondaryButtonStyle())
        }
        if app.glassesCamera.needsGlassesAppUpdate {
          Button("Update the app on the glasses") {
            Task { await glasses.openDATGlassesAppUpdate() }
          }
          .buttonStyle(SecondaryButtonStyle())
        }
        if let error = glasses.errorMessage ?? app.glassesCamera.errorMessage {
          Notice(text: error, isError: true)
        }
      }

      Card(title: "Camera") {
        Picker("Camera", selection: $settings.cameraChoice) {
          ForEach(CameraChoice.allCases) { choice in
            Text(choice.label).tag(choice)
          }
        }
        .pickerStyle(.segmented)
        Notice(
          text: "The iPhone camera only watches with the app open; the glasses keep going "
            + "in the background.")
      }

      #if DEBUG
      MockCard(mock: app.mock)
      #endif
    }
  }

  private func checkHealth() async {
    guard let backend = app.backend else {
      healthText = BackendError.noAddress.localizedDescription
      return
    }
    isChecking = true
    defer { isChecking = false }
    do {
      let health = try await backend.health()
      healthText = health.ok
        ? "The backend answered (protocol \(health.protocolVersion))." : "The backend is not ok."
      await app.refreshSkills()
    } catch {
      healthText = error.localizedDescription
    }
  }
}

#if DEBUG
/// Debug builds only: the DAT mock glasses, for the simulator or a phone without glasses.
private struct MockCard: View {
  let mock: MockGlassesController

  var body: some View {
    Card(title: "Mock glasses (Debug)") {
      if mock.isActive {
        Notice(
          text: mock.isFolded
            ? "Mock Meta Glasses paired and on, folded."
            : "Mock Meta Glasses paired, on, unfolded and worn.")
        HStack {
          Button(mock.isFolded ? "Unfold" : "Fold") {
            if mock.isFolded { mock.unfold() } else { mock.fold() }
          }
          .buttonStyle(SecondaryButtonStyle())
          Button("Tap the touchpad") { mock.tapTouchpad() }
            .buttonStyle(SecondaryButtonStyle())
            .disabled(mock.isFolded)
        }
        Button("Remove the mock glasses") { mock.stop() }
          .buttonStyle(SecondaryButtonStyle())
      } else {
        Button("Use mock glasses") { Task { await mock.start() } }
          .buttonStyle(SecondaryButtonStyle())
        Notice(text: "Plays mock_feed.mp4 if the app has one, else the iPhone's back camera.")
      }
      if let error = mock.errorMessage { Notice(text: error, isError: true) }
    }
  }
}
#endif
