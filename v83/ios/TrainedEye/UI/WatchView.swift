//
// WatchView.swift
//
// Watch mode: start and stop it, see that the camera is on and what the backend sees, the
// last alert (and whether it was spoken on the glasses), each skill's latest score, the
// connection, the glasses' state and battery, and Save clip for the last 10 seconds. While
// the glasses are folded or out of reach it says so (watching goes on when they are back),
// and after a stay in the background (the screen locked) it shows what went out and came
// back acked meanwhile, for the screen-locked check on the glasses.
//

import SwiftUI

struct WatchView: View {
  let app: AppModel

  var body: some View {
    Screen(title: "Watch") {
      Card {
        HStack(spacing: 10) {
          StatusDot(on: app.isWatching && !app.isGlassesInterrupted)
          Text(app.isWatching ? "Watching" : "Not watching").font(Theme.title)
        }
        Text(cameraText).foregroundStyle(Theme.secondary)
        if app.isWatching {
          Button("Stop watching") { app.stopWatching() }
            .buttonStyle(SecondaryButtonStyle())
        } else {
          Button("Start watching") { Task { await app.startWatching() } }
            .buttonStyle(PrimaryButtonStyle())
            .disabled(app.isCameraOn)
        }
        if let message = app.message { Notice(text: message) }
      }

      if app.isCameraOn { FramePreview(jpeg: app.lastFrame) }

      Card(title: "Last alert") {
        if let alert = app.lastAlert {
          Text(alert.text).font(Theme.headline)
          InfoRow(label: "Skill", value: alert.skill)
          InfoRow(label: "At", value: alert.receivedAt.formatted(date: .omitted, time: .standard))
          if let p = alert.probability {
            InfoRow(label: "Probability", value: String(format: "%.2f", p))
          }
          InfoRow(label: "Cloud check", value: cloudText(alert.confirmed))
          InfoRow(
            label: "Spoken", value: alert.spoken ? "On the glasses" : "No, no Bluetooth audio")
        } else {
          Notice(text: "No alert yet.")
        }
      }

      if app.isWatching {
        Card(title: "Scores") {
          if app.socket.watching.isEmpty {
            Notice(text: "The backend is scoring no skill, so frames are only acked.")
          }
          ForEach(app.socket.watching, id: \.self) { name in
            InfoRow(label: name, value: scoreText(name))
          }
        }
      }

      Card(title: "Connection") {
        InfoRow(label: "Backend", value: app.socket.status.label)
        if let session = app.socket.session { InfoRow(label: "Session", value: session) }
        InfoRow(label: "Frames sent", value: "\(app.socket.framesSent)")
        InfoRow(label: "Frames acked", value: "\(app.socket.framesAcked)")
        if let error = app.socket.lastError { Notice(text: error, isError: true) }
      }

      if let stay = app.lastBackgroundStay {
        Card(title: "Last time in the background") {
          InfoRow(label: "For", value: durationText(stay.seconds))
          InfoRow(label: "Frames from the camera", value: "\(stay.framesFromCamera)")
          InfoRow(label: "Frames sent", value: "\(stay.framesSent)")
          InfoRow(label: "Frames acked", value: "\(stay.framesAcked)")
          InfoRow(label: "Repeated frames skipped", value: "\(stay.repeatsSkipped)")
          Notice(
            text: "Times in the background while watching: \(app.backgroundStays). About one "
              + "frame a second should go out and come back acked with the screen locked.")
        }
      }

      Card(title: "Glasses") {
        InfoRow(label: "Camera", value: app.glassesCamera.statusText)
        if let size = app.lastFrameSize, app.source == .glasses || app.source == .mock {
          InfoRow(label: "Frame size", value: size)
        }
        if app.glassesCamera.recoveries > 0 {
          let times = app.glassesCamera.recoveries
          InfoRow(label: "Came back by itself", value: times == 1 ? "Once" : "\(times) times")
        }
        InfoRow(label: "Battery", value: app.glasses.batteryLevel.map { "\($0)%" } ?? "Unknown")
        if let interruption = app.glassesCamera.interruptionText {
          Notice(text: interruption)
        }
        if let error = app.glassesCamera.errorMessage { Notice(text: error, isError: true) }
      }

      Button("Save the last 10 seconds") { app.saveClip() }
        .buttonStyle(SecondaryButtonStyle())
        .disabled(!app.isWatching)
      Notice(
        text: "Frames are not stored unless you save a clip. The app never touches the "
          + "glasses' capture LED, which should stay on while their camera streams.")
    }
  }

  private var cameraText: String {
    switch app.source {
    case .phone:
      return "The iPhone camera is on. Keep the app open: iOS stops it in the background."
    case .glasses, .mock:
      if let interruption = app.glassesCamera.interruptionText { return interruption }
      return "The \(app.source?.label ?? "glasses") camera is on: \(app.glassesCamera.statusText)."
    case nil:
      return "The camera is off."
    }
  }

  private func scoreText(_ name: String) -> String {
    guard let entry = app.socket.scores[name] else { return "Waiting" }
    guard let p = entry else { return "No score (the model failed)" }
    let threshold = app.skills.first { $0.name == name }?.threshold
    return threshold.map { String(format: "%.2f, threshold %.2f", p, $0) }
      ?? String(format: "%.2f", p)
  }

  private func durationText(_ seconds: TimeInterval) -> String {
    let whole = max(0, Int(seconds))
    return whole < 60 ? "\(whole) s" : "\(whole / 60) min \(whole % 60) s"
  }

  private func cloudText(_ confirmed: Bool?) -> String {
    switch confirmed {
    case .some(true): return "Muse Spark agreed"
    case .some(false): return "Muse Spark disagreed"
    case .none: return "None, or it failed"
    }
  }
}
