//
// WatchView.swift
//
// Watch mode, the hero screen (DESIGN.md): a film strip of the last five frames the
// backend scored, each edge-printed with its time and score; each skill as a sentence with
// its state; one primary button. When an alert fires the ring draws around the newest
// frame and the spoken line fills the screen in Display type for a few seconds. The
// connection, the glasses' state and battery, and what went out and came back acked during
// a stay in the background (the screen-locked check) are under Details.
//

import SwiftUI

struct WatchView: View {
  let app: AppModel
  @State private var shown: AlertEvent?
  @State private var ringed: Int?

  var body: some View {
    ZStack {
      Screen(title: "Watch") {
        FilmStrip(frames: filmFrames, frameWidth: 68, ringed: ringed)
          .padding(.horizontal, -Theme.gutter)
        Text(stripCaption).font(Theme.caption).foregroundStyle(Theme.secondary)

        VStack(alignment: .leading, spacing: 10) {
          if app.isWatching, let since = app.socket.wakingSince {
            WakingModel(since: since, expected: app.socket.wakeExpected)
          }
          ForEach(skillLines, id: \.self) { line in Text(line).font(Theme.body) }
          ForEach(holdBars, id: \.name) { bar in
            HoldBar(fraction: bar.fraction)
              .accessibilityLabel("\(plain(bar.name)) hold")
              .accessibilityValue("\(Int(bar.fraction * 100)) percent")
          }
        }
        .padding(.top, 4)

        if let alert = app.lastAlert {
          VStack(alignment: .leading, spacing: 4) {
            Text("\u{201C}\(alert.text)\u{201D}").font(Theme.title)
            Text(alertLine(alert)).font(Theme.caption).foregroundStyle(Theme.secondary)
          }
          .padding(.top, 4)
        }

        if app.isWatching {
          Button("Stop watching") { app.stopWatching() }.buttonStyle(PrimaryButtonStyle())
        } else {
          Button("Start watching") { Task { await app.startWatching() } }
            .buttonStyle(PrimaryButtonStyle())
            .disabled(app.isCameraOn)
        }
        if let message = app.message { Notice(text: message, isError: true) }
        Text(cameraText).font(Theme.caption).foregroundStyle(Theme.secondary)

        if app.isWatching {
          Button("Save the last 10 seconds") { app.saveClip() }
            .buttonStyle(SecondaryButtonStyle())
        }

        DisclosureGroup {
          details.padding(.top, 8)
        } label: {
          Text("Details").font(Theme.label)
        }
        .tint(Theme.text)
        .padding(.top, 8)

        Notice(
          text: "Frames are not stored unless you save a clip. The app never touches the "
            + "glasses' capture light, which stays on while their camera streams.")
      }
      if let shown { SpokenLine(alert: shown) { self.shown = nil } }
    }
    .onChange(of: app.lastAlert?.id) {
      guard let alert = app.lastAlert else { return }
      ringed = app.scoredFrames.last?.id
      shown = alert
      Task {
        try? await Task.sleep(for: .seconds(4))
        if shown?.id == alert.id { shown = nil }
      }
    }
  }

  /// Always five frames: the ones scored so far, then empty film.
  private var filmFrames: [FilmFrame] {
    let scored = app.scoredFrames.map { f in
      FilmFrame(id: f.id, jpeg: f.jpeg, edge: edgeText(f.seconds, f.score))
    }
    let empty = (0..<max(0, 5 - scored.count)).map { FilmFrame(id: -1 - $0, jpeg: nil, edge: "") }
    return scored + empty
  }

  private var stripCaption: String {
    if !app.scoredFrames.isEmpty { return "The last frames it scored, with their scores." }
    return app.isWatching ? "Waiting for the first score." : "Start watching to see what it sees."
  }

  /// Each switched-on skill as a sentence with its state.
  private var skillLines: [String] {
    let names = app.isWatching && !app.socket.watching.isEmpty
      ? app.socket.watching : Array(app.settings.watchedSkills).sorted()
    if names.isEmpty { return ["No skill is switched on. Choose one in Skills."] }
    return names.map { "\(plain($0)): \(state($0))" }
  }

  /// For each skill whose condition is on: how far through its hold it is.
  private var holdBars: [(name: String, fraction: Double)] {
    guard app.isWatching, app.socket.status == .live else { return [] }
    return app.socket.watching.compactMap { name in
      guard let heldFor = app.socket.held[name] ?? nil,
        let hold = app.skills.first(where: { $0.name == name })?.holdSeconds, hold > 0
      else { return nil }
      return (name, min(1, heldFor / Double(hold)))
    }
  }

  /// Watching, connecting, or how far the condition is through its hold before it speaks.
  private func state(_ name: String) -> String {
    guard app.isWatching else { return "not watching" }
    switch app.socket.status {
    case .live: break
    case .connecting, .off: return "connecting to the Mac"
    case .retrying: return app.socket.status.label.lowercased()
    }
    if app.socket.modelState == "waking" { return "waiting for the model" }
    if app.socket.modelState == "failed" {
      return "the model did not answer; check te serve-student on the Mac"
    }
    guard let entry = app.socket.scores[name] else { return "watching, waiting for the first score" }
    guard entry != nil else { return "the model did not answer" }
    if let heldFor = app.socket.held[name] ?? nil,
      let hold = app.skills.first(where: { $0.name == name })?.holdSeconds, hold > 0
    {
      let soFar = min(Int(heldFor.rounded()), hold)
      return "on for \(soFar) of \(hold) s, then it speaks"
    }
    return "watching"
  }

  private func plain(_ name: String) -> String {
    let words = name.replacingOccurrences(of: "_", with: " ")
    return words.prefix(1).uppercased() + words.dropFirst()
  }

  private func edgeText(_ seconds: TimeInterval, _ score: Double?) -> String {
    let whole = max(0, Int(seconds))
    let time = String(format: "%d:%02d", whole / 60, whole % 60)
    return score.map { time + String(format: " %.2f", $0) } ?? time
  }

  private func alertLine(_ alert: AlertEvent) -> String {
    let at = alert.receivedAt.formatted(date: .omitted, time: .shortened)
    let spoken = alert.spoken ? "Spoken on the glasses" : "Not spoken: no Bluetooth audio"
    let check: String
    switch alert.confirmed {
    case .some(true): check = ", and Muse Spark agreed"
    case .some(false): check = ", though Muse Spark disagreed"
    case .none: check = ""
    }
    return "\(spoken) at \(at)\(check)."
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

  @ViewBuilder private var details: some View {
    VStack(alignment: .leading, spacing: 8) {
      InfoRow(
        label: "Backend",
        value: app.isWatching ? app.socket.status.label : "Connects when you start watching")
      if let session = app.socket.session { InfoRow(label: "Session", value: session) }
      InfoRow(label: "Frames sent", value: "\(app.socket.framesSent)")
      InfoRow(label: "Frames acked", value: "\(app.socket.framesAcked)")
      if let error = app.socket.lastError { Notice(text: error, isError: true) }
      InfoRow(label: "Glasses camera", value: app.glassesCamera.statusText)
      if let size = app.lastFrameSize, app.source == .glasses || app.source == .mock {
        InfoRow(label: "Frame size", value: size)
      }
      if app.glassesCamera.recoveries > 0 {
        let times = app.glassesCamera.recoveries
        InfoRow(label: "Came back by itself", value: times == 1 ? "Once" : "\(times) times")
      }
      InfoRow(label: "Battery", value: app.glasses.batteryLevel.map { "\($0)%" } ?? "Unknown")
      if let error = app.glassesCamera.errorMessage { Notice(text: error, isError: true) }
      if let stay = app.lastBackgroundStay {
        InfoRow(label: "Last time in the background", value: durationText(stay.seconds))
        InfoRow(label: "Frames from the camera", value: "\(stay.framesFromCamera)")
        InfoRow(label: "Frames sent then", value: "\(stay.framesSent)")
        InfoRow(label: "Frames acked then", value: "\(stay.framesAcked)")
        InfoRow(label: "Repeated frames skipped", value: "\(stay.repeatsSkipped)")
        Notice(
          text: "Times in the background while watching: \(app.backgroundStays). About one "
            + "frame a second should go out and come back acked with the screen locked.")
      }
    }
  }

  private func durationText(_ seconds: TimeInterval) -> String {
    let whole = max(0, Int(seconds))
    return whole < 60 ? "\(whole) s" : "\(whole / 60) min \(whole % 60) s"
  }
}

/// The wait while the backend wakes the model on Modal: the time so far, against the usual
/// wait, on a bar that fills to 95% and holds there until the model answers.
private struct WakingModel: View {
  let since: Date
  let expected: Double

  var body: some View {
    TimelineView(.periodic(from: .now, by: 1)) { context in
      let elapsed = max(0, context.date.timeIntervalSince(since))
      VStack(alignment: .leading, spacing: 6) {
        Text("Waking the model on Modal: \(clock(elapsed)) so far. The first time takes "
          + "about \(Int((expected / 60).rounded())) minutes; frames are scored once it answers.")
          .font(Theme.body)
        HoldBar(fraction: min(0.95, elapsed / max(1, expected)))
      }
    }
    .accessibilityElement(children: .combine)
  }

  private func clock(_ seconds: TimeInterval) -> String {
    let whole = Int(seconds)
    return String(format: "%d:%02d", whole / 60, whole % 60)
  }
}

/// A thin bar filling as the condition holds: Ink on a hairline track.
private struct HoldBar: View {
  let fraction: Double

  var body: some View {
    GeometryReader { geo in
      ZStack(alignment: .leading) {
        Rectangle().fill(Theme.rule)
        Rectangle().fill(Theme.text).frame(width: geo.size.width * fraction)
      }
    }
    .frame(height: 4)
    .animation(.linear(duration: 0.9), value: fraction)
  }
}

/// The spoken line, filling the screen in Display type for a few seconds. Tap to close.
private struct SpokenLine: View {
  let alert: AlertEvent
  let close: () -> Void

  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      Spacer()
      Text("\u{201C}\(alert.text)\u{201D}")
        .font(Theme.displayHuge)
        .tracking(-0.8)
        .minimumScaleFactor(0.5)
      Text(alert.spoken ? "Spoken on the glasses." : "Not spoken: no Bluetooth audio.")
        .font(Theme.caption).foregroundStyle(Theme.secondary)
      Spacer()
    }
    .padding(24)
    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
    .background(Theme.ground)
    .contentShape(Rectangle())
    .onTapGesture(perform: close)
    .transition(.opacity)
    .accessibilityAddTraits(.isButton)
    .accessibilityHint("Closes the alert")
  }
}
