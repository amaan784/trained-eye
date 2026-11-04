//
// TeachView.swift
//
// Teaching a new skill, as numbered steps (a real sequence, DESIGN.md): say what to watch
// for (hold to talk, or type it), look at it for 10 seconds with the countdown shown as a
// strip of film filling frame by frame, name it and send it. Then the Mac's two questions as
// sentences: the cost ("Teach it" or "Cancel") and the read-back ("That's right" or "Fix
// it"), and the run's progress. A session picked up again after a relaunch shows the same
// way. The glasses stream 504x896 for the capture; the SDK lowers it when Bluetooth is short.
//

import SwiftUI

struct TeachView: View {
  let app: AppModel
  @Bindable var flow: TeachFlow

  var body: some View {
    Screen(title: "Teach") {
      if flow.phase == .onServer, let status = flow.status {
        ServerSteps(flow: flow, status: status)
      } else {
        composeSteps
      }
      if let error = flow.errorMessage { Notice(text: error, isError: true) }
      if flow.canStartOver, flow.status != nil || flow.frameCount > 0 || flow.hasRequest {
        Button("Start over") { flow.startOver() }
          .buttonStyle(SecondaryButtonStyle())
      }
    }
  }

  @ViewBuilder private var composeSteps: some View {
    Step(number: 1, title: "Say what to watch for") {
      Text(flow.isRecording ? "Listening. Let go to stop." : "Hold to talk")
        .font(Theme.headline)
        .foregroundStyle(flow.isRecording ? Theme.onText : Theme.text)
        .frame(maxWidth: .infinity, minHeight: 64)
        .background(flow.isRecording ? Theme.text : Color.clear)
        .overlay(Rectangle().stroke(Theme.text, lineWidth: 1.5))
        // Pressing stays true until the finger lifts or strays, or after 30 seconds.
        .onLongPressGesture(
          minimumDuration: 30, maximumDistance: 80, perform: {},
          onPressingChanged: { pressing in
            if pressing { flow.pressToTalk() } else { flow.releaseToTalk() }
          })
        .accessibilityAddTraits(.isButton)
      if let seconds = flow.requestSeconds {
        HStack {
          Notice(text: String(format: "Recorded %.1f s on the %@ mic.", seconds,
                              flow.requestMicrophone))
          Spacer()
          Button("Clear") { flow.clearRecording() }.buttonStyle(SecondaryButtonStyle())
        }
      } else {
        TextField("Or type it: tell me if I leave the burner on", text: $flow.typedRequest)
          .underlinedField()
      }
      Notice(text: "Objects, displays, signs, text and vehicles only, never a person.")
    }

    Step(number: 2, title: "Look at it for 10 seconds") {
      FramePreview(jpeg: app.lastFrame)
        .padding(.horizontal, -Theme.gutter)
      FilmStrip(frames: captureFrames, frameWidth: 28)
        .padding(.horizontal, -Theme.gutter)
      Text(captureText).font(Theme.title)
      Button("Start looking") { flow.startCapture() }
        .buttonStyle(SecondaryButtonStyle())
        .disabled(flow.phase != .compose || app.isWatching)
      Notice(
        text: app.isWatching
          ? "Stop watching first: one camera job at a time."
          : "Point the camera at it and move a little. The frames stay on the phone until "
            + "you send them.")
    }

    Step(number: 3, title: "Name it and send it") {
      TextField("burner_left_on", text: $flow.skillName)
        .textInputAutocapitalization(.never)
        .autocorrectionDisabled()
        .font(Theme.data)
        .underlinedField()
      Notice(text: "Lower-case letters, digits and underscores, starting with a letter.")
      Button(flow.phase == .uploading ? "Sending" : "Send to the Mac") { flow.send() }
        .buttonStyle(PrimaryButtonStyle())
        .disabled(!flow.canSend)
    }
  }

  /// Ten frames of film, one a second, filling as the capture goes.
  private var captureFrames: [FilmFrame] {
    (0..<TeachFlow.captureSeconds).map { i in
      FilmFrame(id: i, jpeg: i < flow.strip.count ? flow.strip[i] : nil,
                edge: i < flow.strip.count ? "0:\(String(format: "%02d", i + 1))" : "")
    }
  }

  private var captureText: String {
    switch flow.phase {
    case .startingCamera: return "Starting the camera"
    case .countdown(let n): return "Starting in \(n)"
    case .capturing(let left): return left == 1 ? "1 second left" : "\(left) seconds left"
    default: return flow.frameCount > 0 ? "\(flow.frameCount) frames, ready to send" : "Ready"
    }
  }
}

/// A numbered step: the numeral in Marker, Display width; the step a sentence.
private struct Step<Content: View>: View {
  let number: Int
  let title: String
  @ViewBuilder let content: Content

  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      HStack(alignment: .firstTextBaseline, spacing: 12) {
        Text("\(number)").font(Theme.display).foregroundStyle(Theme.marker)
        Text(title).font(Theme.title)
      }
      content
    }
    .frame(maxWidth: .infinity, alignment: .leading)
    .padding(.top, 10)
  }
}

/// The run on the Mac: its two questions as sentences, its progress and how it ended.
private struct ServerSteps: View {
  @Bindable var flow: TeachFlow
  let status: TeachStatus
  @State private var fixing = false

  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      Text("Teaching \(status.skill ?? flow.activeSkill ?? "it") on the Mac")
        .font(Theme.caption).foregroundStyle(Theme.secondary)
      if status.asksSpend {
        Text(spendText).font(Theme.title)
        if let warning = status.warning { Notice(text: warning) }
        Button("Teach it") { flow.answerSpend(true) }.buttonStyle(PrimaryButtonStyle())
          .disabled(flow.isAnswering)
        Button("Cancel") { flow.answerSpend(false) }.buttonStyle(SecondaryButtonStyle())
          .disabled(flow.isAnswering)
      }
      if status.asksReadBack, let readBack = status.readBack {
        Text(readBack).font(Theme.title)
        Button("That's right") { flow.confirmReadBack() }.buttonStyle(PrimaryButtonStyle())
          .disabled(flow.isAnswering)
        if fixing {
          TextField("Say what to change", text: $flow.correction).underlinedField()
          Button("Send the fix") { flow.sendCorrection() }.buttonStyle(SecondaryButtonStyle())
            .disabled(flow.isAnswering
              || flow.correction.trimmingCharacters(in: .whitespaces).isEmpty)
        } else {
          Button("Fix it") { fixing = true }.buttonStyle(SecondaryButtonStyle())
        }
        Button("Stop teaching") { flow.rejectReadBack() }.buttonStyle(SecondaryButtonStyle())
          .disabled(flow.isAnswering)
      }
      Text(stateText).font(Theme.headline)
      ForEach(Array(status.lines.suffix(8).enumerated()), id: \.offset) { _, line in
        Text(line).font(Theme.caption).foregroundStyle(Theme.secondary)
      }
      if let stopped = status.stopped { Notice(text: "Stopped: \(stopped)") }
      if let error = status.error { Notice(text: error, isError: true) }
      if status.state == .done, let summary = status.summary {
        Text("It made \(summary.examples ?? 0) examples, \(summary.yes ?? 0) yes and "
          + "\(summary.no ?? 0) no. Next, on the Mac: te train, then te calibrate.")
      }
      if let spent = status.spentUSD {
        Notice(text: String(format: "It spent $%.2f on Meta API calls.", spent))
      }
    }
    .padding(.top, 6)
  }

  private var stateText: String {
    let words = status.state.rawValue.replacingOccurrences(of: "_", with: " ")
    return words.prefix(1).uppercased() + words.dropFirst()
  }

  private var spendText: String {
    let total = status.estimateTotal.map { String(format: "$%.2f", $0) } ?? "an unknown amount"
    let most = status.estimateAtMost.map { String(format: ", at most $%.2f", $0) } ?? ""
    return "This costs about \(total)\(most). Teach it?"
  }
}
