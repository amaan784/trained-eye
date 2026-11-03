//
// TeachView.swift
//
// Teaching a new skill: a name, the request (hold to talk, or type it), a 10 second
// capture with a countdown, then the Mac's estimate to agree to, the read-back to confirm
// or correct, and the run's progress lines. A session picked up again after a relaunch
// shows the same way. The capture card shows the frames' size (504x896 asked from the
// glasses; the SDK lowers it when Bluetooth bandwidth is short).
//

import SwiftUI

struct TeachView: View {
  let app: AppModel
  @Bindable var flow: TeachFlow

  var body: some View {
    Screen(title: "Teach") {
      if flow.phase == .onServer, let status = flow.status {
        ServerCard(flow: flow, status: status)
      } else {
        composeCards
      }
      if let error = flow.errorMessage { Notice(text: error, isError: true) }
      if flow.canStartOver, flow.status != nil || flow.frameCount > 0 || flow.hasRequest {
        Button("Start over") { flow.startOver() }
          .buttonStyle(SecondaryButtonStyle())
      }
    }
  }

  @ViewBuilder private var composeCards: some View {
    Card(title: "1. Name") {
      TextField("washer_done", text: $flow.skillName)
        .textInputAutocapitalization(.never)
        .autocorrectionDisabled()
        .font(Theme.data)
        .padding(10)
        .background(Theme.ground, in: RoundedRectangle(cornerRadius: 8))
      Notice(text: "Lower-case letters, digits and underscores, starting with a letter.")
    }

    Card(title: "2. What to notice") {
      Text(flow.isRecording ? "Listening. Let go to stop." : "Hold to talk")
        .font(Theme.headline)
        .foregroundStyle(flow.isRecording ? Theme.ground : Theme.text)
        .frame(maxWidth: .infinity, minHeight: 64)
        .background(
          flow.isRecording ? Theme.accent : Theme.ground,
          in: RoundedRectangle(cornerRadius: Theme.corner)
        )
        // Pressing stays true until the finger lifts or strays, or after 30 seconds.
        .onLongPressGesture(
          minimumDuration: 30, maximumDistance: 80, perform: {},
          onPressingChanged: { pressing in
            if pressing { flow.pressToTalk() } else { flow.releaseToTalk() }
          })
      if let seconds = flow.requestSeconds {
        HStack {
          Notice(text: String(format: "Recorded %.1f s, %@ mic.", seconds, flow.requestMicrophone))
          Spacer()
          Button("Clear") { flow.clearRecording() }.font(Theme.label)
        }
      } else {
        TextField("Or type it: tell me when the washer is done", text: $flow.typedRequest)
          .padding(10)
          .background(Theme.ground, in: RoundedRectangle(cornerRadius: 8))
      }
      Notice(text: "Objects, displays, signs, text and vehicles only: never a person.")
    }

    Card(title: "3. Show it for 10 seconds") {
      FramePreview(jpeg: app.lastFrame)
      if let size = app.lastFrameSize, app.lastFrame != nil {
        InfoRow(label: "Frame size", value: size)
      }
      Text(captureText).font(Theme.big).frame(maxWidth: .infinity)
      Button("Start the capture") { flow.startCapture() }
        .buttonStyle(PrimaryButtonStyle())
        .disabled(flow.phase != .compose || app.isWatching)
      if app.isWatching {
        Notice(text: "Stop watching first: one capture at a time.")
      } else {
        Notice(text: "Point the camera at the thing, moving a little. Frames stay in memory.")
      }
    }

    Button(flow.phase == .uploading ? "Sending" : "Send to the Mac") { flow.send() }
      .buttonStyle(PrimaryButtonStyle())
      .disabled(!flow.canSend)
  }

  private var captureText: String {
    switch flow.phase {
    case .startingCamera: return "Starting"
    case .countdown(let n): return "Starting in \(n)"
    case .capturing(let left): return "\(left) s"
    default: return flow.frameCount > 0 ? "\(flow.frameCount) frames" : "Ready"
    }
  }
}

/// The run on the Mac: its two questions, its progress and how it ended.
private struct ServerCard: View {
  @Bindable var flow: TeachFlow
  let status: TeachStatus

  var body: some View {
    Card(title: "On the Mac: \(status.skill ?? flow.activeSkill ?? "")") {
      if status.asksSpend {
        Text(spendText).font(Theme.headline)
        if let warning = status.warning { Notice(text: warning) }
        HStack {
          Button("Spend it") { flow.answerSpend(true) }.buttonStyle(PrimaryButtonStyle())
          Button("No") { flow.answerSpend(false) }.buttonStyle(SecondaryButtonStyle())
        }
        .disabled(flow.isAnswering)
      }
      if status.asksReadBack, let readBack = status.readBack {
        Text(readBack).font(Theme.headline)
        Button("Yes, that's it") { flow.confirmReadBack() }
          .buttonStyle(PrimaryButtonStyle())
        TextField("Or say what to change", text: $flow.correction)
          .padding(10)
          .background(Theme.ground, in: RoundedRectangle(cornerRadius: 8))
        HStack {
          Button("Correct it") { flow.sendCorrection() }.buttonStyle(SecondaryButtonStyle())
            .disabled(flow.correction.trimmingCharacters(in: .whitespaces).isEmpty)
          Button("Stop") { flow.rejectReadBack() }.buttonStyle(SecondaryButtonStyle())
        }
        .disabled(flow.isAnswering)
      }
      InfoRow(label: "State", value: status.state.rawValue.replacingOccurrences(of: "_", with: " "))
      ForEach(Array(status.lines.suffix(8).enumerated()), id: \.offset) { _, line in
        Text(line).font(Theme.caption).foregroundStyle(Theme.secondary)
      }
      if let stopped = status.stopped { Notice(text: "Stopped: \(stopped)") }
      if let error = status.error { Notice(text: error, isError: true) }
      if status.state == .done, let summary = status.summary {
        InfoRow(label: "Examples", value: "\(summary.examples ?? 0)")
        InfoRow(label: "Yes and no", value: "\(summary.yes ?? 0) and \(summary.no ?? 0)")
        Notice(text: "Next, on the Mac: te train, then te calibrate.")
      }
      if let spent = status.spentUSD {
        InfoRow(label: "Spent", value: String(format: "$%.2f", spent))
      }
    }
  }

  private var spendText: String {
    let total = status.estimateTotal.map { String(format: "$%.2f", $0) } ?? "an unknown amount"
    let most = status.estimateAtMost.map { String(format: " (at most $%.2f)", $0) } ?? ""
    return "This skill costs about \(total)\(most) in Meta API calls. Spend it?"
  }
}
