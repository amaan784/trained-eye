//
// SkillsView.swift
//
// The taught skills from GET /skills, as rows rather than cards (DESIGN.md): the skill as
// a sentence, its state in plain words, and a switch for watching. Changing a switch while
// watching says hello again.
//

import SwiftUI

struct SkillsView: View {
  let app: AppModel

  var body: some View {
    Screen(title: "Skills") {
      if let error = app.skillsError {
        Notice(text: error, isError: true)
      } else if app.skills.isEmpty {
        Notice(text: "No skills yet. Teach one, or check the Mac's address in Connect.")
      }
      ModelLine(app: app)
      VStack(spacing: 0) {
        ForEach(app.skills) { skill in
          SkillRow(app: app, skill: skill)
        }
      }
      Notice(
        text: "Switched-on skills are watched for. A skill is ready once its model is "
          + "trained, served from the Mac (te serve-student) and calibrated (te calibrate).")
      Button("Refresh") { Task { await app.refreshSkills() } }
        .buttonStyle(SecondaryButtonStyle())
    }
    .refreshable { await app.refreshSkills() }
    .task {
      await app.refreshSkills()
      if !app.settings.watchedSkills.isEmpty { app.wakeModel() }
    }
  }
}

/// The model on Modal: asleep, waking (the time so far against the usual wait), ready, or
/// what to check. Opening Skills with a skill switched on, or switching one on, wakes it.
private struct ModelLine: View {
  let app: AppModel

  var body: some View {
    TimelineView(.periodic(from: .now, by: 1)) { context in
      VStack(alignment: .leading, spacing: 6) {
        Text(text(now: context.date)).font(Theme.caption).foregroundStyle(Theme.secondary)
        if let status = app.modelStatus, status.state == "waking" {
          ThinBar(fraction: min(0.95, elapsed(status, now: context.date)
            / max(1, status.expected ?? 180)))
        }
      }
    }
    .accessibilityElement(children: .combine)
  }

  private func elapsed(_ status: ModelStatus, now: Date) -> Double {
    (status.elapsed ?? 0) + now.timeIntervalSince(app.modelStatusAt)
  }

  private func text(now: Date) -> String {
    guard let status = app.modelStatus else {
      return "The model on Modal wakes when you switch a skill on or start watching."
    }
    switch status.state {
    case "ready":
      return "The model on Modal is awake and ready."
    case "waking":
      let seconds = Int(elapsed(status, now: now))
      let minutes = Int(((status.expected ?? 180) / 60).rounded())
      return String(format: "Waking the model on Modal: %d:%02d so far, usually about %d "
        + "minutes the first time.", seconds / 60, seconds % 60, minutes)
    case "failed":
      return "The model on Modal did not answer. On the Mac, check te serve-student."
    case "unknown":
      return "The Mac has no model endpoint set (TE_STUDENT_URL), so watching cannot score."
    default:
      return "The model on Modal is asleep; it wakes when you switch a skill on."
    }
  }
}

private struct SkillRow: View {
  let app: AppModel
  let skill: SkillInfo

  var body: some View {
    VStack(spacing: 0) {
      Rectangle().fill(Theme.rule).frame(height: 1)
      Toggle(isOn: isOn) {
        VStack(alignment: .leading, spacing: 4) {
          Text(sentence).font(Theme.headline)
          Text(state).font(Theme.caption).foregroundStyle(Theme.secondary)
        }
      }
      .toggleStyle(InkSwitchStyle())
      .padding(.vertical, 14)
    }
    .accessibilityElement(children: .combine)
  }

  /// The skill as the user would say it: its name in words, and what it says.
  private var sentence: String {
    let words = skill.name.replacingOccurrences(of: "_", with: " ")
    return words.prefix(1).uppercased() + words.dropFirst()
  }

  /// Its state in plain words.
  private var state: String {
    var parts: [String] = []
    if let training = skill.training, training.state == .running {
      parts.append(training.started.map { "Training, started \(TrainingRun.display($0))" }
        ?? "Training")
    } else if skill.isReady {
      parts.append("Trained")
    } else if skill.served == true {
      parts.append("Trained; calibrate it on the Mac before watching")
    } else if let training = skill.training, training.state == .done {
      parts.append("Trained; serve and calibrate it on the Mac before watching")
    } else if let training = skill.training, training.state == .failed {
      parts.append("Training failed; see te train on the Mac")
    } else if skill.dataset {
      parts.append("Taught; train it on the Mac")
    } else {
      parts.append("Taught; its examples are not built yet")
    }
    if let hold = skill.holdSeconds, hold > 0 { parts.append("speaks after \(hold) s") }
    return parts.joined(separator: ", ")
  }

  private var isOn: Binding<Bool> {
    Binding(
      get: { app.settings.watchedSkills.contains(skill.name) },
      set: { on in
        if on {
          app.settings.watchedSkills.insert(skill.name)
          app.wakeModel()
        } else {
          app.settings.watchedSkills.remove(skill.name)
        }
        app.watchedSkillsChanged()
      })
  }
}
