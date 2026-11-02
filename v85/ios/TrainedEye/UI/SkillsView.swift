//
// SkillsView.swift
//
// The taught skills from GET /skills, each with an on/off switch for watch mode and its
// status (dataset built, calibrated). Changing a switch while watching says hello again.
//

import SwiftUI

struct SkillsView: View {
  let app: AppModel

  var body: some View {
    Screen(title: "Skills") {
      if let error = app.skillsError {
        Notice(text: error, isError: true)
      } else if app.skills.isEmpty {
        Notice(text: "No taught skills yet. Teach one, or check the backend address.")
      }
      ForEach(app.skills) { skill in
        SkillRow(app: app, skill: skill)
      }
      Notice(
        text: "Switched-on skills are scored while watching. That needs the trained model's "
          + "endpoint set on the Mac (TE_STUDENT_URL).")
      Button("Refresh") { Task { await app.refreshSkills() } }
        .buttonStyle(SecondaryButtonStyle())
    }
    .refreshable { await app.refreshSkills() }
    .task { await app.refreshSkills() }
  }
}

private struct SkillRow: View {
  let app: AppModel
  let skill: SkillInfo

  var body: some View {
    Card {
      Toggle(isOn: isOn) {
        VStack(alignment: .leading, spacing: 4) {
          Text(skill.name).font(Theme.headline)
          if let alert = skill.alert {
            Text("Says: \(alert)").font(Theme.caption).foregroundStyle(Theme.secondary)
          }
          Text(skill.statusText)
            .font(Theme.caption)
            .foregroundStyle(skill.calibrated ? Theme.accent : Theme.secondary)
        }
      }
      if let cooldown = skill.cooldownSeconds, cooldown > 0 {
        InfoRow(label: "Quiet after an alert", value: "\(max(1, cooldown / 60)) min")
      }
    }
  }

  private var isOn: Binding<Bool> {
    Binding(
      get: { app.settings.watchedSkills.contains(skill.name) },
      set: { on in
        if on {
          app.settings.watchedSkills.insert(skill.name)
        } else {
          app.settings.watchedSkills.remove(skill.name)
        }
        app.watchedSkillsChanged()
      })
  }
}
