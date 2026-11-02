//
// Components.swift
//
// The few building blocks every screen uses: a screen with its title, a flat card, the
// two button styles, a status dot, a label and value row, and the frame preview.
//

import SwiftUI
import UIKit

/// A scrolling screen with a navigation title on the dark ground.
struct Screen<Content: View>: View {
  let title: String
  @ViewBuilder let content: Content

  var body: some View {
    NavigationStack {
      ScrollView {
        VStack(alignment: .leading, spacing: Theme.gap) { content }
          .padding(Theme.gutter)
      }
      .background(Theme.ground)
      .navigationTitle(title)
      .toolbarBackground(Theme.ground, for: .navigationBar)
      .toolbarBackground(.visible, for: .navigationBar)
      .toolbarBackground(Theme.surface, for: .tabBar)
      .toolbarBackground(.visible, for: .tabBar)
    }
  }
}

/// A flat raised surface with a hairline border and an optional small heading.
struct Card<Content: View>: View {
  var title: String? = nil
  @ViewBuilder let content: Content

  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      if let title {
        Text(title.uppercased())
          .font(Theme.label)
          .tracking(0.8)
          .foregroundStyle(Theme.secondary)
      }
      content
    }
    .frame(maxWidth: .infinity, alignment: .leading)
    .padding(Theme.gutter)
    .background(Theme.surface, in: RoundedRectangle(cornerRadius: Theme.corner))
    .overlay(RoundedRectangle(cornerRadius: Theme.corner).stroke(Theme.rule, lineWidth: 1))
  }
}

/// The main action: filled with the accent.
struct PrimaryButtonStyle: ButtonStyle {
  func makeBody(configuration: Configuration) -> some View {
    StyledLabel(configuration: configuration, filled: true)
  }
}

/// Any other action: outlined.
struct SecondaryButtonStyle: ButtonStyle {
  func makeBody(configuration: Configuration) -> some View {
    StyledLabel(configuration: configuration, filled: false)
  }
}

private struct StyledLabel: View {
  let configuration: ButtonStyleConfiguration
  let filled: Bool
  @Environment(\.isEnabled) private var isEnabled

  var body: some View {
    configuration.label
      .font(Theme.headline)
      .foregroundStyle(filled ? Theme.ground : Theme.text)
      .frame(maxWidth: .infinity, minHeight: 48)
      .background(
        filled ? Theme.accent : Color.clear, in: RoundedRectangle(cornerRadius: Theme.corner)
      )
      .overlay(
        RoundedRectangle(cornerRadius: Theme.corner).stroke(filled ? Color.clear : Theme.rule)
      )
      .opacity(isEnabled ? (configuration.isPressed ? 0.7 : 1) : 0.35)
  }
}

struct StatusDot: View {
  let on: Bool

  var body: some View {
    Circle()
      .fill(on ? Theme.accent : Theme.secondary.opacity(0.5))
      .frame(width: 9, height: 9)
  }
}

/// A label on the left and its value on the right.
struct InfoRow: View {
  let label: String
  let value: String

  var body: some View {
    HStack(alignment: .firstTextBaseline) {
      Text(label).foregroundStyle(Theme.secondary)
      Spacer(minLength: 12)
      Text(value).font(Theme.data).multilineTextAlignment(.trailing)
    }
  }
}

/// A small grey line of explanation, or of an error when `isError` is set.
struct Notice: View {
  let text: String
  var isError = false

  var body: some View {
    Label(text, systemImage: isError ? "exclamationmark.circle" : "info.circle")
      .font(Theme.caption)
      .foregroundStyle(Theme.secondary)
  }
}

/// The newest frame, as the backend receives it.
struct FramePreview: View {
  let jpeg: Data?

  var body: some View {
    ZStack {
      Theme.surface
      if let jpeg, let image = UIImage(data: jpeg) {
        Image(uiImage: image).resizable().scaledToFit()
      } else {
        Text("No frame yet").font(Theme.caption).foregroundStyle(Theme.secondary)
      }
    }
    .frame(maxWidth: .infinity)
    .frame(height: 240)
    .clipShape(RoundedRectangle(cornerRadius: Theme.corner))
  }
}
