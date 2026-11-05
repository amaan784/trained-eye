//
// Components.swift
//
// The pieces every screen uses, in the contact-sheet design: a screen with its title in
// Display type, a section under a hairline, the one Ink button and text buttons, a label
// and value row, a plain notice, underlined fields, the camera frame on Film, a film strip
// with sprocket holes and edge print, and the grease-pencil ring that draws itself.
//

import SwiftUI
import UIKit

/// A scrolling screen: its title in Display type, then the content, left-aligned.
struct Screen<Content: View>: View {
  let title: String
  @ViewBuilder let content: Content

  var body: some View {
    NavigationStack {
      ScrollView {
        VStack(alignment: .leading, spacing: Theme.gap) {
          Text(title).font(Theme.display).tracking(-0.6).padding(.top, 8).padding(.bottom, 6)
          content
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(Theme.gutter)
      }
      .scrollDismissesKeyboard(.interactively)  // a drag scrolls and lowers the keyboard
      .background(Theme.ground)
      .toolbar {
        ToolbarItemGroup(placement: .keyboard) {  // a way out of every field
          Spacer()
          Button("Done") {
            UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder),
                                            to: nil, from: nil, for: nil)
          }
        }
      }
      .toolbar(.hidden, for: .navigationBar)
      .toolbarBackground(Theme.ground, for: .tabBar)
      .toolbarBackground(.visible, for: .tabBar)
    }
  }
}

/// A section: a hairline above, an optional sentence-case heading, then its content.
struct Card<Content: View>: View {
  var title: String? = nil
  @ViewBuilder let content: Content

  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      Rectangle().fill(Theme.rule).frame(height: 1)
      if let title { Text(title).font(Theme.headline) }
      content
    }
    .frame(maxWidth: .infinity, alignment: .leading)
    .padding(.top, 6)
  }
}

/// The one primary action on a screen: Ink fill, page-coloured label.
struct PrimaryButtonStyle: ButtonStyle {
  func makeBody(configuration: Configuration) -> some View {
    StyledLabel(configuration: configuration, filled: true)
  }
}

/// Every other action: a text button.
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
    Group {
      if filled {
        configuration.label
          .font(Theme.headline)
          .foregroundStyle(Theme.onText)
          .frame(maxWidth: .infinity, minHeight: 50)
          .background(Theme.text)
      } else {
        configuration.label
          .font(Theme.label)
          .underline(true, color: Theme.secondary)
          .foregroundStyle(Theme.text)
          .frame(minHeight: 44, alignment: .leading)
      }
    }
    .opacity(isEnabled ? (configuration.isPressed ? 0.6 : 1) : 0.35)
  }
}

/// A switch whose state reads at a glance in light and dark mode: on is a solid Ink (or
/// Paper) track with a knob in the page colour, off an outlined track with a grey knob.
struct InkSwitchStyle: ToggleStyle {
  func makeBody(configuration: Configuration) -> some View {
    Button {
      configuration.isOn.toggle()
    } label: {
      HStack(alignment: .center, spacing: 12) {
        configuration.label
        Spacer(minLength: 12)
        ZStack(alignment: configuration.isOn ? .trailing : .leading) {
          Capsule()
            .fill(configuration.isOn ? Theme.text : Color.clear)
            .overlay(Capsule().stroke(configuration.isOn ? Theme.text : Theme.secondary,
                                      lineWidth: 1.5))
            .frame(width: 52, height: 30)
          Circle()
            .fill(configuration.isOn ? Theme.ground : Theme.secondary)
            .frame(width: 22, height: 22)
            .padding(4)
        }
        .animation(.easeOut(duration: 0.15), value: configuration.isOn)
      }
      .contentShape(Rectangle())
    }
    .buttonStyle(.plain)
    .accessibilityElement(children: .combine)
    .accessibilityValue(configuration.isOn ? "On" : "Off")
    .accessibilityAddTraits(.isToggle)
  }
}

/// A thin bar filling to `fraction`: Ink on a hairline track (a hold, a wait).
struct ThinBar: View {
  let fraction: Double

  var body: some View {
    GeometryReader { geo in
      ZStack(alignment: .leading) {
        Rectangle().fill(Theme.rule)
        Rectangle().fill(Theme.text).frame(width: geo.size.width * max(0, min(1, fraction)))
      }
    }
    .frame(height: 4)
    .animation(.linear(duration: 0.9), value: fraction)
  }
}

/// The camera-on dot: Marker while the camera is live.
struct StatusDot: View {
  let on: Bool

  var body: some View {
    Circle()
      .fill(on ? Theme.marker : Theme.graphite)
      .frame(width: 8, height: 8)
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
    .font(Theme.caption)
  }
}

/// A plain line of explanation; an error is set in Ink and says what to do.
struct Notice: View {
  let text: String
  var isError = false

  var body: some View {
    Text(text)
      .font(isError ? Theme.label : Theme.caption)
      .foregroundStyle(isError ? Theme.text : Theme.secondary)
      .fixedSize(horizontal: false, vertical: true)
  }
}

extension View {
  /// A text field as a line to write on: no box, an Ink rule under it.
  func underlinedField() -> some View {
    padding(.vertical, 8)
      .overlay(alignment: .bottom) { Rectangle().fill(Theme.text).frame(height: 1.5) }
  }
}

/// An underlined field hidden behind dots, like a password, until its eye is tapped: for
/// the Mac's address and the server token, so a screen recording does not show them.
struct RevealableField: View {
  let placeholder: String
  @Binding var text: String
  var keyboard: UIKeyboardType = .default
  @State private var shown = false

  var body: some View {
    HStack(spacing: 12) {
      Group {
        if shown {
          TextField(placeholder, text: $text)
        } else {
          SecureField(placeholder, text: $text)
        }
      }
      .keyboardType(keyboard)
      .textInputAutocapitalization(.never)
      .autocorrectionDisabled()
      .font(Theme.data)
      Button { shown.toggle() } label: {
        Image(systemName: shown ? "eye.slash" : "eye")
          .foregroundStyle(Theme.text)
          .frame(width: 32, height: 32)
          .contentShape(Rectangle())
      }
      .buttonStyle(.plain)
      .accessibilityLabel(shown ? "Hide" : "Show")
    }
    .underlinedField()
  }
}

/// The newest frame, as the backend receives it, on Film.
struct FramePreview: View {
  let jpeg: Data?

  var body: some View {
    ZStack {
      Theme.film
      if let jpeg, let image = UIImage(data: jpeg) {
        Image(uiImage: image).resizable().scaledToFit()
      } else {
        Text("No frame yet").font(Theme.caption).foregroundStyle(Theme.edge)
      }
    }
    .frame(maxWidth: .infinity)
    .frame(height: 240)
  }
}

/// One frame on a strip: a 3:4 picture (or an empty slot) and its edge print.
struct FilmFrame: Identifiable {
  let id: Int
  let jpeg: Data?
  let edge: String
}

/// A strip of film: black, sprocket holes in both rebates, the frames 3:4 with 8 pt of
/// black between them, and the edge print in its own band. `ringed` draws the marker ring
/// around that frame, once.
struct FilmStrip: View {
  let frames: [FilmFrame]
  var frameWidth: CGFloat = 60
  var ringed: Int? = nil

  var body: some View {
    let height = frameWidth * 4 / 3
    VStack(spacing: 0) {
      Holes()
      HStack(spacing: 8) {
        ForEach(frames) { frame in
          VStack(alignment: .leading, spacing: 3) {
            ZStack {
              Color(white: 0.07)
              if let jpeg = frame.jpeg, let image = UIImage(data: jpeg) {
                Image(uiImage: image).resizable().scaledToFill()
              }
            }
            .frame(width: frameWidth, height: height)
            .clipped()
            Text(frame.edge)
              .font(Theme.edgePrint)
              .tracking(1.3)
              .foregroundStyle(Theme.edge)
              .lineLimit(1)
              .frame(width: frameWidth, alignment: .leading)
          }
          .overlay {
            if ringed == frame.id {
              MarkerRing().frame(width: frameWidth * 1.45, height: height * 1.35)
                .offset(y: 6)
            }
          }
          .accessibilityElement(children: .ignore)
          .accessibilityLabel(frame.edge.isEmpty ? "Empty frame" : "Frame at \(frame.edge)")
        }
      }
      .padding(.horizontal, 10)
      .padding(.vertical, 6)
      Holes()
    }
    .background(Theme.film)
  }
}

private struct Holes: View {
  var body: some View {
    GeometryReader { geo in
      let count = max(0, Int((geo.size.width - 12) / 14))
      HStack(spacing: 7) {
        ForEach(0..<count, id: \.self) { _ in
          RoundedRectangle(cornerRadius: 1.5).fill(Color.white.opacity(0.9))
            .frame(width: 7, height: 5)
        }
      }
      .padding(.leading, 6)
      .frame(maxHeight: .infinity)
    }
    .frame(height: 13)
  }
}

/// The grease-pencil ring: a hand-drawn ellipse that overshoots its start, two passes, in
/// Marker. It draws itself once (about 600 ms) when it appears; with Reduce Motion it is
/// simply there.
struct MarkerRing: View {
  @State private var drawn: CGFloat = 0
  @Environment(\.accessibilityReduceMotion) private var reduceMotion

  var body: some View {
    ZStack {
      HandEllipse(seed: 3, overshoot: 0.18, rotation: -7)
        .trim(from: 0, to: drawn)
        .stroke(Theme.marker.opacity(0.95), style: StrokeStyle(lineWidth: 4, lineCap: .round))
      HandEllipse(seed: 11, overshoot: 0.08, rotation: -5)
        .trim(from: 0, to: drawn)
        .stroke(Theme.marker.opacity(0.7), style: StrokeStyle(lineWidth: 2, lineCap: .round))
    }
    .onAppear {
      if reduceMotion {
        drawn = 1
      } else {
        withAnimation(.timingCurve(0.3, 0.6, 0.4, 1, duration: 0.6)) { drawn = 1 }
      }
    }
    .allowsHitTesting(false)
    .accessibilityHidden(true)
  }
}

/// The design kit's hand-drawn ellipse, as a shape: it wobbles, opens up as it comes round
/// again and overshoots its start. A seeded generator keeps it the same every time.
struct HandEllipse: Shape {
  let seed: UInt64
  let overshoot: Double
  let rotation: Double

  func path(in rect: CGRect) -> Path {
    var state = seed
    func next() -> Double {
      state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
      return Double(state >> 11) / Double(1 << 53)
    }
    let rot = rotation * .pi / 180
    let start = (195 + next() * 20) * .pi / 180
    let sweep = 2 * .pi * (1 + overshoot)
    let wobble = (0..<8).map { _ in next() * 2 - 1 }
    let cx = rect.midX, cy = rect.midY, rx = rect.width / 2.2, ry = rect.height / 2.2
    var path = Path()
    for i in 0...90 {
      let a = start + sweep * Double(i) / 90
      var sum = 0.0
      for (k, w) in wobble.enumerated() { sum += w * sin(Double(k + 1) * a + Double(k)) }
      let noise = 1 + 0.035 * sum / 4
      let drift = 1 + 0.06 * Double(i) / 90
      let x0 = rx * noise * drift * cos(a), y0 = ry * noise * drift * sin(a)
      let point = CGPoint(x: cx + x0 * cos(rot) - y0 * sin(rot),
                          y: cy + x0 * sin(rot) + y0 * cos(rot))
      if i == 0 { path.move(to: point) } else { path.addLine(to: point) }
    }
    return path
  }
}
