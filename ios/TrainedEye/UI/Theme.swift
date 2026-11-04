//
// Theme.swift
//
// The contact sheet (DESIGN.md): Paper screens, Film for anything showing the camera,
// Edge for the print on film, Marker only for the ring, its leader, step numbers and the
// camera-on dot, Ink for text and the one primary button, Graphite for the rest. In dark
// mode Film is the background and Paper the text; frames never change.
//
// Type is Archivo (Fonts/, SIL Open Font License), bundled as four static instances cut
// with fontTools' instancer (python -m backend.art.fonts): Display (width 125, weight 800)
// for titles only, Text (100/400) and TextSemibold (100/650) for everything else, and Edge
// (62/500) for times and scores on film. The system font stands in if they are missing.
//

import SwiftUI
import UIKit

enum Theme {
  // The tokens.
  static let paperUI = UIColor(red: 0xF4 / 255, green: 0xF5 / 255, blue: 0xF2 / 255, alpha: 1)
  static let filmUI = UIColor.black
  static let inkUI = UIColor(red: 0x1D / 255, green: 0x1E / 255, blue: 0x20 / 255, alpha: 1)
  static let film = Color(filmUI)
  static let edge = Color(red: 0xE2 / 255, green: 0xDF / 255, blue: 0xD3 / 255)
  static let marker = Color(red: 0xD1 / 255, green: 0x33 / 255, blue: 0x2B / 255)
  static let graphite = Color(red: 0x6E / 255, green: 0x71 / 255, blue: 0x76 / 255)

  /// The page: Paper, or Film in dark mode.
  static let ground = adaptive(light: paperUI, dark: filmUI)
  /// Text and the primary button: Ink, or Paper in dark mode.
  static let text = adaptive(light: inkUI, dark: paperUI)
  /// The primary button's label: the page colour.
  static let onText = ground
  static let secondary = adaptive(
    light: UIColor(red: 0x6E / 255, green: 0x71 / 255, blue: 0x76 / 255, alpha: 1),
    dark: UIColor(red: 0x9A / 255, green: 0x9D / 255, blue: 0xA2 / 255, alpha: 1))
  static let rule = adaptive(
    light: inkUI.withAlphaComponent(0.14), dark: paperUI.withAlphaComponent(0.16))
  /// Kept for the screens that still use the names: the raised surface is the page itself
  /// now, and the accent is Ink.
  static let surface = ground
  static let accent = text

  static let gutter: CGFloat = 16
  static let gap: CGFloat = 14

  static let display = font("Archivo-Display", 32, .heavy, relativeTo: .largeTitle)
  static let displayHuge = font("Archivo-Display", 44, .heavy, relativeTo: .largeTitle)
  static let title = font("Archivo-TextSemibold", 21, .semibold, relativeTo: .title2)
  static let headline = font("Archivo-TextSemibold", 17, .semibold, relativeTo: .headline)
  static let body = font("Archivo-Text", 17, .regular, relativeTo: .body)
  static let label = font("Archivo-TextSemibold", 15, .semibold, relativeTo: .subheadline)
  static let caption = font("Archivo-Text", 15, .regular, relativeTo: .footnote)
  static let data = font("Archivo-Text", 15, .regular, relativeTo: .body).monospacedDigit()
  static let edgePrint = font("Archivo-Edge", 11, .medium, relativeTo: .caption2)
    .monospacedDigit()
  static let big = font("Archivo-Display", 44, .heavy, relativeTo: .largeTitle)
    .monospacedDigit()

  /// One of Archivo's static instances by its PostScript name, else the system font in the
  /// same text style and weight.
  static func font(
    _ name: String, _ size: CGFloat, _ weight: Font.Weight, relativeTo style: Font.TextStyle
  ) -> Font {
    if UIFont(name: name, size: size) != nil {
      return .custom(name, size: size, relativeTo: style)
    }
    return .system(style, weight: weight)
  }

  static func adaptive(light: UIColor, dark: UIColor) -> Color {
    Color(UIColor { $0.userInterfaceStyle == .dark ? dark : light })
  }
}

/// Applied once to the root view. The app follows the phone's light or dark setting.
struct ThemeModifier: ViewModifier {
  func body(content: Content) -> some View {
    content
      .font(Theme.body)
      .foregroundStyle(Theme.text)
      .tint(Theme.text)
  }
}
