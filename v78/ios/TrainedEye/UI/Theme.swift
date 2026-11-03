//
// Theme.swift
//
// The look from HANDOVER section 6: dark and minimal, one accent colour, flat shapes and
// no gradients, in Inter 4.1 (Fonts/, SIL Open Font License), with the system font as the
// fallback if the fonts are missing from the target. Laid
// out like the theme in Meta's BirdSpotter sample: the tokens live in one place and are
// applied once, at the root.
//

import SwiftUI
import UIKit

enum Theme {
  // A near-black ground, one raised surface, two greys for text, and one accent.
  static let ground = Color(red: 0.05, green: 0.06, blue: 0.07)
  static let surface = Color(red: 0.10, green: 0.11, blue: 0.13)
  static let rule = Color.white.opacity(0.10)
  static let text = Color(white: 0.94)
  static let secondary = Color(white: 0.60)
  /// The one accent: live states, the main action and the camera-on indicator.
  static let accent = Color(red: 0.24, green: 0.86, blue: 0.60)

  static let corner: CGFloat = 12
  static let gutter: CGFloat = 16
  static let gap: CGFloat = 12

  static let title = font(22, .semibold, relativeTo: .title2)
  static let headline = font(17, .semibold, relativeTo: .headline)
  static let body = font(16, .regular, relativeTo: .body)
  static let label = font(13, .medium, relativeTo: .subheadline)
  static let caption = font(12, .regular, relativeTo: .caption)
  static let data = font(15, .medium, relativeTo: .body).monospacedDigit()
  static let big = font(44, .semibold, relativeTo: .largeTitle).monospacedDigit()

  /// Inter's static faces from Fonts/ (in the target and listed under UIAppFonts), else
  /// the system font in the same text style and weight. The names are the faces'
  /// PostScript names, read from the Inter 4.1 files.
  static func font(_ size: CGFloat, _ weight: Font.Weight, relativeTo style: Font.TextStyle)
    -> Font
  {
    let name: String
    switch weight {
    case .semibold: name = "Inter-SemiBold"
    case .medium: name = "Inter-Medium"
    case .bold: name = "Inter-Bold"
    default: name = "Inter-Regular"
    }
    if UIFont(name: name, size: size) != nil {
      return .custom(name, size: size, relativeTo: style)
    }
    return .system(style, weight: weight)
  }
}

/// Applied once to the root view.
struct ThemeModifier: ViewModifier {
  func body(content: Content) -> some View {
    content
      .font(Theme.body)
      .foregroundStyle(Theme.text)
      .tint(Theme.accent)
      .preferredColorScheme(.dark)
  }
}
