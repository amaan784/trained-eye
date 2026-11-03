//
// Settings.swift
//
// What the app remembers between launches, in UserDefaults: the backend address (the Mac
// on the home Wi-Fi, or a tunnel to it), the skills switched on for watching, and which
// camera to use. The server token, which the teach routes need, is in the keychain.
//

import Foundation
import Observation
import Security

/// Which camera watch mode and the teach capture use.
enum CameraChoice: String, CaseIterable, Identifiable {
  case auto
  case glasses
  case phone

  var id: String { rawValue }

  var label: String {
    switch self {
    case .auto: return "Glasses, else iPhone"
    case .glasses: return "Glasses"
    case .phone: return "iPhone"
    }
  }
}

/// Observable by hand (an ObservationRegistrar rather than the macro), so every read and
/// write goes straight to UserDefaults and SwiftUI still sees each change.
@MainActor
final class AppSettings: Observable {
  /// The address `te serve` prints, as http://<Mac IP>:8765 on the home Wi-Fi or
  /// https://<name> through a tunnel. ws:// and wss:// addresses are accepted too.
  var backendURL: String {
    get {
      registrar.access(self, keyPath: \.backendURL)
      return defaults.string(forKey: Keys.backendURL) ?? "http://192.168.1.20:8765"
    }
    set {
      registrar.withMutation(of: self, keyPath: \.backendURL) {
        defaults.set(newValue, forKey: Keys.backendURL)
      }
    }
  }

  /// The skills switched on in the Skills screen; watch mode asks the backend for these.
  var watchedSkills: Set<String> {
    get {
      registrar.access(self, keyPath: \.watchedSkills)
      return Set(defaults.stringArray(forKey: Keys.watchedSkills) ?? [])
    }
    set {
      registrar.withMutation(of: self, keyPath: \.watchedSkills) {
        defaults.set(newValue.sorted(), forKey: Keys.watchedSkills)
      }
    }
  }

  var cameraChoice: CameraChoice {
    get {
      registrar.access(self, keyPath: \.cameraChoice)
      return CameraChoice(rawValue: defaults.string(forKey: Keys.cameraChoice) ?? "") ?? .auto
    }
    set {
      registrar.withMutation(of: self, keyPath: \.cameraChoice) {
        defaults.set(newValue.rawValue, forKey: Keys.cameraChoice)
      }
    }
  }

  /// SERVER_TOKEN from the Mac's .env: the teach routes refuse a request without it.
  var serverToken: String {
    get {
      registrar.access(self, keyPath: \.serverToken)
      return Keychain.read(Keys.serverToken) ?? ""
    }
    set {
      registrar.withMutation(of: self, keyPath: \.serverToken) {
        Keychain.write(Keys.serverToken, newValue.trimmingCharacters(in: .whitespacesAndNewlines))
      }
    }
  }

  private let registrar = ObservationRegistrar()
  private let defaults: UserDefaults

  private enum Keys {
    static let backendURL = "backendURL"
    static let watchedSkills = "watchedSkills"
    static let cameraChoice = "cameraChoice"
    static let serverToken = "serverToken"
  }

  init(defaults: UserDefaults = .standard) {
    self.defaults = defaults
  }

  /// The backend's HTTP base address (scheme, host and port only), or nil while the text
  /// is not a usable address.
  var baseURL: URL? {
    let text = backendURL.trimmingCharacters(in: .whitespacesAndNewlines)
    guard var parts = URLComponents(string: text),
      let scheme = parts.scheme?.lowercased(),
      let host = parts.host, !host.isEmpty
    else { return nil }
    switch scheme {
    case "http", "ws": parts.scheme = "http"
    case "https", "wss": parts.scheme = "https"
    default: return nil
    }
    parts.path = ""
    parts.query = nil
    parts.fragment = nil
    return parts.url
  }

  /// The WebSocket address: ws:// for http, wss:// for https, at /ws.
  var webSocketURL: URL? {
    guard let base = baseURL, var parts = URLComponents(url: base, resolvingAgainstBaseURL: false)
    else { return nil }
    parts.scheme = parts.scheme == "https" ? "wss" : "ws"
    parts.path = "/ws"
    return parts.url
  }

  /// True for an address on the local network (a private IPv4 address or a .local name),
  /// false for anything else, which is taken to be a tunnel.
  var isLocalNetwork: Bool {
    guard let host = baseURL?.host?.lowercased() else { return false }
    if host.hasSuffix(".local") || host == "localhost" { return true }
    let octets = host.split(separator: ".").compactMap { Int($0) }
    guard octets.count == 4 else { return false }
    switch (octets[0], octets[1]) {
    case (10, _), (192, 168), (169, 254): return true
    case (172, let second): return (16...31).contains(second)
    default: return false
    }
  }

  /// One line for the Connect screen saying what kind of address this is.
  var addressKind: String {
    guard let base = baseURL else { return "Not a usable address yet" }
    if isLocalNetwork { return "Local Wi-Fi: the iPhone and the Mac must share the network" }
    return base.scheme == "https"
      ? "Tunnel over HTTPS" : "Plain HTTP outside the local network: use an HTTPS tunnel"
  }
}

/// One generic-password item per key, readable only on this device after the first
/// unlock (so a backup or another phone never gets it).
enum Keychain {
  private static let service = "TrainedEye"

  private static func query(_ key: String) -> [String: Any] {
    [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
     kSecAttrAccount as String: key]
  }

  static func read(_ key: String) -> String? {
    var q = query(key)
    q[kSecReturnData as String] = true
    q[kSecMatchLimit as String] = kSecMatchLimitOne
    var item: CFTypeRef?
    guard SecItemCopyMatching(q as CFDictionary, &item) == errSecSuccess,
      let data = item as? Data
    else { return nil }
    return String(data: data, encoding: .utf8)
  }

  /// Stores `value`, or removes the item when it is empty.
  static func write(_ key: String, _ value: String) {
    SecItemDelete(query(key) as CFDictionary)
    guard !value.isEmpty else { return }
    var q = query(key)
    q[kSecValueData as String] = Data(value.utf8)
    q[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
    SecItemAdd(q as CFDictionary, nil)
  }
}
