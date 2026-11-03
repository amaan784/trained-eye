//
// ClipStore.swift
//
// Clips the user saved from the Watch screen, and nothing else: one folder of JPEG frames
// per clip (f000.jpg, f001.jpg, ...) with a clip.json, under Documents/Clips. A folder of
// frames is what `te watch --clip` and `te teach --clip` read, so a clip copied to the Mac
// (Files app or AirDrop) can be replayed there. Nothing is written unless the user taps
// Save clip, and a clip is deleted from here only when the user deletes it.
//

import Foundation
import Observation

/// What clip.json holds.
struct ClipInfo: Codable {
  let savedAt: Date
  let source: String
  let fps: Int
  let skills: [String]
  let lastAlert: String?
}

struct Clip: Identifiable, Hashable {
  let id: String
  let folder: URL
  let frames: [URL]
  let savedAt: Date
  let note: String

  static func == (a: Clip, b: Clip) -> Bool { a.id == b.id }
  func hash(into hasher: inout Hasher) { hasher.combine(id) }
}

@MainActor
@Observable
final class ClipStore {
  private(set) var clips: [Clip] = []

  @ObservationIgnored private let root = URL.documentsDirectory.appending(
    path: "Clips", directoryHint: .isDirectory)

  init() {
    reload()
  }

  func reload() {
    let manager = FileManager.default
    let folders =
      (try? manager.contentsOfDirectory(
        at: root, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])) ?? []
    clips = folders.compactMap(Self.load).sorted { $0.savedAt > $1.savedAt }
  }

  /// Writes the frames as a new clip. Called only from the Save clip button.
  func save(frames: [Data], info: ClipInfo) throws {
    let formatter = DateFormatter()
    formatter.locale = Locale(identifier: "en_US_POSIX")
    formatter.dateFormat = "yyyyMMdd-HHmmss"
    let name = formatter.string(from: info.savedAt)
    var folder = root.appending(path: name, directoryHint: .isDirectory)
    var n = 1
    while FileManager.default.fileExists(atPath: folder.path(percentEncoded: false)) {
      n += 1
      folder = root.appending(path: "\(name)-\(n)", directoryHint: .isDirectory)
    }
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    for (i, jpeg) in frames.enumerated() {
      try jpeg.write(to: folder.appending(path: String(format: "f%03d.jpg", i)))
    }
    let encoder = JSONEncoder()
    encoder.dateEncodingStrategy = .iso8601
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(info).write(to: folder.appending(path: "clip.json"))
    reload()
  }

  func delete(_ clip: Clip) {
    try? FileManager.default.removeItem(at: clip.folder)
    reload()
  }

  nonisolated private static func load(_ folder: URL) -> Clip? {
    let files =
      (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil))
      ?? []
    let frames = files.filter { $0.pathExtension.lowercased() == "jpg" }
      .sorted { $0.lastPathComponent < $1.lastPathComponent }
    guard !frames.isEmpty else { return nil }
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601
    let info = (try? Data(contentsOf: folder.appending(path: "clip.json")))
      .flatMap { try? decoder.decode(ClipInfo.self, from: $0) }
    var note = "\(frames.count) frames"
    if let info {
      note += " from the \(info.source)"
      if let alert = info.lastAlert { note += ", last alert: \(alert)" }
    }
    return Clip(
      id: folder.lastPathComponent, folder: folder, frames: frames,
      savedAt: info?.savedAt ?? .distantPast, note: note)
  }
}
