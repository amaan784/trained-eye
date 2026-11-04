//
// ClipsView.swift
//
// The clips the user saved, newest first, and nothing else. Each opens to its frames and
// can be deleted. They are ordinary files: the Files app shows them under On My iPhone,
// Trained Eye, Clips, from where a folder can go to the Mac by AirDrop.
//

import SwiftUI
import UIKit

struct ClipsView: View {
  let store: ClipStore

  var body: some View {
    Screen(title: "Clips") {
      if store.clips.isEmpty {
        Notice(text: "No clips. Save one from the Watch screen: it keeps the last 10 seconds.")
      }
      VStack(spacing: Theme.gap) {
        ForEach(store.clips) { clip in
          NavigationLink(value: clip) {
            ClipRow(clip: clip)
          }
          .buttonStyle(.plain)
        }
      }
      // Inside the navigation stack, where the links can see it.
      .navigationDestination(for: Clip.self) { clip in
        ClipDetail(store: store, clip: clip)
      }
      Notice(
        text: "To replay a clip on the Mac, copy its folder there and run "
          + "te watch --skills <skill> --clip <folder>. Blur faces before publishing any frame.")
    }
    .onAppear { store.reload() }
  }
}

private struct ClipRow: View {
  let clip: Clip

  var body: some View {
    Card {
      HStack(spacing: Theme.gap) {
        Thumbnail(url: clip.frames.first)
          .frame(width: 56, height: 72)
          .clipped()
        VStack(alignment: .leading, spacing: 4) {
          Text(clip.savedAt.formatted(date: .abbreviated, time: .shortened))
            .font(Theme.headline)
          Text(clip.note).font(Theme.caption).foregroundStyle(Theme.secondary)
        }
        Spacer()
        Image(systemName: "chevron.right").foregroundStyle(Theme.secondary)
      }
    }
  }
}

private struct ClipDetail: View {
  let store: ClipStore
  let clip: Clip
  @Environment(\.dismiss) private var dismiss
  @State private var confirmDelete = false

  var body: some View {
    ScrollView {
      LazyVStack(alignment: .leading, spacing: Theme.gap) {
        Text(clip.note).font(Theme.caption).foregroundStyle(Theme.secondary)
        ForEach(clip.frames, id: \.self) { url in
          Thumbnail(url: url)
            .frame(maxWidth: .infinity)
            .clipped()
        }
        Button("Delete this clip") { confirmDelete = true }
          .buttonStyle(SecondaryButtonStyle())
      }
      .padding(Theme.gutter)
    }
    .background(Theme.ground)
    .navigationTitle(clip.id)
    .confirmationDialog("Delete this clip?", isPresented: $confirmDelete) {
      Button("Delete", role: .destructive) {
        store.delete(clip)
        dismiss()
      }
    }
  }
}

/// One frame from disk; the frames are small JPEGs, read as they scroll into view.
private struct Thumbnail: View {
  let url: URL?

  var body: some View {
    if let url, let image = UIImage(contentsOfFile: url.path(percentEncoded: false)) {
      Image(uiImage: image).resizable().scaledToFit()
    } else {
      Theme.film
    }
  }
}
