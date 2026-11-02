//
// RootView.swift
//
// The five tabs, and the camera-on indicator above every one of them: while any camera is
// on, a bar says so and which camera it is, so watch mode is always visible in the app.
//

import SwiftUI

struct RootView: View {
  let app: AppModel
  @State private var tab = Tab.connect

  enum Tab: Hashable {
    case connect, teach, skills, watch, clips
  }

  var body: some View {
    @Bindable var glassesCamera = app.glassesCamera
    TabView(selection: $tab) {
      ConnectView(app: app)
        .tabItem { Label("Connect", systemImage: "link") }
        .tag(Tab.connect)
      TeachView(app: app, flow: app.teach)
        .tabItem { Label("Teach", systemImage: "graduationcap") }
        .tag(Tab.teach)
      SkillsView(app: app)
        .tabItem { Label("Skills", systemImage: "list.bullet") }
        .tag(Tab.skills)
      WatchView(app: app)
        .tabItem { Label("Watch", systemImage: "eye") }
        .tag(Tab.watch)
      ClipsView(store: app.clips)
        .tabItem { Label("Clips", systemImage: "film") }
        .tag(Tab.clips)
    }
    .safeAreaInset(edge: .top, spacing: 0) {
      if app.isCameraOn {
        CameraOnBar(app: app) { tab = .watch }
      }
    }
    // CameraAccess's flow: confirm before the switch to Meta AI for camera permission.
    .alert("Allow the glasses' camera", isPresented: $glassesCamera.needsPermissionConfirm) {
      Button("Continue") {
        Task { await app.glassesCamera.confirmCameraPermission() }
      }
      Button("Not now", role: .cancel) {
        app.stopWatching()
      }
    } message: {
      Text(
        "The Meta AI app opens so you can let Trained Eye use the glasses' camera. "
          + "Choose Allow once or Allow always, then come back.")
    }
  }
}

/// The always-visible indicator. Tapping it opens the Watch screen.
private struct CameraOnBar: View {
  let app: AppModel
  let open: () -> Void

  var body: some View {
    Button(action: open) {
      HStack(spacing: 8) {
        StatusDot(on: true)
        Text(text).font(Theme.label)
        Spacer()
        if app.isWatching {
          Text(app.socket.status.label).font(Theme.caption).foregroundStyle(Theme.secondary)
        }
      }
      .padding(.horizontal, Theme.gutter)
      .padding(.vertical, 8)
      .background(Theme.surface)
      .overlay(alignment: .bottom) {
        Rectangle().fill(Theme.accent).frame(height: 1)
      }
    }
    .buttonStyle(.plain)
    .accessibilityLabel(text)
  }

  private var text: String {
    let camera = app.source?.label ?? "camera"
    return app.isWatching ? "Watching with the \(camera)" : "Camera on: the \(camera)"
  }
}
