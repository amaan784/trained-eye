//
// TrainedEyeApp.swift
//
// The app entry. Configures Meta's DAT SDK once at launch and forwards the Meta AI app's
// callbacks (registration and permission flows) to it, as CameraAccessApp.swift and
// RegistrationView.swift in Meta's CameraAccess sample do. The callback scheme
// (trainedeye://) and the SDK's settings are in ios/TrainedEye-Info.plist.
//

import MWDATCore
import SwiftUI

@main
struct TrainedEyeApp: App {
  @State private var app: AppModel

  init() {
    do {
      try Wearables.configure()
    } catch {
      #if DEBUG
      NSLog("[TrainedEye] Failed to configure Wearables SDK: \(error)")
      #endif
    }
    // After configure(): AppModel reads Wearables.shared.
    _app = State(wrappedValue: AppModel())
  }

  var body: some Scene {
    WindowGroup {
      RootView(app: app)
        .modifier(ThemeModifier())
        .onOpenURL { url in
          guard
            let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
            // Only the DAT SDK's own callbacks carry metaWearablesAction.
            components.queryItems?.contains(where: { $0.name == "metaWearablesAction" }) == true
          else {
            return
          }
          Task {
            do {
              _ = try await Wearables.shared.handleUrl(url)
            } catch let error as RegistrationError {
              app.glasses.errorMessage = error.description
            } catch {
              app.glasses.errorMessage = error.localizedDescription
            }
          }
        }
    }
  }
}
