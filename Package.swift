// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "RoboVerseStudio",
    platforms: [.macOS(.v14)],  // SwiftUI + Swift Charts
    targets: [
        .executableTarget(
            name: "RoboVerseStudio",
            path: "Sources/MjbatchStudio",
            resources: [.process("Resources")]
        )
    ]
)
