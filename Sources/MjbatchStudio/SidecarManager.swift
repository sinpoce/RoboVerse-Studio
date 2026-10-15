// SPDX-License-Identifier: Apache-2.0
// Launches (and tears down) the Python sidecar, and discovers the port it binds.

import Foundation

/// Resolves where the Python interpreter and server.py live, launches the
/// sidecar, scrapes the "MJBATCH_STUDIO_PORT <n>" line from its stdout, and
/// keeps its stderr for diagnostics. All callbacks are delivered on the main queue.
final class SidecarManager {
    enum State: Equatable {
        case idle
        case launching
        case running(port: Int)
        case failed(String)
    }

    private var process: Process?
    private var stdoutBuffer = Data()
    private var stderrTail = ""
    private let onState: (State) -> Void
    private let training: Bool

    init(training: Bool = false, onState: @escaping (State) -> Void) {
        self.training = training
        self.onState = onState
    }

    /// If MJBATCH_STUDIO_PORT is exported, we assume a sidecar is already running
    /// (handy for debugging) and skip launching.
    func start() {
        let env = ProcessInfo.processInfo.environment
        if let portStr = env["MJBATCH_STUDIO_PORT"], let port = Int(portStr) {
            emit(.running(port: port))
            return
        }
        guard let (python, server) = resolvePaths() else {
            emit(.failed("""
            Could not find the Python sidecar. Run scripts/setup_python.sh, then set \
            MJBATCH_STUDIO_HOME to the project directory (or launch from it).
            """))
            return
        }
        launch(python: python, server: server)
    }

    func stop() {
        process?.terminationHandler = nil
        process?.terminate()
        process = nil
    }

    // MARK: - path resolution

    private func resolvePaths() -> (python: String, server: String)? {
        let env = ProcessInfo.processInfo.environment
        let fm = FileManager.default

        // 1. explicit overrides
        if let py = env["MJBATCH_STUDIO_PYTHON"], let sv = env["MJBATCH_STUDIO_SERVER"],
           fm.isExecutableFile(atPath: py), fm.fileExists(atPath: sv),
           training || URL(fileURLWithPath: sv).lastPathComponent == "server.py" {
            return (py, sv)
        }

        // A distributable build carries a relocatable Python runtime and all model assets.
        if let resources = Bundle.main.resourcePath {
            let py = resources + "/runtime/bin/python3"
            let sv = resources + (training ? "/examples/generic_train_server.py" : "/python/server.py")
            if fm.isExecutableFile(atPath: py), fm.fileExists(atPath: sv) {
                return (py, sv)
            }
        }

        // 2. candidate project roots, most specific first
        var roots: [String] = []
        if let home = env["MJBATCH_STUDIO_HOME"] { roots.append(home) }
        roots.append(fm.currentDirectoryPath)
        // resources inside a built .app bundle
        if let resource = Bundle.main.resourcePath { roots.append(resource) }
        // walk up from the executable (…/.build/... or …/Contents/MacOS)
        var dir = URL(fileURLWithPath: Bundle.main.bundlePath)
        for _ in 0..<6 {
            roots.append(dir.path)
            dir.deleteLastPathComponent()
        }

        for root in roots {
            let py = root + "/.venv/bin/python"
            let sv = root + (training ? "/examples/generic_train_server.py" : "/python/server.py")
            if fm.isExecutableFile(atPath: py), fm.fileExists(atPath: sv) {
                return (py, sv)
            }
        }
        return nil
    }

    // MARK: - launch

    private func launch(python: String, server: String) {
        emit(.launching)
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: python)
        proc.arguments = ["-B", server, "--port", "0"]
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
        environment.removeValue(forKey: "PYTHONPATH")
        environment.removeValue(forKey: "PYTHONHOME")
        proc.environment = environment
        // Native solver logs and runtime scratch files must never modify the signed app bundle.
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("RoboVerse Studio", isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: support, withIntermediateDirectories: true)
            proc.currentDirectoryURL = support
        } catch {
            emit(.failed("Cannot create runtime directory: \(error.localizedDescription)"))
            return
        }

        let outPipe = Pipe()
        let errPipe = Pipe()
        proc.standardOutput = outPipe
        proc.standardError = errPipe

        outPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let chunk = handle.availableData
            guard !chunk.isEmpty else { return }
            self?.consumeStdout(chunk)
        }
        errPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let chunk = handle.availableData
            guard let text = String(data: chunk, encoding: .utf8), !text.isEmpty else { return }
            self?.stderrTail = String((self?.stderrTail ?? "") + text).suffix(2000).description
        }

        proc.terminationHandler = { [weak self] p in
            guard let self else { return }
            let tail = self.stderrTail.isEmpty ? "(no stderr)" : self.stderrTail
            self.emit(.failed("Sidecar exited (code \(p.terminationStatus)).\n\(tail)"))
        }

        do {
            try proc.run()
            process = proc
        } catch {
            emit(.failed("Failed to launch sidecar: \(error.localizedDescription)"))
        }
    }

    private func consumeStdout(_ chunk: Data) {
        stdoutBuffer.append(chunk)
        while let nl = stdoutBuffer.firstIndex(of: 0x0A) {
            let lineData = stdoutBuffer.subdata(in: stdoutBuffer.startIndex..<nl)
            stdoutBuffer.removeSubrange(stdoutBuffer.startIndex...nl)
            guard let line = String(data: lineData, encoding: .utf8) else { continue }
            if line.hasPrefix("MJBATCH_STUDIO_PORT ") {
                let portStr = line.replacingOccurrences(of: "MJBATCH_STUDIO_PORT ", with: "")
                    .trimmingCharacters(in: .whitespaces)
                if let port = Int(portStr) {
                    emit(.running(port: port))
                }
            }
        }
    }

    private func emit(_ state: State) {
        DispatchQueue.main.async { self.onState(state) }
    }
}
