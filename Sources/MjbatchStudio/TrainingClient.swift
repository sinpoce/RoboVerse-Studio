// SPDX-License-Identifier: Apache-2.0
// WebSocket client for the sidecar. Text -> ServerMessage, binary -> JPEG frame.

import Foundation

protocol TrainingClientDelegate: AnyObject {
    func trainingClient(didReceive message: ServerMessage)
    func trainingClient(didReceiveFrame data: Data)
    func trainingClientDidConnect()
    func trainingClient(didFail reason: String)
}

/// Thin wrapper over URLSessionWebSocketTask. Sending is fire-and-forget JSON;
/// receiving re-arms itself after every message. Delegate calls hop to main.
final class TrainingClient {
    weak var delegate: TrainingClientDelegate?

    private var task: URLSessionWebSocketTask?
    private let session = URLSession(configuration: .default)
    private var isOpen = false

    func connect(port: Int) {
        disconnect()
        guard let url = URL(string: "ws://127.0.0.1:\(port)") else {
            fail("bad url for port \(port)")
            return
        }
        let task = session.webSocketTask(with: url)
        self.task = task
        task.resume()
        isOpen = true
        // The socket opens lazily; announce and start pumping. The first send
        // (hello handshake happens server-side on connect) will confirm liveness.
        DispatchQueue.main.async { self.delegate?.trainingClientDidConnect() }
        receiveLoop()
    }

    func disconnect() {
        isOpen = false
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
    }

    /// Serialise a command dictionary and send it as a text frame.
    func send(_ dict: [String: Any]) {
        guard isOpen, let task else { return }
        guard let data = try? JSONSerialization.data(withJSONObject: dict),
              let text = String(data: data, encoding: .utf8) else { return }
        task.send(.string(text)) { [weak self] error in
            if let error { self?.fail("send failed: \(error.localizedDescription)") }
        }
    }

    private func receiveLoop() {
        task?.receive { [weak self] result in
            guard let self, self.isOpen else { return }
            switch result {
            case .failure(let error):
                self.fail("receive failed: \(error.localizedDescription)")
            case .success(let message):
                switch message {
                case .string(let text):
                    self.handleText(text)
                case .data(let data):
                    DispatchQueue.main.async { self.delegate?.trainingClient(didReceiveFrame: data) }
                @unknown default:
                    break
                }
                self.receiveLoop()  // re-arm
            }
        }
    }

    private func handleText(_ text: String) {
        guard let data = text.data(using: .utf8) else { return }
        do {
            let msg = try JSONDecoder().decode(ServerMessage.self, from: data)
            DispatchQueue.main.async { self.delegate?.trainingClient(didReceive: msg) }
        } catch {
            // ignore malformed frames rather than tearing down the connection
        }
    }

    private func fail(_ reason: String) {
        guard isOpen else { return }
        isOpen = false
        DispatchQueue.main.async { self.delegate?.trainingClient(didFail: reason) }
    }
}
