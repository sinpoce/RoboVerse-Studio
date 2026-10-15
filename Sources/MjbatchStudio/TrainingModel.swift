// SPDX-License-Identifier: Apache-2.0

import AppKit
import Foundation
import SwiftUI

/// One metric sample at a completed training iteration.
struct ChartSample: Identifiable {
    let id: Int
    let x: Double
    let y: Double
}

/// Training session state. Scene editing and physics belong to StudioModel.
final class TrainingModel: ObservableObject, TrainingClientDelegate {
    enum Phase: Equatable {
        case starting, launching, connecting, connected, failed(String)
    }
    @Published var phase: Phase = .starting
    @Published var models: [NamedOption] = []
    @Published var metrics: [String] = []
    @Published var algorithms: [NamedOption] = []
    @Published var selectedAlgorithm = ""
    @Published var trainIteration = 0
    @Published var trainIterations = 0
    @Published var trainElapsed = 0.0
    @Published var trainTerms: [NamedValue] = []
    private var trainHistory: [(iteration: Int, values: [String: Double])] = []
    @Published var taskConfigPath = ""
    @Published var checkpointPath = "runs/policy.pt"
    @Published var trainingParametersText = "{}"
    @Published var trainingStarting = false
    @Published var trainingStopping = false
    @Published var trainingLogs: [StudioLog] = []
    @Published var activity = "training"
    @Published var evaluationSummary = ""
    @Published var evaluationReportPath = ""
    @Published var evaluationRecordingPath = ""
    @Published var numSimsText = "8"
    @Published var modelInfo = TrainingRobotInfo()
    @Published var metricName = "reward"
    @Published var running = false
    @Published var frame: NSImage?
    @Published var stepsPerSecond = 0.0
    @Published var metricValue = 0.0
    @Published var throughputSamples: [ChartSample] = []
    @Published var metricSamples: [ChartSample] = []
    @Published var statusMessage = "Starting…"

    var evaluating: Bool { activity == "evaluation" }
    var canEvaluate: Bool { parameter("adapter") == "stagewise_mujoco" && parameter("task_config").contains("backflip") }
    var trainingBusy: Bool { running || trainingStarting || trainingStopping }
    var projectTitle: String { models.first?.label ?? L("训练项目") }
    var algorithmTitle: String { algorithms.first(where: { $0.id == selectedAlgorithm })?.label ?? L("选择算法") }
    private var sidecar: SidecarManager?
    private let client = TrainingClient()
    private var numSims: Int { max(1, Int(numSimsText) ?? 8) }

    func start() {
        guard sidecar == nil else { return }
        client.delegate = self
        phase = .launching
        statusMessage = "Launching Python sidecar…"
        let manager = SidecarManager(training: true) { [weak self] state in
            self?.handleSidecar(state)
        }
        sidecar = manager
        manager.start()
    }

    func retry() {
        shutdown()
        start()
    }

    func shutdown() {
        client.send(["cmd": "pause"])
        client.disconnect()
        sidecar?.stop()
        sidecar = nil
        trainingStarting = false
        trainingStopping = false
    }

    private func handleSidecar(_ state: SidecarManager.State) {
        switch state {
        case .idle:
            break
        case .launching:
            phase = .launching
            statusMessage = "Launching Python sidecar…"
        case .running(let port):
            phase = .connecting
            statusMessage = "Connecting to sidecar on port \(port)…"
            client.connect(port: port)
        case .failed(let reason):
            phase = .failed(reason)
            statusMessage = "Sidecar error"
        }
    }

    func trainingClientDidConnect() {
        phase = .connected
        statusMessage = "Connected."
    }

    func trainingClient(didFail reason: String) {
        phase = .failed(reason)
        statusMessage = "Connection lost"
        running = false
        trainingStarting = false
        trainingStopping = false
    }

    func trainingClient(didReceiveFrame data: Data) {
        if let image = NSImage(data: data) { frame = image }
    }

    func trainingClient(didReceive message: ServerMessage) {
        switch message.type {
        case "hello":
            break
        case "model_list":
            models = message.models ?? []
            metrics = message.metrics ?? []
            if metricName.isEmpty { metricName = metrics.first ?? "reward" }
        case "model_info":
            modelInfo = TrainingRobotInfo(name: message.name ?? "", nu: message.nu ?? 0)
            if let modelMetrics = message.metrics, !modelMetrics.isEmpty {
                metrics = modelMetrics
                metricName = modelMetrics[0]
            }
            algorithms = message.algorithms ?? []
            if !algorithms.contains(where: { $0.id == selectedAlgorithm }) {
                selectedAlgorithm = algorithms.first?.id ?? ""
            }
        case "configured":
            running = false
            if let n = message.num_sims { numSimsText = String(n) }
            resetCharts()
            statusMessage = L("训练环境已就绪")
        case "status":
            activity = message.activity ?? "training"
            running = message.running ?? false
            if running { trainingStarting = false }
            if !running { trainingStopping = false }
            if let algo = message.algo { selectedAlgorithm = algo }
            stepsPerSecond = message.steps_per_s ?? 0
            metricValue = message.metric_value ?? 0
            if !evaluating { recordIteration(message) }
        case "training_config":
            trainingParametersText = message.config_json ?? "{}"
            checkpointPath = message.checkpoint ?? checkpointPath
            taskConfigPath = message.config_path ?? taskConfigPath
            if let algo = message.algo { selectedAlgorithm = algo }
            if let sims = message.num_sims { numSimsText = String(sims) }
        case "evaluation_result":
            evaluationSummary = String(format: L("后空翻成功率 %.0f%% · %d 回合"), (message.success_rate ?? 0)*100, message.episodes ?? 0)
            evaluationReportPath = message.report_path ?? ""
            evaluationRecordingPath = message.recording_path ?? ""
            trainingStarting = false
        case "error":
            statusMessage = "⚠︎ " + (message.message ?? "unknown error")
            trainingStarting = false
            trainingLogs.append(StudioLog(text: statusMessage, error: true))
        case "log":
            statusMessage = message.message ?? ""
            trainingLogs.append(StudioLog(text: statusMessage, error: false))
            if trainingLogs.count > 300 { trainingLogs.removeFirst(trainingLogs.count - 300) }
        default:
            break
        }
    }

    private func resetCharts() {
        throughputSamples.removeAll(keepingCapacity: true)
        metricSamples.removeAll(keepingCapacity: true)
        trainHistory.removeAll()
        trainIteration = 0
        trainElapsed = 0
        trainTerms = []
    }

    private func recordIteration(_ message: ServerMessage) {
        if let n = message.iterations { trainIterations = n }
        if let t = message.elapsed { trainElapsed = t }
        if let terms = message.terms { trainTerms = terms }
        guard let it = message.iteration else { return }
        if it < trainIteration { trainHistory.removeAll() }  // a new run started
        trainIteration = it
        guard it > 0, let values = message.values, trainHistory.last?.iteration != it else { return }
        trainHistory.append((it, values))
        rebuildTrainingCharts()
    }

    private func rebuildTrainingCharts() {
        throughputSamples = trainHistory.map {
            ChartSample(id: $0.iteration, x: Double($0.iteration), y: $0.values["steps/s (throughput)"] ?? 0)
        }
        metricSamples = trainHistory.map {
            ChartSample(id: $0.iteration, x: Double($0.iteration), y: $0.values[metricName] ?? 0)
        }
    }

    func setAlgorithm(_ id: String) {
        evaluationSummary = ""
        evaluationReportPath = ""
        selectedAlgorithm = id
        running = false
        client.send(["cmd": "algorithm", "name": id])
        resetCharts()
        frame = nil
        modelInfo = TrainingRobotInfo()
    }

    func loadTaskConfig(_ path: String) {
        evaluationSummary = ""
        evaluationReportPath = ""
        taskConfigPath = path
        frame = nil
        resetCharts()
        client.send(["cmd": "task_config", "path": path])
    }

    func startTraining() {
        guard !trainingBusy else { return }
        evaluationSummary = ""
        evaluationReportPath = ""
        trainingStarting = true
        statusMessage = L("正在准备训练环境…")
        client.send(["cmd": "start_training", "config_json": trainingParametersText,
                     "num_sims": numSims, "checkpoint": checkpointPath])
    }

    func evaluatePolicy() {
        guard !trainingBusy else { return }
        trainingStarting = true
        evaluationSummary = ""
        evaluationReportPath = ""
        statusMessage = L("正在加载策略进行独立评估…")
        client.send(["cmd": "evaluate_policy", "checkpoint": checkpointPath, "episodes": 5, "seed": 1001])
    }

    func loadCheckpoint(_ path: String) {
        guard !trainingBusy else { return }
        evaluationSummary = ""
        evaluationReportPath = ""
        client.send(["cmd": "load_checkpoint", "path": path])
    }

    func parameter(_ key: String, fallback: String = "") -> String {
        guard let data = trainingParametersText.data(using: .utf8),
              let values = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let value = values[key] else { return fallback }
        return String(describing: value)
    }

    func setParameter(_ key: String, value: String) {
        guard let data = trainingParametersText.data(using: .utf8),
              var values = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
        values[key] = Int(value).map { $0 as Any } ?? value
        if let encoded = try? JSONSerialization.data(withJSONObject: values, options: [.prettyPrinted, .sortedKeys]),
           let text = String(data: encoded, encoding: .utf8) { trainingParametersText = text }
        if key == "num_envs" { numSimsText = value }
    }

    func stopTraining() {
        trainingStopping = true
        statusMessage = evaluating ? L("正在停止评估…") : L("正在停止并保存检查点…")
        client.send(["cmd": "pause"])
    }

    func pushMetric() {
        client.send(["cmd": "metric", "name": metricName])
        rebuildTrainingCharts()
    }
}
