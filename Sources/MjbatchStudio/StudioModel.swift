import AppKit
import Foundation
import SwiftUI
import UniformTypeIdentifiers

struct StudioBody: Decodable, Identifiable {
    let id: Int
    let name: String
    let parent: Int
    let depth: Int
    let mass: Double
    let position: [Double]
    let quaternion: [Double]
    let friction: Double
    let dynamic: Bool
    let geoms: Int
}
struct StudioJoint: Decodable, Identifiable {
    let id: Int
    let name: String
    let body: Int
    let lower: Double
    let upper: Double
    let unit: String
    let qadr: Int
    let vadr: Int
}
struct StudioActuator: Decodable, Identifiable {
    let id: Int
    let name: String
    let lower: Double
    let upper: Double
    let kind: String
    let pd: Bool
}
struct StudioSensor: Decodable, Identifiable {
    let id: Int
    let name: String
    let address: Int
    let dimension: Int
}
struct StudioController: Decodable, Equatable {
    let mode: String
    let target: Double
    let kp: Double
    let kd: Double
}
struct StudioScene: Decodable {
    let revision: Int
    let name: String
    let bodies: [StudioBody]
    let joints: [StudioJoint]
    let actuators: [StudioActuator]
    let sensors: [StudioSensor]
    let keyframes: [String]
    let gravity: [Double]
    let timestep: Double
    let iterations: Int
    let nq: Int
    let nv: Int
    let nu: Int
}
struct StudioStatus: Decodable, Equatable {
    var running = false
    var time = 0.0
    var speed = 1.0
    var rtf = 0.0
    var selected = 0
    var dirty = false
    var project_path = ""
    var qpos: [Double] = []
    var qvel: [Double] = []
    var ctrl: [Double] = []
    var actuator_force: [Double] = []
    var sensordata: [Double] = []
    var contacts = 0
    var contact_force = 0.0
    var recording = false
    var record_count = 0
    var replaying = false
    var replay_index = 0
    var replay_count = 0
    var can_undo = false
    var can_redo = false
    var controllers: [String: StudioController] = [:]
    var notice = ""
    var warnings = 0
    var selected_position: [Double]?
    var selected_quaternion: [Double]?
    var trajectory: String?
}
struct StudioLog: Identifiable {
    let id = UUID()
    let date = Date()
    let text: String
    let error: Bool
}
struct StudioSample: Identifiable {
    let id: Int
    let time: Double
    let value: Double
}
struct StudioSignal: Identifiable, Hashable {
    let id: String
    let name: String
    let family: String
    let index: Int
}

@MainActor
final class StudioVideo: ObservableObject {
    @Published var image: NSImage?
}

@MainActor
final class StudioModel: ObservableObject {
    @Published var trainingWorkspace = false
    @Published var connected = false
    @Published var connecting = true
    @Published var failure: String?
    @Published var scene: StudioScene?
    @Published var status = StudioStatus()
    let video = StudioVideo()
    @Published var logs: [StudioLog] = []
    @Published var notice = L("正在启动仿真引擎…")
    @Published var engineName = "MuJoCo"
    @Published var signalID = "contacts"
    @Published var samples: [StudioSample] = []
    @Published var showContacts = false
    @Published var showJoints = false
    @Published var collisionView = false
    @Published var inspectorTab = 0
    @Published var bottomTab = 0
    @Published var pending = 0
    @Published var recordingNeedsExport = false
    private var previousRecordCount = 0
    private var manager: SidecarManager?
    private var socket: URLSessionWebSocketTask?
    private let session = URLSession(configuration: .default)
    private var generation = UUID()
    private var requestIDs: Set<String> = []
    private var completions: [String: () -> Void] = [:]
    private var timeout: DispatchWorkItem?
    private var resizeWork: DispatchWorkItem?
    private var lastSize: (Int, Int)?
    private var lastSampleTime = -1.0
    private var sampleID = 0
    private var lastNotice = ""
    private var pendingURL: URL?

    var editable: Bool { connected && !status.running && !status.recording && !status.replaying && pending == 0 }
    var selectedBody: StudioBody? { scene?.bodies.first { $0.id == status.selected } }
    var title: String { status.project_path.isEmpty ? (scene?.name ?? L("新世界")) : URL(fileURLWithPath: status.project_path).deletingPathExtension().lastPathComponent }
    var signals: [StudioSignal] {
        var values = [StudioSignal(id: "contacts", name: L("接触点数量"), family: "contacts", index: 0),
                      StudioSignal(id: "force", name: L("接触法向力合计 · N"), family: "force", index: 0),
                      StudioSignal(id: "rtf", name: L("实际实时倍率"), family: "rtf", index: 0)]
        for joint in scene?.joints ?? [] {
            values.append(StudioSignal(id: "q\(joint.id)", name: "\(joint.name) · \(joint.unit)", family: "qpos", index: joint.qadr))
            values.append(StudioSignal(id: "v\(joint.id)", name: "\(joint.name) · \(joint.unit)/s", family: "qvel", index: joint.vadr))
        }
        for actuator in scene?.actuators ?? [] {
            values.append(StudioSignal(id: "u\(actuator.id)", name: L("\(actuator.name) · 输入"), family: "ctrl", index: actuator.id))
            values.append(StudioSignal(id: "f\(actuator.id)", name: L("\(actuator.name) · 执行器力"), family: "actuator_force", index: actuator.id))
        }
        for sensor in scene?.sensors ?? [] {
            for i in 0..<sensor.dimension {
                values.append(StudioSignal(id: "s\(sensor.id)-\(i)", name: "\(sensor.name)[\(i)]", family: "sensor", index: sensor.address+i))
            }
        }
        return values
    }
    var signal: StudioSignal { signals.first { $0.id == signalID } ?? signals[0] }
    var signalValue: Double {
        switch signal.family {
        case "qpos": return status.qpos[safe: signal.index] ?? 0
        case "qvel": return status.qvel[safe: signal.index] ?? 0
        case "ctrl": return status.ctrl[safe: signal.index] ?? 0
        case "actuator_force": return status.actuator_force[safe: signal.index] ?? 0
        case "sensor": return status.sensordata[safe: signal.index] ?? 0
        case "force": return status.contact_force
        case "rtf": return status.rtf
        default: return Double(status.contacts)
        }
    }

    func start() {
        shutdown()
        generation = UUID()
        let current = generation
        connecting = true
        failure = nil
        notice = L("正在启动仿真引擎…")
        let runtime = SidecarManager { [weak self] state in
            Task { @MainActor in
                guard let self, self.generation == current else { return }
                switch state {
                case .running(let port): self.connect(port: port, generation: current)
                case .failed(let reason): self.fail(reason)
                default: break
                }
            }
        }
        manager = runtime
        runtime.start()
        let work = DispatchWorkItem { [weak self] in
            Task { @MainActor in
                guard let self, !self.connected, self.generation == current else { return }
                self.fail(L("启动超时。请查看运行日志后重试。"))
                self.manager?.stop()
            }
        }
        timeout = work
        DispatchQueue.main.asyncAfter(deadline: .now()+30, execute: work)
    }
    func shutdown() {
        generation = UUID()
        timeout?.cancel()
        resizeWork?.cancel()
        socket?.cancel(with: .goingAway, reason: nil)
        socket = nil
        manager?.stop()
        manager = nil
        connected = false
        requestIDs.removeAll()
        completions.removeAll()
        pending = 0
    }
    private func connect(port: Int, generation current: UUID) {
        let task = session.webSocketTask(with: URL(string: "ws://127.0.0.1:\(port)")!)
        socket = task
        task.resume()
        receive(task, generation: current)
    }
    private func receive(_ task: URLSessionWebSocketTask, generation current: UUID) {
        task.receive { [weak self] result in
            Task { @MainActor in
                guard let self, self.generation == current else { return }
                switch result {
                case .failure(let error): self.fail(L("连接断开：\(error.localizedDescription)"))
                case .success(let message):
                    switch message {
                    case .data(let data): self.video.image = NSImage(data: data)
                    case .string(let text): self.handle(text)
                    @unknown default: break
                    }
                    self.receive(task, generation: current)
                }
            }
        }
    }
    private func handle(_ text: String) {
        do {
            let data = Data(text.utf8)
            let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
            switch object["type"] as? String {
            case "hello":
                guard object["version"] as? Int == 2 else { fail(L("后端协议不匹配，请重新构建应用。")); return }
                connected = true
                connecting = false
                failure = nil
                timeout?.cancel()
                engineName = object["engine"] as? String ?? "MuJoCo"
                log(L("物理引擎已连接 · \(engineName)"))
                if let size = lastSize { send("view", ["width": size.0, "height": size.1], tracked: false) }
                if let url = pendingURL { pendingURL = nil; openURL(url) }
            case "scene":
                scene = try JSONDecoder().decode(StudioScene.self, from: data)
                clearSamples()
            case "status":
                let updated = try JSONDecoder().decode(StudioStatus.self, from: data)
                if updated != status { status = updated }
                if status.record_count > previousRecordCount { recordingNeedsExport = true }
                if status.record_count == 0 && recordingNeedsExport { recordingNeedsExport = false }
                previousRecordCount = status.record_count
                if status.time < lastSampleTime { clearSamples() }
                if status.time != lastSampleTime {
                    samples.append(StudioSample(id: sampleID, time: status.time, value: signalValue))
                    sampleID += 1
                    lastSampleTime = status.time
                    if samples.count > 1200 { samples.removeFirst(samples.count-1200) }
                }
                if !status.notice.isEmpty && status.notice != lastNotice {
                    lastNotice = status.notice
                    log(status.notice)
                }
            case "ack":
                if object["command"] as? String == "export_recording" { recordingNeedsExport = false }
                if let id = object["request_id"] as? String {
                    requestIDs.remove(id)
                    let completion = completions.removeValue(forKey: id)
                    pending = requestIDs.count
                    completion?()
                }
                if let message = object["message"] as? String, !message.isEmpty { log(message) }
            case "error":
                if let id = object["request_id"] as? String {
                    requestIDs.remove(id)
                    completions.removeValue(forKey: id)
                    pending = requestIDs.count
                }
                log(object["message"] as? String ?? L("未知错误"), error: true)
                if !connected { fail(object["message"] as? String ?? L("引擎启动失败")) }
                bottomTab = 2
            default: break
            }
        } catch {
            log(L("无法解析引擎响应：\(error.localizedDescription)"), error: true)
        }
    }
    private func fail(_ reason: String) {
        connected = false
        connecting = false
        failure = reason
        status.running = false
        requestIDs.removeAll()
        completions.removeAll()
        pending = 0
        log(reason, error: true)
    }
    func log(_ text: String, error: Bool = false) {
        notice = text
        logs.append(StudioLog(text: text, error: error))
        if logs.count > 250 { logs.removeFirst() }
    }
    func send(_ command: String, _ fields: [String: Any] = [:], tracked: Bool = true, completion: (() -> Void)? = nil) {
        guard connected, let socket else { return }
        var payload = fields
        payload["cmd"] = command
        let id = UUID().uuidString
        if tracked {
            payload["request_id"] = id
            requestIDs.insert(id)
            if let completion { completions[id] = completion }
            pending = requestIDs.count
        }
        do {
            let data = try JSONSerialization.data(withJSONObject: payload)
            socket.send(.string(String(decoding: data, as: UTF8.self))) { [weak self] error in
                if let error {
                    Task { @MainActor in
                        self?.requestIDs.remove(id)
                        self?.completions.removeValue(forKey: id)
                        self?.pending = self?.requestIDs.count ?? 0
                        self?.log(L("命令发送失败：\(error.localizedDescription)"), error: true)
                    }
                }
            }
        } catch { requestIDs.remove(id); completions.removeValue(forKey: id); pending = requestIDs.count; log(error.localizedDescription, error: true) }
    }
    func clearSamples() { samples.removeAll(); lastSampleTime = -1; sampleID = 0 }
    func select(_ id: Int) { send("select", ["id": id], tracked: false) }
    func toggleRun() { send(status.running ? "pause" : "run") }
    func toggleRecord() {
        if !status.recording && status.record_count > 0 {
            let alert = NSAlert()
            alert.messageText = L("开始新的录制？")
            alert.informativeText = L("当前内存中的录制将被替换。需要保留时，请先导出录制文件。")
            alert.addButton(withTitle: L("开始新录制"))
            alert.addButton(withTitle: L("取消"))
            guard alert.runModal() == .alertFirstButtonReturn else { return }
        }
        send(status.recording ? "record_stop" : "record_start")
    }
    func setSize(_ size: CGSize) {
        let scale = min(2.0, 1920/max(size.width, 1), 1200/max(size.height, 1))
        let w = max(320, Int(size.width*scale)/16*16)
        let h = max(240, Int(size.height*scale)/16*16)
        if let lastSize, lastSize == (w,h) { return }
        lastSize = (w,h)
        resizeWork?.cancel()
        let work = DispatchWorkItem { [weak self] in
            Task { @MainActor in self?.send("view", ["width": w, "height": h], tracked: false) }
        }
        resizeWork = work
        DispatchQueue.main.asyncAfter(deadline: .now()+0.2, execute: work)
    }
    func updateView() { send("view", ["contacts": showContacts, "joints": showJoints, "collision": collisionView], tracked: false) }
    func replaceWorld(_ action: @escaping () -> Void) {
        guard editable else { log(L("请先暂停仿真并结束录制／回放。"), error: true); return }
        guard status.dirty || recordingNeedsExport else { action(); return }
        let alert = NSAlert()
        alert.messageText = L("保存当前项目的更改？")
        alert.informativeText = recordingNeedsExport ? L("未导出的录制将被替换。项目保存不包含录制；需要保留录制时请取消并先导出。") : L("未保存的场景编辑和物理状态将被替换。")
        alert.addButton(withTitle: L("保存"))
        alert.addButton(withTitle: L("取消"))
        alert.addButton(withTitle: L("不保存"))
        switch alert.runModal() {
        case .alertFirstButtonReturn: saveProject(completion: action)
        case .alertThirdButtonReturn: action()
        default: break
        }
    }
    func newWorld(_ builtin: String = "world") { replaceWorld { self.send("load", ["builtin": builtin]) } }
    func openURL(_ url: URL) {
        guard url.isFileURL else { return }
        if !connected { pendingURL = url; return }
        let command: String
        switch url.pathExtension.lowercased() {
        case "mjstudio": command = "open_project"
        case "mjrec": command = "open_recording"
        case "xml", "urdf": command = "load"
        default: log(L("不支持的文件格式"), error: true); return
        }
        replaceWorld { self.send(command, ["path": url.path]) }
    }
    func openExample() {
        let bundled = Bundle.main.resourceURL?.appendingPathComponent("projects/机器人实验场景.mjstudio")
        let local = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("examples/projects/机器人实验场景.mjstudio")
        let url = bundled.flatMap { FileManager.default.fileExists(atPath: $0.path) ? $0 : nil } ?? local
        openURL(url)
    }
    func importModel() {
        replaceWorld {
            let panel = NSOpenPanel()
            panel.title = L("导入 MuJoCo / URDF 模型")
            panel.allowedContentTypes = [.xml, UTType(filenameExtension: "urdf") ?? .data]
            panel.canChooseDirectories = false
            guard panel.runModal() == .OK, let url = panel.url else { return }
            self.send("load", ["path": url.path])
        }
    }
    func insertModel() {
        let panel = NSOpenPanel()
        panel.title = L("插入模型到当前世界（保留世界物理设置）")
        panel.allowedContentTypes = [.xml, UTType(filenameExtension: "urdf") ?? .data]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        send("insert_model", ["path": url.path])
    }
    func loadTrajectory() {
        let panel = NSOpenPanel()
        panel.title = L("加载执行器轨迹 CSV")
        panel.allowedContentTypes = [.commaSeparatedText]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        send("load_trajectory", ["path": url.path])
    }
    func openProject() {
        replaceWorld {
            let panel = NSOpenPanel()
            panel.title = L("打开仿真项目")
            panel.allowedContentTypes = [UTType(filenameExtension: "mjstudio") ?? .data]
            guard panel.runModal() == .OK, let url = panel.url else { return }
            self.send("open_project", ["path": url.path])
        }
    }
    func saveProject(asNew: Bool = false, completion: (() -> Void)? = nil) {
        let bundled = Bundle.main.resourcePath.map { status.project_path.hasPrefix($0 + "/") } ?? false
        if !asNew && !bundled && !status.project_path.isEmpty {
            send("save_project", ["path": status.project_path], completion: completion)
            return
        }
        let panel = NSSavePanel()
        panel.title = L("保存仿真项目")
        panel.nameFieldStringValue = title + ".mjstudio"
        panel.allowedContentTypes = [UTType(filenameExtension: "mjstudio") ?? .data]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        send("save_project", ["path": url.path], completion: completion)
    }
    func export(_ csv: Bool) {
        let panel = NSSavePanel()
        let ext = csv ? "csv" : "mjrec"
        panel.nameFieldStringValue = title + "." + ext
        panel.allowedContentTypes = [UTType(filenameExtension: ext) ?? .data]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        send(csv ? "export_csv" : "export_recording", ["path": url.path])
    }
    func openRecording() {
        replaceWorld {
            let panel = NSOpenPanel()
            panel.allowedContentTypes = [UTType(filenameExtension: "mjrec") ?? .data]
            guard panel.runModal() == .OK, let url = panel.url else { return }
            self.send("open_recording", ["path": url.path])
        }
    }
    func confirmRemove() {
        guard let body = selectedBody, body.id > 0 else { return }
        let alert = NSAlert()
        alert.messageText = L("删除 \(body.name)？")
        alert.informativeText = L("将删除该物体及其子节点。此操作可以撤销。")
        alert.addButton(withTitle: L("删除"))
        alert.addButton(withTitle: L("取消"))
        if alert.runModal() == .alertFirstButtonReturn { send("remove") }
    }
}

extension Array {
    subscript(safe index: Int) -> Element? { indices.contains(index) ? self[index] : nil }
}
