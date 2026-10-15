import AppKit
import Charts
import SwiftUI
import UniformTypeIdentifiers

/// Integrated task configuration, training, and policy evaluation.
struct TrainingWorkspaceView: View {
    @ObservedObject private var appLanguage = AppLanguage.shared
    @EnvironmentObject var model: TrainingModel
    @EnvironmentObject var studio: StudioModel
    @State private var bottomTab = 0
    @State private var advanced = false

    private var connected: Bool { model.phase == .connected }
    private var metrics: [String] {
        Array(Set(model.metrics + model.trainTerms.map(\.id))).sorted()
    }

    var body: some View {
        VStack(spacing: 0) {
            HSplitView {
                projectBrowser.frame(minWidth: 205, idealWidth: 235, maxWidth: 280)
                VStack(spacing: 0) {
                    trainingBar
                    Divider()
                    viewport.frame(minHeight: 280)
                    Divider()
                    progressStrip
                    Divider()
                    diagnostics.frame(height: 205)
                }.frame(minWidth: 480, maxWidth: .infinity)
                inspector.frame(minWidth: 295, idealWidth: 310, maxWidth: 350)
            }
            Divider()
            HStack(spacing: 10) {
                Circle().fill(connected ? studioAccent : .orange).frame(width: 6, height: 6)
                Text(L(model.statusMessage)).lineLimit(1).textSelection(.enabled)
                Spacer(minLength: 8)
                Text(L("MuJoCo · 本机")).foregroundStyle(studioAccent)
            }
            .font(.system(size: 10, design: .monospaced)).foregroundStyle(.secondary)
            .padding(.horizontal, 14).padding(.vertical, 7).background(studioPanel)
        }
        .overlay { connectionOverlay }
    }

    private var projectBrowser: some View {
        VStack(alignment: .leading, spacing: 0) {
            title(L("训练项目"), "PROJECT").padding(14)
            Divider()
            VStack(alignment: .leading, spacing: 12) {
                Label(L(model.projectTitle), systemImage: "folder.fill")
                    .font(.system(size: 12, weight: .medium)).foregroundStyle(studioAccent)
                    .lineLimit(2)
                Text(URL(fileURLWithPath: model.taskConfigPath).lastPathComponent).font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.secondary).lineLimit(3).textSelection(.enabled).help(model.taskConfigPath)
                Button { importProject() } label: { Label(L("导入已有项目…"), systemImage: "folder.badge.plus") }
                    .accessibilityIdentifier("training.importProject")
                Button { importConfig() } label: { Label(L("加载任务配置…"), systemImage: "doc.badge.gearshape") }
            }
            .buttonStyle(.bordered).controlSize(.small).padding(14)
            .disabled(!connected || model.trainingBusy)
            Divider()
            title(L("任务与算法"), "EXPERIMENTS").padding(14)
            ScrollView {
                LazyVStack(spacing: 5) {
                    ForEach(model.algorithms) { algorithm in
                        Button { model.setAlgorithm(algorithm.id) } label: {
                            HStack(alignment: .top, spacing: 9) {
                                Image(systemName: algorithm.id == model.selectedAlgorithm ? "circle.inset.filled" : "circle")
                                    .foregroundStyle(algorithm.id == model.selectedAlgorithm ? studioAccent : .secondary)
                                    .padding(.top, 2)
                                Text(algorithm.label).font(.system(size: 12)).multilineTextAlignment(.leading)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                            .padding(10).contentShape(Rectangle())
                            .background(algorithm.id == model.selectedAlgorithm ? studioAccent.opacity(0.13) : .white.opacity(0.025),
                                        in: RoundedRectangle(cornerRadius: 6))
                        }.buttonStyle(.plain).disabled(!connected || model.trainingBusy)
                    }
                }.padding(.horizontal, 8)
            }
            Divider()
            Text(L("\(model.algorithms.count) 个训练配置"))
                .font(.system(size: 10, design: .monospaced)).foregroundStyle(.secondary).padding(14)
        }.background(studioPanel.opacity(0.7))
    }

    private var trainingBar: some View {
        HStack(spacing: 12) {
            if model.running || model.trainingStopping {
                Button { model.stopTraining() } label: {
                    Label(model.trainingStopping ? L("正在停止…") : model.evaluating ? L("停止评估") : L("停止并保存"), systemImage: "stop.fill")
                }.disabled(model.trainingStopping)
                    .accessibilityIdentifier("training.stop")
            } else {
                Button { model.startTraining() } label: {
                    Label(model.trainingStarting ? L("正在启动…") : L("开始训练"), systemImage: "play.fill")
                }.disabled(!connected || model.trainingStarting || model.algorithms.isEmpty)
                    .accessibilityIdentifier("training.start")
            }
            Button { model.evaluatePolicy() } label: {
                Label(L("评估策略"), systemImage: "checkmark.circle")
            }.disabled(!connected || model.trainingBusy || !model.canEvaluate)
                .accessibilityIdentifier("training.evaluate")
            Spacer()
            Text(model.algorithmTitle).font(.system(size: 11, weight: .medium)).foregroundStyle(.secondary).lineLimit(1)
        }
        .buttonStyle(.borderedProminent).controlSize(.small)
        .padding(.horizontal, 14).padding(.vertical, 9).background(studioPanel.opacity(0.7))
    }

    private var viewport: some View {
        ZStack {
            Color.black.opacity(0.25)
            if let frame = model.frame {
                Image(nsImage: frame).resizable().interpolation(.high).aspectRatio(contentMode: .fit)
                    .accessibilityLabel(L("训练环境实时仿真画面"))
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "cube.transparent").font(.system(size: 42)).foregroundStyle(studioAccent.opacity(0.55))
                    Text(model.trainingStarting ? L("正在加载机器人与物理环境") : L("等待训练环境画面"))
                        .font(.system(size: 13)).foregroundStyle(.secondary)
                }
            }
            VStack {
                HStack {
                    HStack(spacing: 6) {
                        Circle().fill(model.running ? .green : studioAccent).frame(width: 5, height: 5)
                        Text(model.trainingStopping ? L("正在停止") : model.running ? (model.evaluating ? L("独立策略评估") : L("训练运行中")) : L("训练工作区"))
                    }
                    .font(.system(size: 10, weight: .medium)).padding(8)
                    .background(.ultraThinMaterial, in: Capsule())
                    Spacer()
                    if model.frame != nil {
                        Text("ENV 00 · MuJoCo").font(.system(size: 10, design: .monospaced))
                            .padding(8).background(.ultraThinMaterial, in: Capsule())
                    }
                }
                Spacer()
                if model.modelInfo.nu > 0 {
                    HStack {
                        Text(model.modelInfo.name).lineLimit(1)
                        Spacer()
                        Text(L("\(model.modelInfo.nu) 个执行器"))
                    }.font(.system(size: 10, design: .monospaced)).foregroundStyle(.secondary)
                        .padding(8).background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 6))
                }
            }.padding(12)
        }.frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var progressStrip: some View {
        HStack(spacing: 16) {
            Text(L("迭代 \(model.trainIteration) / \(max(model.trainIterations, 1))"))
                .accessibilityIdentifier("training.iteration")
            ProgressView(value: Double(model.trainIteration), total: Double(max(1, model.trainIterations, model.trainIteration)))
                .frame(maxWidth: 170)
            Spacer(minLength: 0)
            Text(String(format: "%.0f steps/s", model.stepsPerSecond))
            Text(String(format: "%02d:%02d", Int(model.trainElapsed)/60, Int(model.trainElapsed)%60))
        }
        .font(.system(size: 10, design: .monospaced)).foregroundStyle(.secondary)
        .padding(.horizontal, 14).padding(.vertical, 10).background(studioPanel)
    }

    private var inspector: some View {
        VStack(alignment: .leading, spacing: 0) {
            title(L("训练设置"), "INSPECTOR").padding(14)
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text(model.algorithmTitle).font(.system(size: 13, weight: .semibold))
                    parameterField(L("并行环境"), "num_envs", fallback: model.numSimsText)
                    parameterField(L("每次采样步数"), "horizon", fallback: "128")
                    parameterField(L("训练迭代次数"), "iterations", fallback: "1000")
                    parameterField(L("随机种子"), "seed", fallback: "1")
                    if !model.parameter("adapter").isEmpty {
                        parameterField(L("恢复检查点步数"), "model_num", fallback: "0")
                        parameterField(L("保存间隔 / 迭代"), "checkpoint_every", fallback: "25")
                    }
                    if model.canEvaluate {
                        Button(L("加载检查点…")) {
                            let panel = NSOpenPanel()
                            panel.title = L("加载策略检查点（需保留同名 .run 目录）")
                            panel.allowsMultipleSelection = false
                            if panel.runModal() == .OK, let url = panel.url { model.loadCheckpoint(url.path) }
                        }.accessibilityIdentifier("training.loadCheckpoint")
                    }
                    if !model.evaluationSummary.isEmpty {
                        Text(L(model.evaluationSummary)).font(.caption).textSelection(.enabled)
                            .accessibilityIdentifier("training.evaluationResult")
                        if !model.evaluationRecordingPath.isEmpty {
                            Button(L("在场景中回放")) {
                                studio.openURL(URL(fileURLWithPath: model.evaluationRecordingPath))
                                studio.trainingWorkspace = false
                            }.disabled(!studio.editable)
                                .accessibilityIdentifier("training.replayEvaluation")
                        }
                        if !model.evaluationReportPath.isEmpty {
                            Button(L("查看评估报告与回放")) {
                                NSWorkspace.shared.open(URL(fileURLWithPath: model.evaluationReportPath).deletingLastPathComponent())
                            }
                        }
                    }
                    if model.parameter("project_algorithm") == "student" {
                        Text(L("教师策略目录")).font(.system(size: 11)).foregroundStyle(.secondary)
                        TextField("teacher.run", text: parameterBinding("teacher_run"))
                            .font(.system(size: 11, design: .monospaced)).textFieldStyle(.roundedBorder)
                    }
                    Divider()
                    HStack {
                        Text(L("检查点文件")).font(.system(size: 11, weight: .medium))
                        Spacer()
                        Button { chooseCheckpoint() } label: { Image(systemName: "folder") }
                            .buttonStyle(.borderless).accessibilityLabel(L("选择检查点保存位置"))
                    }
                    TextField("policy.pt", text: $model.checkpointPath)
                        .textFieldStyle(.roundedBorder).font(.system(size: 11, design: .monospaced))
                        .accessibilityIdentifier("training.checkpoint")
                    DisclosureGroup(L("完整算法参数"), isExpanded: $advanced) {
                        TextEditor(text: $model.trainingParametersText)
                            .font(.system(size: 10, design: .monospaced)).frame(height: 270)
                            .scrollContentBackground(.hidden).padding(6)
                            .background(.black.opacity(0.2), in: RoundedRectangle(cornerRadius: 6))
                            .accessibilityLabel(L("算法参数 JSON"))
                    }.font(.system(size: 11, weight: .medium))
                }.padding(14).disabled(!connected || model.trainingBusy)
            }
        }.background(studioPanel.opacity(0.7))
    }

    private var diagnostics: some View {
        VStack(spacing: 10) {
            HStack {
                Picker(L("训练监测"), selection: $bottomTab) {
                    Text(L("训练曲线")).tag(0)
                    Text(L("奖励与损失")).tag(1)
                    Text(L("日志")).tag(2)
                }.pickerStyle(.segmented).labelsHidden().frame(width: 300)
                Spacer()
                if bottomTab == 0 {
                    Picker(L("指标"), selection: Binding(get: { model.metricName }, set: { model.metricName = $0; model.pushMetric() })) {
                        ForEach(metrics, id: \.self) { Text($0).tag($0) }
                    }.labelsHidden().frame(maxWidth: 200)
                }
            }
            if bottomTab == 0 {
                HStack(spacing: 20) {
                    metricChart(model.metricName, model.metricValue, model.metricSamples, studioAccent)
                    metricChart("steps/s", model.stepsPerSecond, model.throughputSamples, .green)
                }
            } else if bottomTab == 1 {
                ScrollView {
                    LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                        ForEach(model.trainTerms) { term in
                            HStack { Text(term.id).foregroundStyle(.secondary); Spacer(); Text(String(format: "%.5g", term.value)) }
                                .font(.system(size: 10, design: .monospaced)).padding(8)
                                .background(studioPanel, in: RoundedRectangle(cornerRadius: 5))
                        }
                    }
                }
            } else {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 6) {
                            ForEach(model.trainingLogs) { log in
                                HStack(alignment: .top, spacing: 8) {
                                    Text(log.date, style: .time).foregroundStyle(.secondary)
                                    Text(L(log.text)).foregroundStyle(log.error ? .orange : .primary).textSelection(.enabled)
                                }.font(.system(size: 10, design: .monospaced)).id(log.id)
                            }
                        }.frame(maxWidth: .infinity, alignment: .leading)
                    }.onChange(of: model.trainingLogs.count) { _ in
                        if let id = model.trainingLogs.last?.id { proxy.scrollTo(id, anchor: .bottom) }
                    }
                }
            }
        }.padding(12).background(studioBackground)
    }

    private func metricChart(_ name: String, _ value: Double, _ samples: [ChartSample], _ color: Color) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text(name.isEmpty ? "reward" : name).foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "%.4g", value)).foregroundStyle(color)
            }.font(.system(size: 11, design: .monospaced))
            Chart(samples) { sample in
                LineMark(x: .value(L("迭代"), sample.x), y: .value(name, sample.y))
                    .foregroundStyle(color).lineStyle(StrokeStyle(lineWidth: 1.5))
            }
            .chartYAxis { AxisMarks(position: .leading) }
            .chartXAxis { AxisMarks(values: .automatic(desiredCount: 4)) }
            .overlay {
                if samples.isEmpty { Text(L("等待训练数据")).font(.caption).foregroundStyle(.tertiary) }
            }
        }
    }

    @ViewBuilder private var connectionOverlay: some View {
        if !connected {
            ZStack {
                studioBackground.opacity(0.94)
                VStack(spacing: 16) {
                    if case .failed(let reason) = model.phase {
                        Image(systemName: "exclamationmark.triangle").font(.largeTitle).foregroundStyle(.orange)
                        Text(L("训练引擎未连接")).font(.title3)
                        Text(L(reason)).font(.system(.caption, design: .monospaced)).textSelection(.enabled).frame(maxWidth: 540)
                        Button(L("重新连接")) { model.retry() }.buttonStyle(.borderedProminent)
                    } else {
                        ProgressView().controlSize(.large)
                        Text(L("正在准备训练工作区")).font(.title3)
                    }
                }.padding(30)
            }
        }
    }

    private func parameterBinding(_ key: String, fallback: String = "") -> Binding<String> {
        Binding(get: { model.parameter(key, fallback: fallback) }, set: { model.setParameter(key, value: $0) })
    }

    private func parameterField(_ label: String, _ key: String, fallback: String) -> some View {
        HStack {
            Text(L(label)).font(.system(size: 11)).foregroundStyle(.secondary)
            Spacer()
            TextField(L(label), text: parameterBinding(key, fallback: fallback))
                .textFieldStyle(.roundedBorder).multilineTextAlignment(.trailing)
                .font(.system(size: 12, design: .monospaced)).frame(width: 96)
                .accessibilityIdentifier("training.\(key)")
        }
    }

    private func title(_ label: String, _ english: String) -> some View {
        HStack(spacing: 8) {
            Text(L(label)).font(.system(size: 12, weight: .semibold))
            Text(english).font(.system(size: 9, design: .monospaced)).foregroundStyle(.tertiary).tracking(1)
        }
    }

    private func importProject() {
        let panel = NSOpenPanel()
        panel.title = L("导入训练项目")
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = L("导入项目")
        if panel.runModal() == .OK, let url = panel.url { model.loadTaskConfig(url.path) }
    }

    private func importConfig() {
        let panel = NSOpenPanel()
        panel.title = L("加载任务与算法配置")
        panel.allowedContentTypes = [.json]
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url { model.loadTaskConfig(url.path) }
    }

    private func chooseCheckpoint() {
        let panel = NSSavePanel()
        panel.title = L("训练检查点")
        panel.nameFieldStringValue = URL(fileURLWithPath: model.checkpointPath).lastPathComponent
        panel.canCreateDirectories = true
        if panel.runModal() == .OK, let url = panel.url { model.checkpointPath = url.path }
    }
}
