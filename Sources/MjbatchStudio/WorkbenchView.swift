import AppKit
import Charts
import SwiftUI

let studioBackground = Color(red: 0.075, green: 0.085, blue: 0.105)
let studioPanel = Color(red: 0.105, green: 0.12, blue: 0.145)
let studioAccent = Color(red: 0.24, green: 0.78, blue: 0.88)

struct WorkbenchView: View {
    @ObservedObject private var appLanguage = AppLanguage.shared
    @EnvironmentObject var studio: StudioModel
    @EnvironmentObject var training: TrainingModel

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if studio.trainingWorkspace {
                TrainingWorkspaceView()
            } else {
              HSplitView {
                sceneBrowser.frame(minWidth: 205, idealWidth: 235, maxWidth: 280)
                VStack(spacing: 0) {
                    simulationBar
                    Divider()
                    StudioViewport(video: studio.video).frame(minHeight: 280)
                    Divider()
                    recordingBar
                    Divider()
                    diagnostics.frame(height: 205)
                }
                .frame(minWidth: 480, maxWidth: .infinity)
                StudioInspector().frame(minWidth: 295, idealWidth: 310, maxWidth: 350)
              }
            Divider()
            footer
            }
        }
        .background(studioBackground)
        .preferredColorScheme(.dark)
        .tint(studioAccent)
        .overlay { if !studio.trainingWorkspace { connectionOverlay } }
        .onChange(of: studio.trainingWorkspace) { active in
            if active { training.start() }
        }
    }

    private var header: some View {
        HStack(spacing: 14) {
            Image(systemName: "cube.transparent.fill").font(.title2).foregroundStyle(studioAccent)
            VStack(alignment: .leading, spacing: 2) {
                Text("ROBOVERSE STUDIO").font(.system(size: 11, weight: .bold, design: .monospaced)).tracking(2)
                HStack(spacing: 5) {
                    Text(L(studio.trainingWorkspace ? training.projectTitle : studio.title)).font(.system(size: 13, weight: .medium)).lineLimit(1)
                    if !studio.trainingWorkspace && studio.status.dirty { Circle().fill(.orange).frame(width: 5, height: 5).help(L("项目有未保存的更改")) }
                }
            }
            Spacer()
            Picker(L("工作区"), selection: $studio.trainingWorkspace) {
                Text(L("场景")).tag(false)
                Text(L("训练")).tag(true)
            }.pickerStyle(.segmented).labelsHidden().frame(width: 150)
            LanguagePicker().fixedSize()
            if !studio.trainingWorkspace {
            Menu {
                Button(L("新建世界")) { studio.newWorld() }
                Button(L("打开项目…")) { studio.openProject() }
                Button(L("导入模型…")) { studio.importModel() }
                Button(L("打开机器人实验场景")) { studio.openExample() }
                Divider()
                Button(L("打开录制…")) { studio.openRecording() }
            } label: { Label(L("打开"), systemImage: "folder") }
            .fixedSize().disabled(!studio.editable)
            Button { studio.saveProject() } label: { Label(L("保存"), systemImage: "square.and.arrow.down") }
                .disabled(!studio.connected || studio.pending > 0)
            Button { studio.send("undo") } label: { Image(systemName: "arrow.uturn.backward") }
                .disabled(!studio.editable || !studio.status.can_undo).help(L("撤销场景编辑"))
            Button { studio.send("redo") } label: { Image(systemName: "arrow.uturn.forward") }
                .disabled(!studio.editable || !studio.status.can_redo).help(L("重做场景编辑"))
            Divider().frame(height: 22)
            Text(studio.engineName).font(.system(size: 10, design: .monospaced)).foregroundStyle(.secondary)
            } else {
                Text(training.running ? L("训练进行中") : L("本地训练工作区"))
                    .font(.system(size: 10, design: .monospaced)).foregroundStyle(.secondary)
            }
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
        .padding(.horizontal, 16).padding(.vertical, 11)
        .background(studioPanel)
    }

    private var sceneBrowser: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                sectionTitle(L("场景"), "SCENE")
                Spacer()
                Menu {
                    Button(L("插入模型文件…")) { studio.insertModel() }
                    Button(L("插入 Unitree G1")) { studio.send("insert_model", ["builtin": "g1"]) }
                    Divider()
                    Button(L("动态立方体")) { studio.send("add", ["shape": "box"]) }
                    Button(L("动态球体")) { studio.send("add", ["shape": "sphere"]) }
                    Button(L("动态圆柱")) { studio.send("add", ["shape": "cylinder"]) }
                    Divider()
                    Button(L("静态障碍物")) { studio.send("add", ["shape": "box", "dynamic": false, "position": [0.7, 0, 0.2], "size": [0.2, 0.4, 0.2]]) }
                } label: { Image(systemName: "plus") }
                .menuStyle(.borderlessButton).fixedSize().disabled(!studio.editable)
            }.padding(14)
            Divider()
            ScrollView {
                LazyVStack(spacing: 2) {
                    ForEach(studio.scene?.bodies ?? []) { body in
                        Button { studio.select(body.id) } label: {
                            HStack(spacing: 7) {
                                Image(systemName: body.id == 0 ? "globe" : body.dynamic ? "cube" : "square.3.layers.3d")
                                    .foregroundStyle(body.id == studio.status.selected ? studioAccent : .secondary)
                                Text(body.id == 0 ? L("世界") : body.name).lineLimit(1)
                                Spacer(minLength: 0)
                                if body.id == studio.status.selected { Circle().fill(studioAccent).frame(width: 4, height: 4) }
                            }
                            .font(.system(size: 12))
                            .padding(.leading, CGFloat(min(body.depth, 6))*10+10)
                            .padding(.trailing, 9).padding(.vertical, 7)
                            .contentShape(Rectangle())
                            .background(body.id == studio.status.selected ? studioAccent.opacity(0.13) : .clear)
                            .clipShape(RoundedRectangle(cornerRadius: 5))
                        }
                        .buttonStyle(.plain)
                    }
                }.padding(7)
            }
            Divider()
            VStack(alignment: .leading, spacing: 12) {
                sectionTitle(L("模型库"), "LIBRARY")
                libraryButton("Unitree G1", subtitle: L("29 自由度人形机器人"), icon: "figure.stand", model: "g1")
                libraryButton(L("倒立摆"), subtitle: L("欠驱动控制实验"), icon: "move.3d", model: "cartpole")
                libraryButton(L("双摆"), subtitle: L("多体动力学"), icon: "point.3.connected.trianglepath.dotted", model: "acrobot")
                Button { studio.importModel() } label: { Label(L("导入 XML / URDF…"), systemImage: "square.and.arrow.down.on.square") }
                    .font(.caption).disabled(!studio.editable)
            }.padding(14)
            Divider()
            HStack {
                Text(L("\(studio.scene?.bodies.count ?? 0) 个节点"))
                Spacer()
                Text(L("\(studio.scene?.nu ?? 0) 个执行器"))
            }.font(.system(size: 10, design: .monospaced)).foregroundStyle(.secondary).padding(12)
        }.background(studioPanel.opacity(0.7))
    }
    private func libraryButton(_ name: String, subtitle: String, icon: String, model: String) -> some View {
        Button { studio.newWorld(model) } label: {
            HStack(spacing: 10) {
                Image(systemName: icon).font(.title3).frame(width: 26).foregroundStyle(studioAccent)
                VStack(alignment: .leading, spacing: 3) {
                    Text(name).font(.system(size: 12, weight: .medium))
                    Text(L(subtitle)).font(.system(size: 10)).foregroundStyle(.secondary)
                }
                Spacer()
                Image(systemName: "arrow.up.right").font(.system(size: 9)).foregroundStyle(.secondary)
            }.padding(10).background(.white.opacity(0.035), in: RoundedRectangle(cornerRadius: 7))
        }.buttonStyle(.plain).disabled(!studio.editable)
    }
    private var simulationBar: some View {
        HStack(spacing: 12) {
            Button { studio.toggleRun() } label: {
                Label(studio.status.running ? L("暂停") : studio.status.replaying ? L("播放") : L("运行"), systemImage: studio.status.running ? "pause.fill" : "play.fill")
                    .frame(width: 62)
            }.buttonStyle(.borderedProminent)
            Button { studio.send("step") } label: { Image(systemName: "forward.frame") }
                .disabled(studio.status.running).help(L("推进一个物理步"))
            Menu {
                Button(L("模型初始状态")) { studio.send("reset") }
                ForEach(Array((studio.scene?.keyframes ?? []).enumerated()), id: \.offset) { index, name in
                    Button(name) { studio.send("reset", ["keyframe": index]) }
                }
            } label: { Image(systemName: "arrow.counterclockwise") }
                .fixedSize().disabled(studio.status.recording || studio.status.replaying)
            Divider().frame(height: 18)
            Picker(L("速度"), selection: Binding(get: { studio.status.speed }, set: { studio.send("speed", ["value": $0]) })) {
                Text("0.1×").tag(0.1); Text("0.25×").tag(0.25); Text("0.5×").tag(0.5)
                Text(L("实时 1×")).tag(1.0); Text("2×").tag(2.0); Text("5×").tag(5.0); Text("10×").tag(10.0)
            }.labelsHidden().frame(width: 92)
            Spacer()
            Text(String(format: "%08.3f s", studio.status.time)).font(.system(size: 13, weight: .medium, design: .monospaced))
        }
        .controlSize(.small).buttonStyle(.borderless)
        .padding(.horizontal, 14).padding(.vertical, 9)
        .disabled(!studio.connected || studio.pending > 0)
        .background(studioPanel.opacity(0.7))
    }
    private var recordingBar: some View {
        HStack(spacing: 12) {
            Button { studio.toggleRecord() } label: {
                Label(studio.status.recording ? L("停止录制") : L("录制"), systemImage: studio.status.recording ? "stop.circle.fill" : "record.circle")
                    .foregroundStyle(studio.status.recording ? .red : .primary)
            }.disabled(studio.status.replaying)
            if studio.status.replaying {
                Slider(value: Binding(get: { Double(studio.status.replay_index) }, set: { studio.send("seek", ["index": Int($0)], tracked: false) }),
                       in: 0...Double(max(1, studio.status.replay_count-1)), step: 1)
                Text("\(studio.status.replay_index+1) / \(studio.status.replay_count)").monospacedDigit()
                Button(L("退出回放")) { studio.send("replay_exit") }
            } else {
                Text(L("\(studio.status.record_count) 帧")).foregroundStyle(.secondary).monospacedDigit()
                Spacer()
                Button { studio.send("replay") } label: { Label(L("回放"), systemImage: "play.rectangle") }
                    .disabled(studio.status.record_count == 0 || studio.status.recording)
            }
            Menu {
                Button(L("录制文件（含模型）…")) { studio.export(false) }
                Button(L("CSV 数据…")) { studio.export(true) }
            } label: { Label(L("导出"), systemImage: "square.and.arrow.up") }
                .fixedSize().disabled(studio.status.recording || (studio.status.record_count == 0 && !studio.status.replaying))
        }
        .font(.system(size: 11)).buttonStyle(.borderless).controlSize(.small)
        .padding(.horizontal, 14).padding(.vertical, 9).background(studioPanel)
        .disabled(!studio.connected || studio.pending > 0)
    }
    private var diagnostics: some View {
        VStack(spacing: 8) {
            HStack {
                Picker(L("监测"), selection: $studio.bottomTab) {
                    Text(L("数据曲线")).tag(0); Text(L("传感器")).tag(1); Text(L("运行日志")).tag(2)
                }.pickerStyle(.segmented).labelsHidden().frame(width: 270)
                Spacer()
                if studio.bottomTab == 0 {
                    Text(String(format: "%.4f", studio.signalValue)).font(.system(size: 12, design: .monospaced)).foregroundStyle(studioAccent)
                    Picker(L("信号"), selection: $studio.signalID) {
                        ForEach(studio.signals) { signal in Text(L(signal.name)).tag(signal.id) }
                    }.labelsHidden().frame(maxWidth: 240).onChange(of: studio.signalID) { _ in studio.clearSamples() }
                }
            }
            if studio.bottomTab == 0 {
                Chart(studio.samples) { sample in
                    LineMark(x: .value(L("仿真时间 / s"), sample.time), y: .value(studio.signal.name, sample.value))
                        .foregroundStyle(studioAccent).lineStyle(StrokeStyle(lineWidth: 1.5))
                }
                .chartXAxisLabel(L("仿真时间 / s"), alignment: .trailing)
                .chartYAxis { AxisMarks(position: .leading) }
                .overlay { if studio.samples.count < 2 { Text(L("运行仿真以查看数据 · 最近 1200 个采样点")).font(.caption).foregroundStyle(.secondary) } }
            } else if studio.bottomTab == 1 {
                if (studio.scene?.sensors ?? []).isEmpty {
                    VStack(spacing: 8) {
                        Image(systemName: "sensor.tag.radiowaves.forward").font(.title2).foregroundStyle(.secondary)
                        Text(L("模型尚未配置传感器")).font(.caption)
                        Text(L("可在物体属性中添加 IMU；已有 MJCF 传感器会自动显示在这里。"))
                            .font(.caption2).foregroundStyle(.secondary)
                    }.frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    ScrollView {
                        VStack(spacing: 5) {
                            ForEach(studio.scene?.sensors ?? []) { sensor in
                                HStack {
                                    Text(sensor.name).foregroundStyle(.secondary)
                                    Spacer()
                                    Text((0..<sensor.dimension).map { String(format: "% .4f", studio.status.sensordata[safe: sensor.address+$0] ?? 0) }.joined(separator: "   "))
                                }.font(.system(size: 11, design: .monospaced)).padding(.vertical, 3)
                            }
                        }
                    }
                }
            } else {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 6) {
                            ForEach(studio.logs) { entry in
                                HStack(alignment: .top, spacing: 10) {
                                    Text(entry.date, style: .time).foregroundStyle(.secondary)
                                    Text(L(entry.text)).foregroundStyle(entry.error ? .orange : .primary).textSelection(.enabled)
                                }.font(.system(size: 11, design: .monospaced)).id(entry.id)
                            }
                        }.frame(maxWidth: .infinity, alignment: .leading)
                    }.onChange(of: studio.logs.count) { _ in if let id = studio.logs.last?.id { proxy.scrollTo(id, anchor: .bottom) } }
                }
            }
        }.padding(12).background(studioBackground)
    }
    private var footer: some View {
        HStack(spacing: 12) {
            Circle().fill(studio.connected ? studioAccent : .orange).frame(width: 6, height: 6)
            Text(L(studio.notice)).lineLimit(1)
            Spacer()
            if studio.status.warnings > 0 { Label(L("\(studio.status.warnings) 条物理警告"), systemImage: "exclamationmark.triangle").foregroundStyle(.orange) }
            Text(L("接触 \(studio.status.contacts)"))
            Text(String(format: "RTF %.2f×", studio.status.rtf))
            Text(String(format: "Δt %.2f ms", (studio.scene?.timestep ?? 0)*1000))
        }.font(.system(size: 10, design: .monospaced)).foregroundStyle(.secondary)
            .padding(.horizontal, 14).padding(.vertical, 7).background(studioPanel)
    }
    @ViewBuilder private var connectionOverlay: some View {
        if !studio.connected {
            ZStack {
                studioBackground.opacity(0.92)
                VStack(spacing: 16) {
                    if studio.connecting { ProgressView().controlSize(.large) }
                    else { Image(systemName: "exclamationmark.triangle").font(.largeTitle).foregroundStyle(.orange) }
                    Text(studio.connecting ? L("正在准备仿真世界") : L("仿真引擎未连接")).font(.title3.weight(.medium))
                    if let reason = studio.failure {
                        Text(L(reason)).font(.system(.caption, design: .monospaced)).textSelection(.enabled).frame(maxWidth: 540)
                        Button(L("重新连接")) { studio.start() }.buttonStyle(.borderedProminent)
                    } else { Text(L("加载物理引擎与本地渲染器")).foregroundStyle(.secondary).font(.caption) }
                }.padding(30)
            }
        }
    }
}

private func sectionTitle(_ title: String, _ subtitle: String) -> some View {
    HStack(spacing: 8) {
        Text(L(title)).font(.system(size: 12, weight: .semibold))
        Text(L(subtitle)).font(.system(size: 9, design: .monospaced)).foregroundStyle(.tertiary).tracking(1)
    }
}

struct StudioViewport: View {
    @ObservedObject private var appLanguage = AppLanguage.shared
    @EnvironmentObject var studio: StudioModel
    @ObservedObject var video: StudioVideo
    @StateObject private var wheel = ScrollWheelInput()
    @State private var dragLast = CGSize.zero
    @State private var pinchLast = 1.0
    var body: some View {
        GeometryReader { geo in
            ZStack {
                Color.black.opacity(0.2)
                if let frame = video.image {
                    Image(nsImage: frame).resizable().interpolation(.high).aspectRatio(contentMode: .fit)
                }
                VStack {
                    HStack(spacing: 8) {
                        HStack(spacing: 6) {
                            Circle().fill(studio.status.running ? .green : studioAccent).frame(width: 5, height: 5)
                            Text(studio.status.replaying ? L("回放") : studio.status.running ? L("仿真运行中") : L("编辑模式"))
                        }.font(.system(size: 10, weight: .medium)).padding(7).background(.ultraThinMaterial, in: Capsule())
                        if studio.status.recording { Text("● REC").font(.system(size: 10, design: .monospaced)).foregroundStyle(.red) }
                        Spacer()
                        Menu {
                            Toggle(L("接触点与接触力"), isOn: $studio.showContacts).onChange(of: studio.showContacts) { _ in studio.updateView() }
                            Toggle(L("关节轴"), isOn: $studio.showJoints).onChange(of: studio.showJoints) { _ in studio.updateView() }
                            Toggle(L("碰撞几何组"), isOn: $studio.collisionView).onChange(of: studio.collisionView) { _ in studio.updateView() }
                            Divider()
                            Button(L("聚焦所选物体")) { studio.send("camera", ["focus": true], tracked: false) }
                            Button(L("重置相机")) { studio.send("camera", ["reset": true], tracked: false) }
                        } label: { Label(L("视图"), systemImage: "eye") }
                        .menuStyle(.borderlessButton).fixedSize().padding(7).background(.ultraThinMaterial, in: Capsule())
                    }
                    Spacer()
                    HStack {
                        Text(L("拖动旋转 · Shift 拖动平移 · 滚动缩放 · 点击选择"))
                            .font(.system(size: 10)).foregroundStyle(.secondary)
                            .padding(7).background(.ultraThinMaterial, in: Capsule())
                        Spacer()
                        Button { studio.send("camera", ["focus": true], tracked: false) } label: { Image(systemName: "scope") }
                            .buttonStyle(.bordered).help(L("聚焦所选物体"))
                    }
                }.padding(13)
            }
            .background(ScrollWheelProbe(input: wheel))
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 4).onChanged { value in
                let dx = value.translation.width-dragLast.width
                let dy = value.translation.height-dragLast.height
                dragLast = value.translation
                if NSEvent.modifierFlags.contains(.shift) {
                    studio.send("camera", ["pan": [dx/max(geo.size.width,1), dy/max(geo.size.height,1)]], tracked: false)
                } else { studio.send("camera", ["orbit": [-dx*0.35, dy*0.35]], tracked: false) }
            }.onEnded { _ in dragLast = .zero })
            .simultaneousGesture(MagnificationGesture().onChanged { value in
                studio.send("camera", ["zoom": pinchLast/Double(value)], tracked: false)
                pinchLast = Double(value)
            }.onEnded { _ in pinchLast = 1 })
            .simultaneousGesture(SpatialTapGesture().onEnded { event in
                guard let image = video.image else { return }
                let scale = min(geo.size.width/image.size.width, geo.size.height/image.size.height)
                let width = image.size.width*scale, height = image.size.height*scale
                let x = (event.location.x-(geo.size.width-width)/2)/width
                let y = (event.location.y-(geo.size.height-height)/2)/height
                if (0...1).contains(x), (0...1).contains(y) { studio.send("pick", ["x": x, "y": y], tracked: false) }
            })
            .onAppear {
                studio.setSize(geo.size)
                wheel.onScroll = { event, _, _ in
                    let delta = event.isDirectionInvertedFromDevice ? -event.scrollingDeltaY : event.scrollingDeltaY
                    studio.send("camera", ["zoom": exp(-Double(delta)*0.025)], tracked: false)
                }
                wheel.onDoubleClick = { studio.send("camera", ["focus": true], tracked: false) }
                wheel.start()
            }
            .onDisappear { wheel.stop() }
            .onChange(of: geo.size) { studio.setSize($0) }
        }
    }
}

struct StudioInspector: View {
    @ObservedObject private var appLanguage = AppLanguage.shared
    @EnvironmentObject var studio: StudioModel
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            sectionTitle(L("检查器"), "INSPECTOR").padding(14)
            Picker(L("检查器"), selection: $studio.inspectorTab) {
                Text(L("物体")).tag(0); Text(L("关节")).tag(1); Text(L("控制")).tag(2); Text(L("物理")).tag(3)
            }.pickerStyle(.segmented).labelsHidden().padding(.horizontal, 10).padding(.bottom, 12)
            Divider()
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 16) {
                    if studio.inspectorTab == 0 {
                        if let body = studio.selectedBody, body.id != 0 {
                            StudioBodyEditor(entity: body).id("\(body.id)-\(studio.scene?.revision ?? 0)")
                        } else {
                            Text(L("世界设置")).font(.headline)
                            Text(L("从场景树或视口中选择物体，即可编辑属性、添加传感器或施加外力。"))
                                .font(.caption).foregroundStyle(.secondary)
                            detailRow(L("模型"), studio.scene?.name ?? "—")
                            detailRow(L("广义坐标"), "\(studio.scene?.nq ?? 0)")
                            detailRow(L("自由度"), "\(studio.scene?.nv ?? 0)")
                            detailRow(L("执行器"), "\(studio.scene?.nu ?? 0)")
                            detailRow(L("传感器"), "\(studio.scene?.sensors.count ?? 0)")
                        }
                    } else if studio.inspectorTab == 1 {
                        Text(L("关节位置编辑")).font(.headline)
                        Text(L("暂停时设置初始姿态；运行中的实时关节数据会持续更新。"))
                            .font(.caption2).foregroundStyle(.secondary)
                        if (studio.scene?.joints ?? []).isEmpty { Text(L("当前模型没有标量关节")).font(.caption).foregroundStyle(.secondary) }
                        ForEach(studio.scene?.joints ?? []) { joint in StudioJointRow(joint: joint) }
                    } else if studio.inspectorTab == 2 {
                        Text(L("执行器控制")).font(.headline)
                        Text(L("原生输入遵循模型执行器定义。PD 模式仅对兼容电机开放，目标为关节位置。"))
                            .font(.caption2).foregroundStyle(.secondary)
                        if (studio.scene?.actuators ?? []).isEmpty { Text(L("当前模型没有执行器")).font(.caption).foregroundStyle(.secondary) }
                        if let trajectory = studio.status.trajectory, !trajectory.isEmpty {
                            Text(L("轨迹：") + trajectory).font(.caption).foregroundStyle(studioAccent)
                            Button(L("停止轨迹控制")) { studio.send("stop_trajectory") }
                        } else if !(studio.scene?.actuators ?? []).isEmpty {
                            Button(L("加载控制轨迹 CSV…")) { studio.loadTrajectory() }.disabled(studio.status.replaying)
                            Text(L("列：time_s、ctrl_执行器名。时间从 0 开始；结束时保持最后输入。"))
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                        ForEach(studio.scene?.actuators ?? []) { actuator in StudioActuatorRow(actuator: actuator).id("\(actuator.id)-\(studio.scene?.revision ?? 0)") }
                    } else {
                        if let scene = studio.scene { StudioPhysicsEditor(scene: scene).id(scene.revision) }
                    }
                }.padding(14)
            }
            Spacer(minLength: 0)
            if !studio.editable {
                Label(studio.status.replaying ? L("退出回放后可编辑场景") : studio.status.recording ? L("停止录制后可编辑场景") : studio.status.running ? L("暂停后可编辑场景") : L("正在处理操作"), systemImage: "lock")
                    .font(.caption2).foregroundStyle(.secondary).padding(12)
            }
        }.background(studioPanel.opacity(0.7))
    }
}
private func detailRow(_ label: String, _ value: String) -> some View {
    HStack(alignment: .top) {
        Text(L(label)).foregroundStyle(.secondary)
        Spacer()
        Text(L(value)).multilineTextAlignment(.trailing).textSelection(.enabled)
    }.font(.system(size: 11, design: .monospaced))
}
private struct StudioBodyEditor: View {
    @ObservedObject private var appLanguage = AppLanguage.shared
    @EnvironmentObject var studio: StudioModel
    let entity: StudioBody
    @State private var x = 0.0
    @State private var y = 0.0
    @State private var z = 0.0
    @State private var qw = 1.0
    @State private var qx = 0.0
    @State private var qy = 0.0
    @State private var qz = 0.0
    @State private var transformEdited = false
    @State private var mass = 1.0
    @State private var friction = 0.8
    @State private var fx = 10.0
    @State private var fy = 0.0
    @State private var fz = 0.0
    var bodyView: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label(entity.name, systemImage: "cube").font(.headline).textSelection(.enabled)
            detailRow(L("节点 ID"), "\(entity.id)")
            detailRow(L("几何体"), "\(entity.geoms)")
            Divider()
            Text(L("位置 · m")).font(.caption.weight(.semibold))
            axisFields(x: transformBinding($x), y: transformBinding($y), z: transformBinding($z)).disabled(!studio.editable)
            Text(L("旋转四元数 · W / X / Y / Z")).font(.caption.weight(.semibold))
            HStack(spacing: 5) {
                axisField("W", value: transformBinding($qw), color: .secondary)
                axisField("X", value: transformBinding($qx), color: .red)
                axisField("Y", value: transformBinding($qy), color: .green)
                axisField("Z", value: transformBinding($qz), color: .blue)
            }.disabled(!studio.editable)
            Text(L("自由物体使用世界坐标；其他节点使用父节点坐标。"))
                .font(.system(size: 10)).foregroundStyle(.secondary)
            numericField(L("质量 · kg"), value: $mass).disabled(!studio.editable || entity.mass <= 0)
            numericField(L("滑动摩擦系数"), value: $friction).disabled(!studio.editable || entity.geoms == 0)
            Button(L("应用物体属性")) {
                var fields: [String: Any] = ["id": entity.id, "position": [x,y,z], "quaternion": [qw,qx,qy,qz]]
                if entity.mass > 0 { fields["mass"] = mass }
                if entity.geoms > 0 { fields["friction"] = friction }
                studio.send("body_edit", fields)
            }.buttonStyle(.borderedProminent).disabled(!studio.editable)
            Divider()
            Text(L("外力 · N")).font(.caption.weight(.semibold))
            axisFields(x: $fx, y: $fy, z: $fz)
            Button(L("施力 0.2 秒")) { studio.send("force", ["id": entity.id, "force": [fx,fy,fz], "duration": 0.2]) }
                .disabled(studio.status.replaying)
            Text(L("以仿真时间计时；暂停时施力将在运行后生效。"))
                .font(.system(size: 10)).foregroundStyle(.secondary)
            Divider()
            Button { studio.send("add_imu", ["id": entity.id]) } label: { Label(L("添加 IMU 传感器"), systemImage: "sensor.tag.radiowaves.forward") }
                .disabled(!studio.editable)
            Button(role: .destructive) { studio.confirmRemove() } label: { Label(L("删除物体"), systemImage: "trash") }
                .disabled(!studio.editable)
        }
        .onAppear { loadValues() }
        .onChange(of: studio.status.running) { running in if !running { loadValues() } }
        .onChange(of: studio.status.selected_position) { _ in if !transformEdited && !studio.status.running { loadTransform() } }
        .onChange(of: studio.status.selected_quaternion) { _ in if !transformEdited && !studio.status.running { loadTransform() } }
        .onChange(of: studio.status.replaying) { _ in loadValues() }
    }
    var body: some View { bodyView }
    private func transformBinding(_ binding: Binding<Double>) -> Binding<Double> {
        Binding(get: { binding.wrappedValue }, set: { binding.wrappedValue = $0; transformEdited = true })
    }
    private func loadTransform() {
        let pos = studio.status.selected_position ?? self.entity.position
        x = pos[safe: 0] ?? 0; y = pos[safe: 1] ?? 0; z = pos[safe: 2] ?? 0
        let q = studio.status.selected_quaternion ?? entity.quaternion
        qw = q[safe: 0] ?? 1; qx = q[safe: 1] ?? 0; qy = q[safe: 2] ?? 0; qz = q[safe: 3] ?? 0
    }
    private func loadValues() {
        loadTransform()
        transformEdited = false
        mass = self.entity.mass; friction = self.entity.friction
    }
}
private func numericField(_ label: String, value: Binding<Double>) -> some View {
    HStack {
        Text(L(label)).font(.system(size: 11)).foregroundStyle(.secondary)
        Spacer()
        TextField(L(label), value: value, format: .number.precision(.fractionLength(0...4)))
            .textFieldStyle(.roundedBorder).font(.system(size: 11, design: .monospaced)).frame(width: 95)
    }
}
private func axisFields(x: Binding<Double>, y: Binding<Double>, z: Binding<Double>) -> some View {
    HStack(spacing: 6) {
        axisField("X", value: x, color: .red)
        axisField("Y", value: y, color: .green)
        axisField("Z", value: z, color: .blue)
    }
}
private func axisField(_ name: String, value: Binding<Double>, color: Color) -> some View {
    HStack(spacing: 3) {
        Text(name).foregroundStyle(color).font(.system(size: 10, weight: .semibold))
        TextField(name, value: value, format: .number.precision(.fractionLength(0...3)))
            .font(.system(size: 11, design: .monospaced)).textFieldStyle(.roundedBorder)
    }
}
private struct StudioJointRow: View {
    @ObservedObject private var appLanguage = AppLanguage.shared
    @EnvironmentObject var studio: StudioModel
    let joint: StudioJoint
    @State private var value = 0.0
    @State private var dragging = false
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(joint.name).font(.system(size: 11, weight: .medium)).lineLimit(1)
            HStack {
                Text(String(format: "%.4f %@", studio.status.qpos[safe: joint.qadr] ?? 0, joint.unit))
                Spacer()
                Text(String(format: "%.3f %@/s", studio.status.qvel[safe: joint.vadr] ?? 0, joint.unit)).foregroundStyle(.secondary)
            }.font(.system(size: 10, design: .monospaced))
            Slider(value: $value, in: min(joint.lower,joint.upper-0.0001)...max(joint.upper,joint.lower+0.0001), onEditingChanged: { active in
                dragging = active
                if !active { studio.send("joint", ["id": joint.id, "value": value]) }
            }).disabled(!studio.editable)
        }.padding(10).background(.white.opacity(0.035), in: RoundedRectangle(cornerRadius: 7))
            .onAppear { value = studio.status.qpos[safe: joint.qadr] ?? 0 }
            .onChange(of: studio.status.qpos) { _ in if !dragging { value = studio.status.qpos[safe: joint.qadr] ?? 0 } }
    }
}
private struct StudioActuatorRow: View {
    @ObservedObject private var appLanguage = AppLanguage.shared
    @EnvironmentObject var studio: StudioModel
    let actuator: StudioActuator
    @State private var mode = "direct"
    @State private var target = 0.0
    @State private var kp = 30.0
    @State private var kd = 3.0
    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack {
                Text(actuator.name).font(.system(size: 11, weight: .medium)).lineLimit(1)
                Spacer()
                Text(actuator.kind.uppercased()).font(.system(size: 8, design: .monospaced)).foregroundStyle(studioAccent)
            }
            Picker(L("模式"), selection: $mode) {
                Text(L("原生输入")).tag("direct")
                if actuator.pd { Text(L("PD 位置控制")).tag("pd") }
                Text(L("零输入")).tag("off")
            }.font(.caption)
            if mode != "off" { numericField(mode == "pd" ? L("目标位置") : L("控制输入"), value: $target) }
            if mode == "pd" {
                numericField("Kp", value: $kp)
                numericField("Kd", value: $kd)
            }
            detailRow(L("实际输入"), String(format: "%.3f", studio.status.ctrl[safe: actuator.id] ?? 0))
            detailRow(L("执行器力"), String(format: "%.3f", studio.status.actuator_force[safe: actuator.id] ?? 0))
            Button(L("应用控制")) { studio.send("control", ["id": actuator.id, "mode": mode, "target": target, "kp": kp, "kd": kd]) }
                .disabled(studio.status.replaying || !studio.connected)
        }.padding(10).background(.white.opacity(0.035), in: RoundedRectangle(cornerRadius: 7))
            .onAppear {
                if let control = studio.status.controllers[actuator.name] {
                    mode = control.mode; target = control.target; kp = control.kp; kd = control.kd
                } else { target = studio.status.ctrl[safe: actuator.id] ?? 0 }
            }
            .onChange(of: studio.status.controllers) { _ in
                if let control = studio.status.controllers[actuator.name] {
                    mode = control.mode; target = control.target; kp = control.kp; kd = control.kd
                } else { mode = "direct"; target = studio.status.ctrl[safe: actuator.id] ?? 0 }
            }
    }
}
private struct StudioPhysicsEditor: View {
    @ObservedObject private var appLanguage = AppLanguage.shared
    @EnvironmentObject var studio: StudioModel
    let scene: StudioScene
    @State private var gx = 0.0
    @State private var gy = 0.0
    @State private var gz = -9.81
    @State private var timestep = 0.002
    @State private var iterations = 100.0
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(L("物理世界")).font(.headline)
            Text(L("重力 · m/s²")).font(.caption.weight(.semibold))
            axisFields(x: $gx, y: $gy, z: $gz)
            numericField(L("物理步长 · s"), value: $timestep)
            numericField(L("求解器迭代次数"), value: $iterations)
            Text(L("较小步长通常提高数值稳定性，也会增加计算量。实时倍率显示实际仿真速度。"))
                .font(.caption2).foregroundStyle(.secondary)
            Button(L("应用物理参数")) {
                studio.send("physics", ["gravity": [gx,gy,gz], "timestep": timestep, "iterations": iterations])
            }.buttonStyle(.borderedProminent)
            Divider()
            Text(L("记录与复现")).font(.caption.weight(.semibold))
            Text(L("项目保存完整场景资源、物理状态和控制参数。录制以 50 Hz 保存状态，支持逐帧回放和 CSV 导出；达到内存上限时自动停止。"))
                .font(.caption2).foregroundStyle(.secondary)
        }.disabled(!studio.editable)
            .onAppear { gx = scene.gravity[0]; gy = scene.gravity[1]; gz = scene.gravity[2]; timestep = scene.timestep; iterations = Double(scene.iterations) }
    }
}
