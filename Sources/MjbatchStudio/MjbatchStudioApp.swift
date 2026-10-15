// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import AppKit

/// When launched as a bare SPM executable (`swift run`) rather than a bundled
/// .app, the process starts as an accessory. Force a normal, foreground app so
/// the window appears and takes focus.
final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate {
    weak var studio: StudioModel?
    weak var trainingModel: TrainingModel?
    var allowTermination = false
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
    func windowShouldClose(_ sender: NSWindow) -> Bool {
        NSApp.terminate(nil)
        return false
    }
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard !allowTermination, let studio, studio.connected, studio.status.dirty || studio.status.recording || studio.recordingNeedsExport else { return .terminateNow }
        let alert = NSAlert()
        alert.messageText = L("保存项目后退出？")
        alert.informativeText = studio.status.recording || studio.recordingNeedsExport
            ? L("项目保存当前场景与状态。录制数据需要单独导出，退出后未导出的录制会丢失。")
            : L("当前项目有未保存的场景或物理状态。")
        alert.addButton(withTitle: L("保存并退出"))
        alert.addButton(withTitle: L("取消"))
        alert.addButton(withTitle: L("直接退出"))
        switch alert.runModal() {
        case .alertFirstButtonReturn:
            studio.send("pause", completion: { [weak self, weak studio] in
                studio?.saveProject(completion: {
                    self?.allowTermination = true
                    NSApp.terminate(nil)
                })
            })
            return .terminateCancel
        case .alertThirdButtonReturn: return .terminateNow
        default: return .terminateCancel
        }
    }
    func applicationWillTerminate(_ notification: Notification) {
        trainingModel?.shutdown()
        studio?.shutdown()
    }
}

@main
struct RoboVerseStudioApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var appLanguage = AppLanguage.shared
    @StateObject private var studio = StudioModel()
    @StateObject private var trainingModel = TrainingModel()

    var body: some Scene {
        Window("RoboVerse Studio", id: "workbench") {
            WorkbenchView().environmentObject(studio).environmentObject(trainingModel)
                .onOpenURL { studio.openURL($0) }
                .onAppear {
                    appDelegate.studio = studio
                    appDelegate.trainingModel = trainingModel
                    studio.start()
                    if ProcessInfo.processInfo.environment["MJBATCH_STUDIO_SERVER"]?.hasSuffix("generic_train_server.py") == true {
                        studio.trainingWorkspace = true
                        trainingModel.start()
                    }
                    DispatchQueue.main.async { NSApp.windows.first(where: { $0.isVisible })?.delegate = appDelegate }
                }
            .environment(\.locale, Locale(identifier: appLanguage.selected.rawValue))
            .frame(minWidth: 1120, minHeight: 780)
        }
        .windowStyle(.titleBar)
        .defaultSize(width: 1440, height: 940)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button(L("新建世界")) { studio.newWorld() }.keyboardShortcut("n").disabled(!studio.editable)
                Button(L("打开项目…")) { studio.openProject() }.keyboardShortcut("o").disabled(!studio.editable)
                Button(L("导入模型…")) { studio.importModel() }.keyboardShortcut("i").disabled(!studio.editable)
            }
            CommandGroup(replacing: .saveItem) {
                Button(L("保存项目")) { studio.saveProject() }.keyboardShortcut("s").disabled(!studio.connected)
                Button(L("项目另存为…")) { studio.saveProject(asNew: true) }.keyboardShortcut("s", modifiers: [.command, .shift]).disabled(!studio.connected)
            }
            CommandGroup(replacing: .undoRedo) {
                Button(L("撤销场景编辑")) { studio.send("undo") }.keyboardShortcut("z").disabled(!studio.editable || !studio.status.can_undo)
                Button(L("重做场景编辑")) { studio.send("redo") }.keyboardShortcut("z", modifiers: [.command, .shift]).disabled(!studio.editable || !studio.status.can_redo)
            }
            CommandMenu(L("仿真")) {
                Button(studio.status.running ? L("暂停") : L("运行")) { studio.toggleRun() }.keyboardShortcut("r").disabled(!studio.connected)
                Button(L("单步")) { studio.send("step") }.keyboardShortcut(".").disabled(!studio.connected || studio.status.running)
                Button(L("重置")) { studio.send("reset") }.disabled(!studio.editable)
                Divider()
                Button(L("打开录制…")) { studio.openRecording() }.disabled(!studio.editable)
                Button(L("导出 CSV…")) { studio.export(true) }.disabled(studio.status.recording || studio.status.record_count == 0)
            }
            CommandMenu(L("语言")) {
                Picker(L("界面语言"), selection: $appLanguage.selected) {
                    ForEach(StudioLanguage.allCases) { Text($0.title).tag($0) }
                }
            }
            CommandMenu(L("训练")) {
                Button(L("打开训练工作区")) { studio.trainingWorkspace = true }
                    .keyboardShortcut("t", modifiers: [.command, .shift])
            }
        }
    }
}
