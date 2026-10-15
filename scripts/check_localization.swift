import Foundation

@main
struct LocalizationCheck {
    static func main() {
        let suite = "RoboVerseStudio.LocalizationTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let url = URL(fileURLWithPath: CommandLine.arguments[1])
        let language = AppLanguage(defaults: defaults, catalogURL: url)
        precondition(language.selected == .english, "Fresh installs must use English")
        precondition(language.text("开始训练") == "Start Training")
        precondition(language.text("迭代 3 / 10") == "Iteration 3 / 10")
        precondition(language.text("后空翻成功率 0% · 5 回合") == "Backflip success 0% · 5 episodes")
        precondition(language.text("ValueError: 找不到策略文件：/tmp/policy.pt") == "ValueError: Policy file not found: /tmp/policy.pt")
        language.selected = .chinese
        precondition(language.text("Start Training") == "开始训练")
        precondition(language.text("Iteration 3 / 10") == "迭代 3 / 10")
        precondition(language.text("/tmp/policy.pt") == "/tmp/policy.pt")
        precondition(AppLanguage(defaults: defaults, catalogURL: url).selected == .chinese)
        language.selected = .english
        precondition(AppLanguage(defaults: defaults, catalogURL: url).selected == .english)
        print("Localization passed: English default, dynamic statuses, error messages, language persistence, and Chinese switching")
    }
}
