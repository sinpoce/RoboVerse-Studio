import Foundation
import SwiftUI

enum StudioLanguage: String, CaseIterable, Identifiable {
    case english = "en"
    case chinese = "zh-Hans"
    var id: String { rawValue }
    var title: String { self == .english ? "English" : "简体中文" }
}

/// UI language is independent of the system language and never restarts a world.
final class AppLanguage: ObservableObject {
    static let shared = AppLanguage()
    @Published var selected: StudioLanguage {
        didSet { defaults.set(selected.rawValue, forKey: "RoboVerseStudio.language") }
    }
    private let defaults: UserDefaults
    private let catalog: [String: String]
    private let reverse: [String: String]
    private let englishRules: [TranslationRule]
    private let chineseRules: [TranslationRule]

    init(defaults: UserDefaults = .standard, catalogURL: URL? = nil) {
        self.defaults = defaults
        selected = StudioLanguage(rawValue: defaults.string(forKey: "RoboVerseStudio.language") ?? "en") ?? .english
        var url = catalogURL ?? Bundle.main.url(forResource: "translations", withExtension: "json")
        #if SWIFT_PACKAGE
        if url == nil { url = Bundle.module.url(forResource: "translations", withExtension: "json") }
        #endif
        let values = url.flatMap { try? Data(contentsOf: $0) }
            .flatMap { try? JSONDecoder().decode([String: String].self, from: $0) } ?? [:]
        catalog = values
        reverse = values.keys.sorted().reduce(into: [:]) { result, key in
            if result[values[key]!] == nil { result[values[key]!] = key }
        }
        englishRules = values.map { TranslationRule(source: $0.key, target: $0.value) }
            .sorted { $0.source.count > $1.source.count }
        chineseRules = values.map { TranslationRule(source: $0.value, target: $0.key) }
            .sorted { $0.source.count > $1.source.count }
    }

    func text(_ source: String) -> String { translate(source, depth: 0) }

    private func translate(_ source: String, depth: Int) -> String {
        guard depth < 5 else { return source }
        let table = selected == .english ? catalog : reverse
        if let value = table[source] { return value }
        let rules = selected == .english ? englishRules : chineseRules
        for rule in rules {
            if let values = rule.match(source) {
                var value = rule.target
                for (token, captured) in values {
                    value = value.replacingOccurrences(of: token, with: translate(captured, depth: depth+1))
                }
                return value
            }
        }
        if source.contains("\n") {
            return source.components(separatedBy: "\n").map { translate($0, depth: depth+1) }.joined(separator: "\n")
        }
        for prefix in ["⚠︎ ", "⚠️ ", "ValueError: ", "RuntimeError: ", "FileNotFoundError: ", "FileExistsError: ", "training failed: "] {
            if source.hasPrefix(prefix) { return prefix + translate(String(source.dropFirst(prefix.count)), depth: depth+1) }
        }
        for rule in rules where !rule.source.contains("{0}") && (rule.source.hasSuffix("：") || rule.source.hasSuffix(": ")) {
            if source.hasPrefix(rule.source) { return rule.target + translate(String(source.dropFirst(rule.source.count)), depth: depth+1) }
        }
        return source
    }
}

private struct TranslationRule {
    let source: String
    let target: String
    let expression: NSRegularExpression?
    let tokens: [String]

    init(source: String, target: String) {
        // printf forms also appear as already-formatted status messages.
        func template(_ value: String) -> String {
            let regex = try! NSRegularExpression(pattern: #"%(?:[0-9]+\$)?[-+0 #]*[0-9]*(?:\.[0-9]+)?[diuoxXfFeEgGaAcsp@%]"#)
            var result = value, index = 0
            let matches = regex.matches(in: value, range: NSRange(value.startIndex..., in: value))
            var replacements: [(NSRange, String)] = []
            for match in matches {
                let text = (value as NSString).substring(with: match.range)
                if text == "%%" { replacements.append((match.range, "%")) }
                else { replacements.append((match.range, "{\(index)}")); index += 1 }
            }
            for (range, replacement) in replacements.reversed() {
                result = (result as NSString).replacingCharacters(in: range, with: replacement)
            }
            return result
        }
        self.source = template(source)
        self.target = template(target)
        let tokenPattern = try! NSRegularExpression(pattern: #"\{[0-9]+\}"#)
        let matches = tokenPattern.matches(in: self.source, range: NSRange(self.source.startIndex..., in: self.source))
        let text = self.source as NSString
        var pattern = "^", cursor = 0, tokens: [String] = []
        for match in matches {
            pattern += NSRegularExpression.escapedPattern(for: text.substring(with: NSRange(location: cursor, length: match.range.location-cursor)))
            pattern += "([\\s\\S]*?)"
            tokens.append(text.substring(with: match.range))
            cursor = NSMaxRange(match.range)
        }
        pattern += NSRegularExpression.escapedPattern(for: text.substring(from: cursor)) + "$"
        self.tokens = tokens
        expression = tokens.isEmpty ? nil : try? NSRegularExpression(pattern: pattern)
    }

    func match(_ value: String) -> [(String, String)]? {
        guard let expression, let match = expression.firstMatch(in: value, range: NSRange(value.startIndex..., in: value)) else { return nil }
        return tokens.enumerated().map { ($0.element, (value as NSString).substring(with: match.range(at: $0.offset+1))) }
    }
}

func L(_ source: String) -> String { AppLanguage.shared.text(source) }

struct LanguagePicker: View {
    @ObservedObject private var language = AppLanguage.shared
    var body: some View {
        Menu {
            ForEach(StudioLanguage.allCases) { item in
                Button { language.selected = item } label: {
                    if language.selected == item {
                        Label(item.title, systemImage: "checkmark")
                    } else { Text(item.title) }
                }
                .accessibilityIdentifier("app.language.\(item.rawValue)")
            }
        } label: { Label(language.selected.title, systemImage: "globe") }
        .accessibilityIdentifier("app.language")
        .help(L("语言更改立即生效，仿真与训练不会重新启动。"))
    }
}
