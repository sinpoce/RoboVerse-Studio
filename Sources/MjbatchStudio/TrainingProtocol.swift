// SPDX-License-Identifier: Apache-2.0
// Messages exchanged with the task training service.

import Foundation

struct NamedOption: Decodable, Identifiable, Hashable {
    let id: String
    let label: String
}

struct NamedValue: Decodable, Identifiable, Hashable {
    let id: String
    let value: Double
}

struct ServerMessage: Decodable {
    let type: String
    var models: [NamedOption]?
    var metrics: [String]?
    var name: String?
    var nu: Int?
    var algorithms: [NamedOption]?
    var config_json: String?
    var checkpoint: String?
    var config_path: String?
    var algo: String?
    var num_sims: Int?
    var running: Bool?
    var activity: String?
    var steps_per_s: Double?
    var metric_value: Double?
    var iteration: Int?
    var iterations: Int?
    var elapsed: Double?
    var values: [String: Double]?
    var terms: [NamedValue]?
    var success_rate: Double?
    var episodes: Int?
    var report_path: String?
    var recording_path: String?
    var message: String?
}

struct TrainingRobotInfo: Equatable {
    var name = ""
    var nu = 0
}
