import json
from pathlib import Path
import tempfile
import unittest

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from algorithm_api import configured_algorithms, config_for_algorithm  # noqa: E402
from import_project import scan_project  # noqa: E402


class AlgorithmPluginTests(unittest.TestCase):
    def test_custom_algorithm_is_discovered_with_own_config(self):
        raw = {
            "algorithm": "algorithms.random_search:train",
            "algorithms": ["ppo", "algorithms.random_search:train"],
            "algorithm_config": {
                "train_random_search": {"iterations": 3},
            },
        }
        algorithms = configured_algorithms(raw)
        self.assertEqual([item.algorithm_id for item in algorithms], ["ppo_train", "train_random_search"])
        custom = algorithms[1]
        self.assertEqual(config_for_algorithm(raw, custom), {"iterations": 3})
        raw["algorithm"] = "train_random_search"
        self.assertEqual([a.algorithm_id for a in configured_algorithms(raw)], ["ppo_train", "train_random_search"])

    def test_existing_project_manifest_discovers_tasks_models_and_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tasks").mkdir()
            (root / "algos" / "comoppo").mkdir(parents=True)
            (root / "assets").mkdir()
            (root / "main_teacher.py").write_text("# external\n", encoding="utf-8")
            (root / "tasks" / "go1_backflip.yaml").write_text("name: Go1Backflip\n", encoding="utf-8")
            (root / "algos" / "comoppo" / "go1_backflip.yaml").write_text("n_steps: 4\n", encoding="utf-8")
            (root / "assets" / "go1.xml").write_text("<mujoco/>\n", encoding="utf-8")
            manifest = scan_project(root)
            self.assertEqual(manifest["schema"], "roboverse.project.v1")
            self.assertEqual(len(manifest["tasks"]), 1)
            self.assertEqual(len(manifest["models"]), 1)
            self.assertEqual(manifest["algorithms"][0]["spec"], "project_runner:train")
            self.assertEqual(manifest["algorithms"][0]["project_root"], str(root.resolve()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
