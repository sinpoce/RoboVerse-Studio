import json
import tempfile
import unittest
from pathlib import Path

from python.task_api import TaskSpec, load_task_spec


class TaskSpecTests(unittest.TestCase):
    def test_roundtrip_resolves_model_relative_to_spec(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "task.json"
            path.write_text(json.dumps({
                "task_id": "jump",
                "label": "Jump",
                "model": "robot.xml",
                "plugin": "tasks.jump:create_task",
                "reward": {"height": 2},
                "ppo": {"learning_rate": 0.0003},
            }), encoding="utf-8")
            spec = load_task_spec(path)
            self.assertEqual(spec.task_id, "jump")
            self.assertEqual(spec.model, str((root / "robot.xml").resolve()))
            self.assertEqual(spec.reward["height"], 2.0)

    def test_invalid_action_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            TaskSpec.from_dict({
                "task_id": "bad",
                "label": "Bad",
                "model": "robot.xml",
                "plugin": "tasks.bad:create_task",
                "action_mode": "keyboard",
            })
