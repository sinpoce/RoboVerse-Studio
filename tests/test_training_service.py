import asyncio
import io
import json
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image
import websockets

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.environ.get("ROBOVERSE_STAGEWISE_PROJECT", Path.home()/"Downloads/Stage-Wise-CMORL-main"))


@unittest.skipUnless((PROJECT/"algos/comoppo/agent.py").is_file(), "Stage-Wise source fixture not available")
class TrainingServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.checkpoint = Path(self.directory.name)/"policy.pt"
        self.process = await asyncio.create_subprocess_exec(sys.executable, str(ROOT/"examples/generic_train_server.py"),
            "--port", "0", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        line = await asyncio.wait_for(self.process.stdout.readline(), 20)
        port = int(line.decode().split()[-1])
        self.ws = await websockets.connect(f"ws://127.0.0.1:{port}")
        self.frames = []
        await self.until("training_config")
        await self.send("task_config", path=str(PROJECT))
        msg = await self.until("training_config", lambda m: "project_comoppo" in m["algo"])
        self.config = json.loads(msg["config_json"])
        self.assertEqual(Path(msg["config_path"]), PROJECT.resolve())
        self.url = f"ws://127.0.0.1:{port}"

    async def asyncTearDown(self):
        await self.ws.close()
        self.process.terminate()
        await asyncio.wait_for(self.process.wait(), 10)
        self.directory.cleanup()

    async def send(self, cmd, **values):
        await self.ws.send(json.dumps({"cmd": cmd, **values}))

    async def until(self, kind, predicate=lambda m: True):
        async def read():
            while True:
                raw = await self.ws.recv()
                if isinstance(raw, bytes):
                    self.frames.append(raw)
                    continue
                message = json.loads(raw)
                if message["type"] == "training_config": self.settings = message
                if message["type"] == "error" and kind != "error": raise AssertionError(message)
                if message["type"] == kind and predicate(message): return message
        return await asyncio.wait_for(read(), 30)

    async def configure(self, iterations):
        self.config.update(num_envs=2, horizon=32, iterations=iterations)
        await self.send("algorithm_config", config_json=json.dumps(self.config), checkpoint=str(self.checkpoint))
        await self.until("training_config")
        await self.send("configure", num_sims=2)
        await self.until("configured")

    async def test_import_train_stream_real_frames_and_save(self):
        self.config.update(num_envs=2, horizon=32, iterations=2)
        await self.send("start_training", config_json=json.dumps(self.config), checkpoint=str(self.checkpoint))
        await self.until("log", lambda m: "训练完成" in m.get("message", ""))
        state = await self.until("status", lambda m: not m.get("running"))
        self.assertEqual(state["iteration"], 2)
        self.assertGreater(state["values"]["steps/s (throughput)"], 0)
        self.assertTrue(self.checkpoint.is_file())
        self.assertTrue(self.frames)
        self.assertEqual(Image.open(io.BytesIO(self.frames[0])).size, (720, 480))

    async def test_invalid_workbench_start_stays_idle_then_recovers(self):
        self.config.update(num_envs=0, horizon=32, iterations=1)
        await self.send("start_training", config_json=json.dumps(self.config), checkpoint=str(self.checkpoint))
        error = await self.until("error")
        self.assertIn("并行环境数", error["message"])
        state = await self.until("status")
        self.assertFalse(state["running"])
        self.assertFalse(self.checkpoint.exists())
        self.config["num_envs"] = 2
        await self.send("start_training", config_json=json.dumps(self.config), checkpoint=str(self.checkpoint))
        await self.until("log", lambda m: "训练完成" in m.get("message", ""))
        metadata = json.loads((Path(self.directory.name)/"policy.run/experiment.json").read_text())
        self.assertEqual(metadata["total_steps"], 64)

    async def test_browser_origin_cannot_launch_local_training(self):
        with self.assertRaises(websockets.exceptions.InvalidStatus) as rejected:
            async with websockets.connect(self.url, origin="https://example.org"):
                pass
        self.assertEqual(rejected.exception.response.status_code, 403)

    async def test_load_evaluate_checkpoint_without_updating_policy(self):
        self.config.update(num_envs=2, horizon=32, iterations=2)
        await self.send("start_training", config_json=json.dumps(self.config), checkpoint=str(self.checkpoint))
        await self.until("log", lambda m: "训练完成" in m.get("message", ""))
        await self.until("status", lambda m: not m.get("running"))
        before = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        await self.send("load_checkpoint", path=str(self.checkpoint))
        settings = await self.until("training_config")
        def reject_nonstandard(value): raise ValueError(f"Nonstandard JSON: {value}")
        restored = json.loads(settings["config_json"], parse_constant=reject_nonstandard)
        self.assertEqual(restored["model_num"], 128)
        self.assertEqual(restored["adapter"], "stagewise_mujoco")
        self.assertEqual(restored["algorithm_overrides"]["model"]["reward_critic"]["clip_range"], ["-np.inf", "np.inf"])
        await self.send("evaluate_policy", checkpoint=str(self.checkpoint), episodes=1, realtime=False)
        result = await self.until("evaluation_result")
        self.assertEqual(result["episodes"], 1)
        self.assertEqual(result["success_rate"], 0)
        report = json.loads(Path(result["report_path"]).read_text())
        self.assertEqual(report["total_steps"], 128)
        self.assertFalse(report["results"][0]["success"])
        self.assertEqual(before, hashlib.sha256(self.checkpoint.read_bytes()).hexdigest())
        self.assertTrue(self.frames)
        sys.path.insert(0, str(ROOT / "python"))
        from studio_world import World
        world = World()
        try:
            world.open(result["recording_path"], recording=True)
            self.assertEqual(world.model.nu, 12)
            self.assertGreater(len(world.replay_times), 1)
            world.seek(len(world.replay_times)-1)
            trace = json.loads((Path(result["report_path"]).parent / "episode-1.json").read_text())
            import numpy as np
            np.testing.assert_allclose(world.data.qpos, trace[-1]["qpos"], atol=1e-12)
        finally:
            world.close()

        await self.until("status", lambda m: not m.get("running"))
        self.config["model_num"] = 0
        await self.send("start_training", config_json=json.dumps(self.config), checkpoint=str(self.checkpoint))
        overwrite = await self.until("error")
        self.assertIn("检查点已存在", overwrite["message"])
        self.assertEqual(before, hashlib.sha256(self.checkpoint.read_bytes()).hexdigest())

        await self.until("status", lambda m: not m.get("running"))
        await self.send("algorithm", name="train_project_comoppo_h1_backflip")
        await self.until("training_config")
        await self.send("load_checkpoint", path=str(self.checkpoint))
        error = await self.until("error")
        self.assertIn("机器人任务", error["message"])

    async def test_stop_saves_and_rejects_algorithm_change_during_training(self):
        await self.configure(1000)
        await self.send("run")
        await self.until("status", lambda m: m.get("iteration", 0) >= 1)
        await self.send("algorithm", name="train_project_comoppo_h1_backflip")
        error = await self.until("error")
        self.assertIn("尚未停止", error["message"])
        await self.send("pause")
        await self.until("log", lambda m: "已停止，检查点" in m.get("message", ""))
        state = await self.until("status", lambda m: not m.get("running"))
        self.assertEqual(state["algo"], "train_project_comoppo_go1_backflip")
        self.assertTrue(self.checkpoint.is_file())
        saved = json.loads((Path(self.directory.name)/"policy.run/experiment.json").read_text())
        self.config = json.loads(self.settings["config_json"])
        self.assertEqual(self.config["model_num"], saved["total_steps"])
        self.assertTrue((Path(self.directory.name)/f"policy.run/checkpoint/model_{self.config['model_num']}.pt").is_file())
        await self.configure(1)
        await self.send("run")
        await self.until("log", lambda m: "训练完成" in m.get("message", ""))
        resumed = json.loads((Path(self.directory.name)/"policy.run/experiment.json").read_text())
        self.assertEqual(resumed["resumed_from"], saved["total_steps"])
        self.assertEqual(resumed["total_steps"], saved["total_steps"] + 64)
