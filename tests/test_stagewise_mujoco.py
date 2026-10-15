"""Acceptance against an unmodified Stage-Wise checkout, when available.

Set ROBOVERSE_STAGEWISE_PROJECT to run these tests on another machine.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from stagewise_mujoco import create_env, TASKS, quat_rotate, quat_rotate_inverse

PROJECT = Path(os.environ.get("ROBOVERSE_STAGEWISE_PROJECT", Path.home() / "Downloads/Stage-Wise-CMORL-main"))


@unittest.skipUnless((PROJECT / "algos/comoppo/agent.py").is_file(), "Stage-Wise source fixture not available")
class StageWiseAcceptanceTests(unittest.TestCase):
    def make_env(self, name="go1_backflip", episode=None):
        path = PROJECT / "tasks" / (name + ".yaml")
        cfg = yaml.safe_load(path.read_text())
        if episode is not None: cfg["env"]["learn"]["episode_length_s"] = episode
        return create_env(PROJECT, path.with_suffix(".py"), cfg, 2, 3)

    def test_six_original_tasks_step_finite_with_expected_dimensions(self):
        for name in sorted(TASKS):
            with self.subTest(task=name):
                env = self.make_env(name)
                env.reset(is_uniform_rollout=False)
                before = env.datas[0].qpos.copy()
                for _ in range(20):
                    obs, reward, done, info = env.step(torch.zeros(2, env.num_actions))
                self.assertEqual(obs["obs"].shape, (2, env.raw_obs_dim * 3))
                self.assertEqual(reward.shape, (2, env.num_rewards))
                self.assertEqual(info["costs"].shape, (2, env.num_costs))
                self.assertTrue(torch.isfinite(reward).all())
                self.assertTrue(torch.isfinite(obs["obs"]).all())
                self.assertFalse(np.array_equal(before, env.datas[0].qpos))
                self.assertAlmostEqual(env.datas[0].time, 0.4)
                self.assertNotIn("isaacgym", sys.modules)

    def test_ground_contact_separates_foot_from_calf(self):
        env = self.make_env()
        env.reset(is_uniform_rollout=False)
        env.start_time_buf.fill_(5)
        for _ in range(30): env.step(torch.zeros(2, 12))
        self.assertGreater(float(env.contact_forces[:, env.foot_indices].norm(dim=-1).sum()), 100)
        self.assertEqual(float(env.contact_forces[:, env.calf_indices].norm(dim=-1).sum()), 0)

    def test_action_range_matches_upstream_unbounded_actor(self):
        env = self.make_env()
        env.reset(is_uniform_rollout=False)
        env.step(torch.full((2, 12), 2.5))
        self.assertTrue(np.isinf(env.clip_actions))
        torch.testing.assert_close(env.prev_actions, torch.full((2, 12), 2.5))

    def test_timeout_preserves_terminal_state_before_reset(self):
        env = self.make_env(episode=0.04)
        env.reset(is_uniform_rollout=False)
        env.step(torch.ones(2, 12)*0.1)
        obs, _, _, info = env.step(torch.ones(2, 12)*0.1)
        self.assertTrue(torch.all(info["dones"] == 1))
        self.assertTrue(torch.all(env.progress_buf == 0))
        self.assertFalse(torch.equal(obs["obs"], info["next_obs"]))

    def test_original_agent_updates_and_writes_reloadable_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy.pt"
            process = subprocess.run([sys.executable, str(ROOT / "examples/train_custom.py"),
                "--config", str(PROJECT), "--num-envs", "2", "--iterations", "2",
                "--checkpoint", str(checkpoint)], capture_output=True, text=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertTrue({"actor", "reward_critic", "cost_critic", "actor_optimizer"} <= payload.keys())
            metadata = json.loads((Path(directory)/"policy.run/experiment.json").read_text())
            self.assertEqual(metadata["total_steps"], 512)
            self.assertGreater(metadata["actor_parameter_delta"], 0)
            self.assertEqual(metadata["backend"], "MuJoCo")
            self.assertTrue((Path(directory)/"policy.run/obs_scale/512.pkl").is_file())

            config = {"project_root": str(PROJECT), "task_config": str(PROJECT/"tasks/go1_backflip.yaml"),
                      "algorithm_config": str(PROJECT/"algos/comoppo/go1_backflip.yaml"),
                      "num_envs": 2, "horizon": 32, "iterations": 1, "model_num": 512}
            config_path = Path(directory)/"config.json"

            def run_config(output):
                config_path.write_text(json.dumps(config))
                child = subprocess.run([sys.executable, str(ROOT/"examples/stagewise_train.py"),
                    "--config", str(config_path), "--checkpoint", str(output)],
                    capture_output=True, text=True, timeout=60)
                self.assertEqual(child.returncode, 0, child.stderr + child.stdout)
                return json.loads(child.stdout.splitlines()[-1])["result"]

            with self.subTest(workflow="resume"):
                resumed = run_config(checkpoint)
                self.assertEqual(resumed["total_steps"], 576)
                self.assertGreater(resumed["actor_parameter_delta"], 0)
                meta = json.loads((Path(directory)/"policy.run/experiment.json").read_text())
                self.assertEqual(meta["resumed_from"], 512)
                after = torch.load(checkpoint, map_location="cpu", weights_only=True)
                optimizer_step = lambda p: next(iter(p["actor_optimizer"]["state"].values()))["step"]
                self.assertGreater(optimizer_step(after), optimizer_step(payload))

            with self.subTest(workflow="student_imitation"):
                config.update(algorithm_config=str(PROJECT/"algos/student/go1_backflip.yaml"), model_num=0,
                              teacher_run=str(Path(directory)/"policy.run"),
                              algorithm_overrides={"batch_size": 32, "replay_size": 1024, "n_actor_iters": 2})
                student_path = Path(directory)/"student.pt"
                student = run_config(student_path)
                self.assertEqual(student["total_steps"], 64)
                self.assertGreater(student["actor_parameter_delta"], 0)
                self.assertTrue(np.isfinite(student["terms"]["actor_loss"]))
                self.assertEqual(set(torch.load(student_path, weights_only=True)), {"actor", "actor_optimizer"})


class QuaternionTests(unittest.TestCase):
    def test_xyzw_rotation_and_inverse(self):
        q = torch.tensor([[0., 0., 2**-0.5, 2**-0.5]])
        v = torch.tensor([[1., 0., 0.]])
        torch.testing.assert_close(quat_rotate(q, v), torch.tensor([[0., 1., 0.]]), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(quat_rotate_inverse(q, quat_rotate(q, v)), v, atol=1e-6, rtol=1e-6)
