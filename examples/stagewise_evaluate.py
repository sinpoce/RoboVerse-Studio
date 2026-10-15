"""Reload a Stage-Wise checkpoint and measure independent, deterministic rollouts.

Backflip success requires a full backward turn, inversion, actual flight, and
at least 0.25 s of upright foot-supported landing without a body collision.
Training rewards and the task's half-turn stage flag are not success criteria.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
from types import SimpleNamespace

import mujoco
import numpy as np
from PIL import Image
import torch
import yaml

from stagewise_mujoco import create_env


class BackflipTracker:
    def __init__(self, initial_height):
        self.initial_height = initial_height
        self.rotation = self.max_rotation = self.flight = self.stable = self.best_stable = 0.0
        self.peak_height = initial_height
        self.inverted = self.body_collision = self.airborne = False

    def update(self, dt, pitch_velocity, height, up, feet, body_contact, linear_speed, angular_speed):
        self.rotation -= pitch_velocity * dt
        self.max_rotation = max(self.max_rotation, self.rotation)
        self.peak_height = max(self.peak_height, height)
        self.inverted |= up < -0.5
        self.body_collision |= body_contact
        if feet == 0:
            self.flight += dt
        else:
            self.flight = 0.0
        self.airborne |= self.flight >= 0.04 and height >= self.initial_height + 0.08
        landing = (self.max_rotation >= math.radians(330) and self.inverted and self.airborne
                   and up >= math.cos(math.radians(30)) and feet >= 2
                   and height >= self.initial_height * 0.65 and linear_speed < 1.0 and angular_speed < 1.5
                   and not self.body_collision)
        self.stable = self.stable + dt if landing else 0.0
        self.best_stable = max(self.best_stable, self.stable)

    def result(self):
        return {"success": bool(self.best_stable >= 0.25),
                "backward_rotation_deg": math.degrees(self.max_rotation),
                "peak_height_m": self.peak_height, "inverted": bool(self.inverted),
                "airborne": bool(self.airborne), "body_collision": bool(self.body_collision),
                "stable_landing_seconds": self.best_stable}


def checkpoint_metadata(checkpoint):
    checkpoint = Path(checkpoint).expanduser().resolve()
    run_dir = checkpoint.parent / (checkpoint.stem + ".run")
    if not checkpoint.is_file(): raise FileNotFoundError(f"找不到策略文件：{checkpoint}")
    meta_path = run_dir / "experiment.json"
    if not meta_path.is_file(): raise FileNotFoundError(f"缺少配套训练目录：{meta_path}")
    metadata = json.loads(meta_path.read_text())
    step = int(metadata["total_steps"])
    required = [run_dir / "checkpoint" / f"model_{step}.pt", run_dir / "obs_scale" / f"{step}.pkl"]
    if metadata["algorithm_config"]["name"].lower() == "comoppo":
        required += [run_dir / "state_scale" / f"{step}.pkl", run_dir / "reward_scale" / f"{step}.pkl"]
    for path in required:
        if not path.is_file(): raise FileNotFoundError(f"检查点不完整：{path}")
    if hashlib.sha256(checkpoint.read_bytes()).digest() != hashlib.sha256(required[0].read_bytes()).digest():
        raise ValueError("策略文件与配套 .run 检查点不一致，请恢复完整的训练输出")
    return checkpoint, run_dir, metadata


def load_policy(checkpoint, project=None):
    checkpoint, run_dir, metadata = checkpoint_metadata(checkpoint)
    project = Path(project or metadata["project_root"]).expanduser().resolve()
    task_cfg, algo_cfg = metadata["task"], dict(metadata["algorithm_config"])
    task_path = next((p for p in (project / "tasks").glob("*.yaml")
                      if yaml.safe_load(p.read_text()).get("name") == task_cfg["name"]), None)
    if task_path is None: raise ValueError("导入项目中找不到检查点对应的任务")
    family = algo_cfg["name"].lower()
    if family not in {"comoppo", "student"}: raise ValueError(f"尚不支持评估算法 {family}")
    for key, path in (("task_source_sha256", task_path.with_suffix(".py")),
                      ("algorithm_source_sha256", project / "algos" / family / "agent.py")):
        if metadata.get(key) and metadata[key] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"检查点与当前项目代码版本不一致：{path}")
    env = create_env(project, task_path.with_suffix(".py"), task_cfg, 1, algo_cfg["history_len"])
    args = dict(algo_cfg)
    args.update(name="policy_evaluation", device=torch.device("cpu"),
                obs_dim=env.num_obs, state_dim=env.num_states-env.num_stages, action_dim=env.num_actions,
                num_stages=env.num_stages, reward_dim=env.num_rewards, cost_dim=env.num_costs,
                n_envs=1, n_steps=2, save_dir=str(run_dir),
                action_bound_min=-np.ones(env.num_actions), action_bound_max=np.ones(env.num_actions),
                obs_sym_mat=env.obs_sym_mat, state_sym_mat=env.state_sym_mat, joint_sym_mat=env.joint_sym_mat)
    sys.path.insert(0, str(project))
    agent = importlib.import_module(f"algos.{family}.agent").Agent(SimpleNamespace(**args))
    agent.load(int(metadata["total_steps"]))
    agent.actor.eval()
    return env, agent, metadata, family


def evaluate(config, checkpoint, report=lambda _: None, stop=lambda: False):
    episodes = int(config.get("evaluation_episodes", 5))
    if not 1 <= episodes <= 100: raise ValueError("评估回合数必须为 1–100")
    seed = int(config.get("evaluation_seed", 1001))
    torch.set_num_threads(max(1, int(config.get("torch_threads", 2))))
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    env, agent, metadata, family = load_policy(checkpoint, config.get("project_root"))
    if "Backflip" not in metadata["task"]["name"]:
        raise ValueError("当前独立成功判据仅适用于后空翻任务")
    output = Path(config.get("evaluation_dir") or Path(checkpoint).parent / f"evaluation-{time.time_ns()}")
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    from studio_project import import_spec, save_archive
    state_flag = mujoco.mjtState.mjSTATE_INTEGRATION
    xml = output / "robot.xml"
    mujoco.mj_saveLastXML(str(xml), env.model)
    recording_spec = import_spec(str(xml))
    renderer = None
    if config.get("preview_path") or config.get("render", True):
        env.model.vis.global_.offwidth = 720
        env.model.vis.global_.offheight = 480
        renderer = mujoco.Renderer(env.model, height=480, width=720)
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = (2.5, 90, -18)
    results, images = [], []
    started = time.perf_counter()
    # Defer the task's automatic reset until after terminal physics is measured.
    reset_idx = env.reset_idx
    simulate = env.gym.simulate
    try:
        for episode in range(episodes):
            if stop(): break
            torch.manual_seed(seed + episode)
            np.random.seed(seed + episode)
            env.reset_idx = reset_idx
            obs = env.reset(is_uniform_rollout=False)
            env.reset_idx = lambda ids: None
            tracker = BackflipTracker(float(env.datas[0].qpos[2]))
            trace = []
            recorded_states, recorded_times = [], []
            def capture():
                state = np.empty(mujoco.mj_stateSize(env.model, state_flag))
                mujoco.mj_getState(env.model, env.datas[0], state, state_flag)
                recorded_states.append(state)
                recorded_times.append(float(env.datas[0].time))
            if episode == 0: capture()
            foot_ids = set(env.foot_indices.tolist())
            body_ids = set(env.terminate_touch_indices.tolist())
            def measured_step(sim):
                simulate(sim)
                data = env.datas[0]
                feet, body_contact = set(), False
                force = np.zeros(6)
                for i in range(data.ncon):
                    contact = data.contact[i]
                    bodies = {int(env.model.geom_bodyid[contact.geom1]), int(env.model.geom_bodyid[contact.geom2])}
                    if 0 not in bodies: continue
                    mujoco.mj_contactForce(env.model, data, i, force)
                    magnitude = np.linalg.norm(force[:3])
                    if magnitude > 10: feet.update(bodies & foot_ids)
                    if magnitude > 1 and bodies & body_ids: body_contact = True
                tracker.update(env.model.opt.timestep, float(data.qvel[4]), float(data.qpos[2]),
                               float(data.xmat[env.base_index].reshape(3, 3)[2, 2]), len(feet), body_contact,
                               float(np.linalg.norm(data.qvel[:3])), float(np.linalg.norm(data.qvel[3:6])))
            env.gym.simulate = measured_step
            failed = False
            for step in range(env.max_episode_length):
                if stop(): break
                tick = time.perf_counter()
                with torch.no_grad():
                    states = obs["states"]
                    actions = (agent.getAction(obs["obs"], states[:, :-env.num_stages], states[:, -env.num_stages:], True)
                               if family == "comoppo" else agent.getAction(obs["obs"], True))
                    obs, _, _, info = env.step(actions)
                    obs = {k: v.clone() for k, v in obs.items()}
                data = env.datas[0]
                if episode == 0: capture()
                trace.append({"time_s": (step+1)*env.control_dt, "height_m": float(data.qpos[2]),
                              "backward_rotation_deg": math.degrees(tracker.rotation),
                              "stage": int(env.stage_buf[0].argmax()), "qpos": data.qpos.tolist()})
                if renderer is not None and step % 2 == 0:
                    camera.lookat[:] = data.qpos[:3]
                    renderer.update_scene(data, camera=camera)
                    frame = Image.fromarray(renderer.render())
                    if episode == 0: images.append(frame.resize((480, 320)))
                    if config.get("preview_path"):
                        path = Path(config["preview_path"])
                        temporary = path.with_suffix(".tmp")
                        frame.save(temporary, format="JPEG"); temporary.replace(path)
                        report({"frame_path": str(path)})
                if config.get("realtime", False):
                    time.sleep(max(0.0, env.control_dt-(time.perf_counter()-tick)))
                failed = bool(info["fails"][0])
                if bool(info["dones"][0]) or tracker.result()["success"]: break
            env.gym.simulate = simulate
            if stop(): break
            result = {"episode": episode+1, "seed": seed+episode, **tracker.result(),
                      "failed": failed, "duration_s": len(trace)*env.control_dt,
                      "final_stage": int(env.stage_buf[0].argmax())}
            result["success"] = result["success"] and not failed
            results.append(result)
            if episode == 0:
                camera_info = {"lookat": [0, 0, tracker.initial_height], "distance": 2.5, "azimuth": 90, "elevation": -18}
                save_archive(str(output / "episode-1.mjrec"), recording_spec,
                             {"kind": "recording", "controllers": {}, "camera": camera_info}, recorded_states, recorded_times)
                save_archive(str(output / "robot.mjstudio"), recording_spec,
                             {"kind": "project", "controllers": {}, "camera": camera_info, "state": recorded_states[0].tolist()})
            with (output / f"episode-{episode+1}.json").open("w") as file: json.dump(trace, file)
            with (output / f"episode-{episode+1}.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["time_s", "height_m", "backward_rotation_deg", "stage"])
                writer.writeheader(); writer.writerows({k: v for k, v in row.items() if k != "qpos"} for row in trace)
            report({"evaluation_progress": result, "iteration": episode+1})
    finally:
        env.gym.simulate = simulate
        env.reset_idx = reset_idx
        if renderer is not None: renderer.close()
    if images:
        images[0].save(output / "rollout.gif", save_all=True, append_images=images[1:], duration=40, loop=0)
    summary = {"checkpoint": str(Path(checkpoint).resolve()), "total_steps": metadata["total_steps"],
               "task": metadata["task"]["name"], "backend": "MuJoCo", "seed": seed,
               "episodes": len(results), "requested_episodes": episodes,
               "success_rate": sum(r["success"] for r in results)/max(1, len(results)),
               "results": results, "output_dir": str(output.resolve()), "elapsed": time.perf_counter()-started,
               "recording_path": str((output / "episode-1.mjrec").resolve()) if results else "",
               "stopped": bool(stop()), "criterion": "backward turn >=330deg, inversion, flight, upright foot landing >=0.25s, no base/hip collision"}
    (output / "evaluation.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    stopped = False
    parent = os.getppid()
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    def report(value): print(json.dumps(value, allow_nan=False), flush=True)
    report({"result": evaluate(json.loads(Path(args.config).read_text()), args.checkpoint, report,
                               lambda: stopped or os.getppid() != parent)})


if __name__ == "__main__": main()
