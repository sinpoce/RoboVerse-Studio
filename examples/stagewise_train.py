# SPDX-License-Identifier: Apache-2.0
"""Train the imported Stage-Wise agent on local MuJoCo, including on macOS."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from stagewise_mujoco import create_env
from algorithm_api import json_configuration


def run(config, checkpoint, report=print, stop=lambda: False):
    project = Path(config["project_root"]).expanduser().resolve()
    task_path, algo_path = Path(config["task_config"]), Path(config["algorithm_config"])
    task_cfg, algo_cfg = yaml.safe_load(task_path.read_text()), yaml.safe_load(algo_path.read_text())
    family = algo_cfg["name"].lower()
    if family not in {"comoppo", "student"}: raise ValueError(f"No Stage-Wise adapter for {family}")
    nenv, horizon, iterations = (int(config.get(k, d)) for k, d in (("num_envs", 8), ("horizon", 128), ("iterations", 1000)))
    if not (1 <= nenv <= 256 and horizon >= 2 and iterations >= 1):
        raise ValueError("num_envs must be 1–256, horizon >= 2 and iterations >= 1")
    seed = int(config.get("seed", 1))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(max(1, int(config.get("torch_threads", 2))))
    checkpoint = Path(checkpoint).expanduser().resolve()
    resume_step = int(config.get("model_num", 0))
    if checkpoint.exists() and not resume_step:
        raise FileExistsError("检查点已存在；请加载检查点续训，或为新训练选择新的保存位置。")
    run_dir = checkpoint.parent / (checkpoint.stem + ".run")
    run_dir.mkdir(parents=True, exist_ok=True)
    # Match the original algorithm hyperparameters unless explicitly overridden.
    algo_cfg.update(config.get("algorithm_overrides", {}))
    # PyYAML's YAML 1.1 resolver treats 3e-5 as a string; upstream ruamel uses
    # YAML 1.2. Preserve the intended numerical learning-rate values.
    for key in ("actor_lr", "critic_lr"):
        if key in algo_cfg: algo_cfg[key] = float(algo_cfg[key])
    algo_cfg.setdefault("history_len", task_cfg["env"]["history_len"])
    env = create_env(project, task_path.with_suffix(".py"), task_cfg, nenv, algo_cfg["history_len"])
    report({"model_info": {"name": f"{task_cfg['name']} · {family.upper()} · MuJoCo / CPU",
                           **{key: int(getattr(env.model, key)) for key in ("nq", "nv", "nu", "nbody", "njnt", "ngeom")}}})
    sys.path.insert(0, str(project))
    agent_type = importlib.import_module(f"algos.{family}.agent").Agent
    args = dict(algo_cfg)
    args.update(name=f"{task_path.stem}_{family}_mujoco", device=torch.device("cpu"),
                obs_dim=env.num_obs, state_dim=env.num_states-env.num_stages, action_dim=env.num_actions,
                num_stages=env.num_stages, reward_dim=env.num_rewards, cost_dim=env.num_costs,
                n_envs=nenv, n_steps=nenv*horizon, save_dir=str(run_dir),
                action_bound_min=-np.ones(env.num_actions), action_bound_max=np.ones(env.num_actions),
                obs_sym_mat=env.obs_sym_mat, state_sym_mat=env.state_sym_mat, joint_sym_mat=env.joint_sym_mat)
    agent = agent_type(SimpleNamespace(**args))
    if resume_step and not (run_dir / "checkpoint" / f"model_{resume_step}.pt").is_file():
        raise FileNotFoundError(f"找不到恢复训练检查点 model_{resume_step}.pt")
    if resume_step:
        saved = json.loads((run_dir / "experiment.json").read_text())
        if saved["task"]["name"] != task_cfg["name"] or saved["algorithm_config"]["name"].lower() != family:
            raise ValueError("恢复检查点的任务或算法与当前训练不一致")
    total_steps = agent.load(resume_step)
    teacher = None
    if family == "student":
        teacher_dir = Path(str(config.get("teacher_run", ""))).expanduser()
        meta_path = teacher_dir / "experiment.json"
        if not meta_path.is_file(): raise ValueError("学生训练需要 teacher_run 指向已完成 COMOPPO 训练的 .run 目录")
        meta = json.loads(meta_path.read_text())
        if meta["algorithm_config"]["name"].lower() != "comoppo" or meta["task"]["name"] != task_cfg["name"]:
            raise ValueError("教师策略必须是同一任务的 COMOPPO 检查点")
        if meta["algorithm_config"]["history_len"] > algo_cfg["history_len"]:
            raise ValueError("学生观测历史长度不能小于教师观测历史长度")
        teacher_args = copy.deepcopy(args)
        teacher_args.update(meta["algorithm_config"])
        teacher_args.update(save_dir=str(teacher_dir), obs_dim=env.raw_obs_dim*meta["algorithm_config"]["history_len"])
        teacher = importlib.import_module("algos.comoppo.agent").Agent(SimpleNamespace(**teacher_args))
        if teacher.load(int(meta["total_steps"])) == 0: raise ValueError("教师策略检查点不存在")
        agent.copyObsRMS(teacher.obs_rms)
    initial_actor = {name: value.detach().clone() for name, value in agent.actor.state_dict().items()}
    # Source provenance makes edits and physics changes visible with checkpoints.
    metadata = {"backend": "MuJoCo", "device": "cpu", "mujoco_version": mujoco_version(),
                "project_root": str(project), "task": task_cfg, "algorithm_config": copy.deepcopy(algo_cfg),
                "num_envs": nenv, "horizon": horizon, "seed": seed, "total_steps": total_steps,
                "resumed_from": resume_step,
                "task_source_sha256": hashlib.sha256(task_path.with_suffix(".py").read_bytes()).hexdigest(),
                "algorithm_source_sha256": hashlib.sha256((project / "algos" / family / "agent.py").read_bytes()).hexdigest()}
    metadata["task_config"] = str(task_path.relative_to(project))
    metadata["algorithm_config_path"] = str(algo_path.relative_to(project))
    metadata["physics_adapter_sha256"] = hashlib.sha256(Path(__file__).with_name("stagewise_mujoco.py").read_bytes()).hexdigest()
    renderer = None
    if config.get("preview_path"):
        import mujoco
        env.model.vis.global_.offwidth = 720
        env.model.vis.global_.offheight = 480
        renderer = mujoco.Renderer(env.model, height=480, width=720)
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = 2.5 if env.num_actions == 12 else 4, 100, -20
    started, last_preview, last = time.perf_counter(), 0.0, {}
    obs_dict = env.reset(is_uniform_rollout=False)
    obs, states = obs_dict["obs"], obs_dict["states"]
    iterations_done = 0

    def save_checkpoint():
        agent.save(total_steps)
        metadata.update(total_steps=total_steps, iterations=iterations_done,
                        actor_parameter_delta=sum(float((value-initial_actor[name]).abs().sum())
                                                  for name, value in agent.actor.state_dict().items()))
        # Keep periodic checkpoints usable even if a later update fails.
        temporary = run_dir / "experiment.json.tmp"
        temporary.write_text(json.dumps(json_configuration(metadata), indent=2, default=str, allow_nan=False))
        temporary.replace(run_dir / "experiment.json")
        shutil.copy2(run_dir / "checkpoint" / f"model_{total_steps}.pt", checkpoint)

    try:
        for iteration in range(1, iterations + 1):
            rewards, costs, fails = [], [], []
            for step in range(horizon):
                if stop(): break
                with torch.no_grad():
                    stages, state = states[:, -env.num_stages:], states[:, :-env.num_stages]
                    if teacher is None:
                        actions = agent.getAction(obs, state, stages, False)
                    else:
                        targets = teacher.getAction(obs[:, -teacher.obs_dim:], state, stages, True)
                        student_actions = agent.getAction(obs, True)
                        agent.step(obs.clone(), targets.clone())
                        actions = targets if total_steps < 10_000_000 or (total_steps//100_000) % 2 == 0 else student_actions
                    new_obs, reward, done, info = env.step(actions)
                    if teacher is None:
                        agent.step(reward, info["costs"], info["dones"], info["fails"],
                                   info["next_obs"], info["next_states"][:, :-env.num_stages], info["next_states"][:, -env.num_stages:])
                    # Clone because the original task overwrites tensor buffers.
                    rewards.append(reward.clone()); costs.append(info["costs"].clone()); fails.append(info["fails"].clone())
                    obs, states = new_obs["obs"].clone(), new_obs["states"].clone()
                    total_steps += nenv
                now = time.perf_counter()
                if renderer is not None and now-last_preview > 0.25:
                    from PIL import Image
                    camera.lookat[:] = env.datas[0].qpos[:3]
                    renderer.update_scene(env.datas[0], camera=camera)
                    path = Path(config["preview_path"])
                    temporary = path.with_suffix(".tmp")
                    Image.fromarray(renderer.render()).save(temporary, format="JPEG")
                    temporary.replace(path)
                    report({"frame_path": str(path)})
                    last_preview = now
            if stop(): break  # never update from a partially filled replay buffer
            update = agent.train()
            elapsed = time.perf_counter()-started
            reward_means, cost_means = torch.stack(rewards).mean((0, 1)), torch.stack(costs).mean((0, 1))
            terms = {name: float(value) for name, value in zip(env.reward_names, reward_means)}
            terms.update({"cost/"+name: float(value) for name, value in zip(env.cost_names, cost_means)})
            terms.update({key: float(value) for key, value in update.items() if np.asarray(value).ndim == 0})
            if not all(np.isfinite(x) for x in terms.values()): raise FloatingPointError("non-finite learning metrics")
            iterations_done = iteration
            last = {"iteration": iteration, "reward": float(reward_means.sum()), "terms": terms,
                    "fail_rate": float(torch.stack(fails).float().mean()), "elapsed": elapsed,
                    "total_steps": total_steps, "steps_per_second": iteration*horizon*nenv/max(elapsed, 1e-9)}
            with (run_dir / "metrics.jsonl").open("a") as metrics:
                metrics.write(json.dumps(last, allow_nan=False) + "\n")
            report(last)
            if iteration % max(1, int(config.get("checkpoint_every", 25))) == 0:
                save_checkpoint()
    finally:
        if renderer is not None: renderer.close()
    save_checkpoint()
    result = {**last, "checkpoint": str(checkpoint), "run_dir": str(run_dir), "backend": "MuJoCo",
              "total_steps": total_steps, "iteration": iterations_done,
              "actor_parameter_delta": metadata["actor_parameter_delta"], "stopped": bool(stop())}
    return result


def mujoco_version():
    import mujoco
    return mujoco.__version__


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    requested_stop = False
    parent = os.getppid()
    def handle_stop(*_):
        nonlocal requested_stop
        requested_stop = True
    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)
    def report(value): print(json.dumps(value, allow_nan=False), flush=True)
    result = run(json.loads(Path(args.config).read_text()), args.checkpoint, report,
                 lambda: requested_stop or os.getppid() != parent)
    report({"result": result})


if __name__ == "__main__": main()
