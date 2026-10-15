# SPDX-License-Identifier: Apache-2.0
"""Small MuJoCo vector-task base class used by algorithm plug-ins."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np
from mjbatch import Batch


class MujocoVectorTask:
    """A safe default task for arbitrary actuator-based MuJoCo models.

    Task plugins normally subclass this and override ``reset_task``,
    ``observe_task`` and ``reward_task``.  The base still provides useful
    behaviour for custom models: normalized actuator actions, qpos/qvel/sensor
    observations, episode timeouts and finite-value checks.
    """

    task_id = "mujoco"
    label = "MuJoCo task"

    def __init__(self, model_path: str | Path, num_envs: int, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.num_envs = int(num_envs)
        self.batch = Batch(self.model, self.num_envs)
        self.qpos = self.batch.bind("qpos")
        self.qvel = self.batch.bind("qvel")
        self.ctrl = self.batch.bind("ctrl")
        self.sensordata = self.batch.bind("sensordata")
        self.actuator_force = self.batch.bind("actuator_force")
        self.act_dim = int(self.model.nu)
        self.nominal_ctrl = (
            self.model.key_ctrl[0].copy() if self.model.nkey else np.zeros(self.act_dim, dtype=float)
        )
        self.ctrl_low = np.where(self.model.actuator_ctrllimited, self.model.actuator_ctrlrange[:, 0], -1.0)
        self.ctrl_high = np.where(self.model.actuator_ctrllimited, self.model.actuator_ctrlrange[:, 1], 1.0)
        self.action_mode = str(self.config.get("action_mode", "position"))
        if self.action_mode not in {"position", "velocity", "torque", "raw"}:
            raise ValueError("action_mode must be position, velocity, torque, or raw")
        self.action_scale = float(self.config.get("action_scale", 1.0))
        self.timestep = float(self.model.opt.timestep)
        self.control_dt = float(self.config.get("control_dt", self.timestep))
        self.decimation = max(1, int(round(self.control_dt / self.timestep)))
        self.control_dt = self.decimation * self.timestep
        self.episode_steps = max(1, int(round(float(self.config.get("episode_seconds", 5.0)) / self.control_dt)))
        self.steps = np.zeros(self.num_envs, dtype=np.int64)
        self.previous_action = np.zeros((self.num_envs, self.act_dim), dtype=np.float32)
        self.obs_dim = int(self.qpos.shape[1] + self.qvel.shape[1] + self.sensordata.shape[1] + self.act_dim)

    def _ids(self, ids: Sequence[int] | None) -> np.ndarray:
        if ids is None:
            return np.arange(self.num_envs, dtype=np.int64)
        return np.asarray(ids, dtype=np.int64)

    def reset_task(self, ids: np.ndarray) -> None:
        """Hook for task-specific reset randomisation."""

    def observe_task(self) -> np.ndarray:
        return np.concatenate((self.qpos, self.qvel, self.sensordata, self.ctrl), axis=1)

    def reward_task(self, action: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        effort = np.mean(np.square(action), axis=1)
        return -0.01 * effort, {"effort": -0.01 * effort}

    def terminated_task(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(self.num_envs, dtype=bool), np.zeros(self.num_envs, dtype=bool)

    def _reset_batch(self, ids: np.ndarray) -> None:
        if ids.size == 0:
            return
        keyframe = 0 if self.model.nkey else -1
        self.batch.reset(ids, keyframe=keyframe)
        self.steps[ids] = 0
        self.previous_action[ids] = 0.0
        self.reset_task(ids)
        self.batch.forward(ids)

    def reset(self, ids: Sequence[int] | None = None) -> np.ndarray:
        selected = self._ids(ids)
        self._reset_batch(selected)
        return self.observe().copy()

    def apply_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.num_envs, self.act_dim):
            raise ValueError(f"action shape {action.shape} != {(self.num_envs, self.act_dim)}")
        action = np.clip(action, -1.0, 1.0)
        if self.action_mode == "position":
            target = self.nominal_ctrl + self.action_scale * action
        else:
            # Velocity, torque and raw modes use the actuator's declared
            # control range.  The XML actuator type and transmission remain
            # authoritative; this layer only maps the policy's [-1, 1] output
            # into that range.
            centre = 0.5 * (self.ctrl_low + self.ctrl_high)
            half_range = 0.5 * (self.ctrl_high - self.ctrl_low)
            target = centre + self.action_scale * action * half_range
        self.ctrl[:] = np.clip(target, self.ctrl_low, self.ctrl_high)

    def observe(self) -> np.ndarray:
        value = np.asarray(self.observe_task(), dtype=np.float32)
        if value.ndim != 2 or value.shape[0] != self.num_envs:
            raise ValueError("task observation must have shape (num_envs, obs_dim)")
        if not np.isfinite(value).all():
            raise FloatingPointError("task observation contains NaN or infinity")
        self.obs_dim = int(value.shape[1])
        return value

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        self.apply_action(action)
        for _ in range(self.decimation):
            self.batch.step()
        self.steps += 1
        reward, terms = self.reward_task(action)
        terminated, success = self.terminated_task()
        timeout = self.steps >= self.episode_steps
        done = terminated | timeout
        info = {name: np.asarray(value, dtype=np.float32) for name, value in terms.items()}
        info["success"] = np.asarray(success, dtype=np.float32)
        info["timeout"] = timeout.astype(np.float32)
        self.previous_action[:] = action
        next_obs = self.observe().copy()
        reset_ids = np.flatnonzero(done)
        if reset_ids.size:
            self._reset_batch(reset_ids)
            next_obs[reset_ids] = self.observe()[reset_ids]
        return next_obs, np.asarray(reward, dtype=np.float32), done, info

    def metadata(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "label": self.label,
            "model": self.model_path,
            "nq": int(self.model.nq),
            "nv": int(self.model.nv),
            "nu": int(self.model.nu),
            "obs_dim": int(self.obs_dim),
            "act_dim": int(self.act_dim),
            "action_mode": self.action_mode,
            "control_dt": self.control_dt,
            "episode_steps": self.episode_steps,
        }
