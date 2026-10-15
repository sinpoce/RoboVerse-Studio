# SPDX-License-Identifier: Apache-2.0
"""A configurable G1 backflip task template.

This is intentionally a task plugin, not a special case in the simulator.  It
is a starting point for tuning a real policy: the reward weights, episode
length, action scale and randomisation are supplied by the JSON task spec.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import MujocoVectorTask


class G1BackflipTask(MujocoVectorTask):
    task_id = "g1_backflip"
    label = "Unitree G1 · Backflip"

    def __init__(self, model_path: str, num_envs: int, config: dict[str, Any] | None = None):
        super().__init__(model_path, num_envs, config)
        if self.model.nu != 29:
            raise ValueError(f"G1 backflip expects 29 actuators, got {self.model.nu}")
        self.pelvis = self.batch.body("pelvis")
        self.left_foot = self.batch.site("left_foot")
        self.right_foot = self.batch.site("right_foot")
        self.nominal_qpos = self.model.key_qpos[0].copy() if self.model.nkey else self.qpos[0].copy()
        self.joint_qpos = self.nominal_qpos[7:].copy()
        self.rotation = np.zeros(self.num_envs, dtype=np.float32)
        self.previous_rotation = np.zeros(self.num_envs, dtype=np.float32)
        self.rng = np.random.default_rng(int(self.config.get("seed", 0)))
        self.reward_weights = {
            "alive": 0.2,
            "takeoff": 0.8,
            "rotation": 2.0,
            "landing": 8.0,
            "upright": 1.0,
            "effort": -0.01,
            "action_rate": -0.01,
            "fall": -4.0,
        }
        self.reward_weights.update({str(k): float(v) for k, v in self.config.get("reward", {}).items()})
        self._reset_batch(np.arange(self.num_envs, dtype=np.int64))
        self.obs_dim = self.observe().shape[1]

    def reset_task(self, ids: np.ndarray) -> None:
        self.rotation[ids] = 0.0
        self.previous_rotation[ids] = 0.0
        # Small pose/velocity noise prevents a policy from memorising one exact
        # trajectory while keeping the initial state recognisably standing.
        if ids.size:
            self.qpos[ids, 7:] = self.joint_qpos + self.rng.normal(0.0, 0.015, (ids.size, 29))
            self.qvel[ids, :] = self.rng.normal(0.0, 0.01, (ids.size, self.qvel.shape[1]))
            self.qpos[ids, 0:3] = self.nominal_qpos[0:3]
            self.qpos[ids, 3:7] = self.nominal_qpos[3:7]

    def _up_z(self) -> np.ndarray:
        # World-up z component of the pelvis frame from its quaternion.
        qw, qx, qy, qz = self.pelvis.xquat.T
        return 1.0 - 2.0 * (qx * qx + qy * qy)

    def observe_task(self) -> np.ndarray:
        foot_relative = np.concatenate((self.left_foot.xpos - self.pelvis.xpos,
                                         self.right_foot.xpos - self.pelvis.xpos), axis=1)
        return np.concatenate((
            self.qpos[:, 2:3],            # base height
            self.qpos[:, 3:7],            # base quaternion
            self.qvel[:, 0:6],            # base linear/angular velocity
            self.qpos[:, 7:] - self.joint_qpos,
            self.qvel[:, 6:],
            self.sensordata,
            foot_relative,
            self.previous_action,
        ), axis=1)

    def reward_task(self, action: np.ndarray):
        # qvel[3:6] is the free body's angular velocity in MuJoCo; y is the
        # lateral axis for a front/back flip.
        self.rotation += self.qvel[:, 4].astype(np.float32) * self.control_dt
        height = self.pelvis.xpos[:, 2]
        foot_grounded = ((self.left_foot.xpos[:, 2] < 0.09) &
                         (self.right_foot.xpos[:, 2] < 0.09)).astype(np.float32)
        up = np.clip(self._up_z(), -1.0, 1.0)
        upright = np.clip((up + 1.0) * 0.5, 0.0, 1.0)
        takeoff = np.clip((height - 0.82) / 0.55, 0.0, 1.0)
        rotation_progress = np.clip(np.abs(self.rotation) / (2.0 * np.pi), 0.0, 1.0)
        landing = foot_grounded * np.clip(rotation_progress, 0.0, 1.0) * upright
        fallen = (height < 0.38) | (up < -0.1)
        effort = np.mean(np.square(action), axis=1)
        action_rate = np.mean(np.square(action - self.previous_action), axis=1)
        rotation_delta = np.maximum(rotation_progress - np.clip(np.abs(self.previous_rotation) / (2.0 * np.pi), 0.0, 1.0), 0.0)
        reward = (
            self.reward_weights["alive"] * (~fallen).astype(np.float32)
            + self.reward_weights["takeoff"] * takeoff
            + self.reward_weights["rotation"] * rotation_delta
            + self.reward_weights["landing"] * landing
            + self.reward_weights["upright"] * upright * foot_grounded
            + self.reward_weights["effort"] * effort
            + self.reward_weights["action_rate"] * action_rate
            + self.reward_weights["fall"] * fallen.astype(np.float32)
        )
        self.previous_rotation[:] = self.rotation
        terms = {
            "alive": self.reward_weights["alive"] * (~fallen).astype(np.float32),
            "takeoff": self.reward_weights["takeoff"] * takeoff,
            "rotation": self.reward_weights["rotation"] * rotation_delta,
            "landing": self.reward_weights["landing"] * landing,
            "upright": self.reward_weights["upright"] * upright * foot_grounded,
            "effort": self.reward_weights["effort"] * effort,
            "action_rate": self.reward_weights["action_rate"] * action_rate,
            "fall": self.reward_weights["fall"] * fallen.astype(np.float32),
        }
        return reward, terms

    def terminated_task(self):
        height = self.pelvis.xpos[:, 2]
        up = self._up_z()
        fallen = (height < 0.38) | (up < -0.1)
        foot_grounded = ((self.left_foot.xpos[:, 2] < 0.09) &
                         (self.right_foot.xpos[:, 2] < 0.09))
        success = (np.abs(self.rotation) > 1.75 * np.pi) & foot_grounded & (up > 0.75)
        return fallen | success, success

    def metadata(self) -> dict[str, Any]:
        result = super().metadata()
        result.update({"label": self.label, "task_id": self.task_id, "reward_terms": list(self.reward_weights)})
        return result


def create_task(model_path: str, num_envs: int, config: dict[str, Any] | None = None) -> G1BackflipTask:
    return G1BackflipTask(model_path, num_envs, config)
