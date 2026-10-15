# SPDX-License-Identifier: Apache-2.0
"""Copy this file to create a custom learning task.

Training algorithms do not know anything about the robot. A task only needs
to provide the VectorTask methods inherited from MujocoVectorTask.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import MujocoVectorTask


class CustomTask(MujocoVectorTask):
    task_id = "custom_task"
    label = "Custom MuJoCo task"

    def observe_task(self) -> np.ndarray:
        # Return shape (num_envs, obs_dim).  Select only the signals that the
        # policy should see; never pass hidden privileged state by accident.
        return np.concatenate((self.qpos, self.qvel, self.sensordata), axis=1)

    def reward_task(self, action: np.ndarray):
        # Return one reward per environment and a dictionary of telemetry terms.
        effort = np.mean(np.square(action), axis=1)
        reward = -0.01 * effort
        return reward, {"effort": reward}

    def terminated_task(self):
        # Return (terminated, success), each a boolean vector.
        terminated = np.zeros(self.num_envs, dtype=bool)
        success = np.zeros(self.num_envs, dtype=bool)
        return terminated, success


def create_task(model_path: str, num_envs: int, config: dict[str, Any] | None = None) -> CustomTask:
    return CustomTask(model_path, num_envs, config)
