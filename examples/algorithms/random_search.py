# SPDX-License-Identifier: Apache-2.0
"""A tiny non-PPO baseline demonstrating the algorithm plug-in contract.

This is intentionally a baseline rather than a claim of a useful backflip
solver.  It is valuable for validating a task's observation/action/reward
interface and for showing that the app can select an algorithm with its own
configuration and checkpoint format.
"""

from __future__ import annotations

import pickle
from pathlib import Path
import time
from typing import Any

import numpy as np


ALGORITHM_ID = "train_random_search"
LABEL = "Random policy baseline"
CONFIG_KIND = "generic"


def train(task, config: dict[str, Any], checkpoint, on_iteration=None, stop=None):
    """Evaluate random action rollouts through the standard task interface."""

    iterations = max(1, int(config.get("iterations", 10)))
    horizon = max(1, int(config.get("horizon", getattr(task, "episode_steps", 128))))
    seed = int(config.get("seed", 0))
    rng = np.random.default_rng(seed)
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    best_reward = -float("inf")
    last: dict[str, Any] = {}

    for iteration in range(1, iterations + 1):
        if stop and stop():
            break
        task.reset()
        episode_reward = np.zeros(task.num_envs, dtype=np.float32)
        term_values: dict[str, list[float]] = {}
        for _ in range(horizon):
            action = rng.uniform(-1.0, 1.0, size=(task.num_envs, task.act_dim)).astype(np.float32)
            _, reward, _, info = task.step(action)
            episode_reward += np.asarray(reward, dtype=np.float32)
            for name, values in info.items():
                term_values.setdefault(name, []).append(float(np.asarray(values).mean()))
            if stop and stop():
                break
        reward = float(episode_reward.mean())
        best_reward = max(best_reward, reward)
        elapsed = time.perf_counter() - started
        last = {
            "iteration": iteration,
            "reward": reward,
            "best_reward": best_reward,
            "success": float(np.mean(term_values.get("success", [0.0]))),
            "steps_per_second": float(iteration * horizon * task.num_envs / max(elapsed, 1e-9)),
            "terms": {name: float(np.mean(values)) for name, values in term_values.items()},
        }
        if on_iteration:
            on_iteration(last)

    payload = {
        "algorithm": ALGORITHM_ID,
        "task": task.metadata(),
        "config": dict(config),
        "result": dict(last),
    }
    with checkpoint_path.open("wb") as handle:
        pickle.dump(payload, handle)
    return {"checkpoint": str(checkpoint_path), **last}
