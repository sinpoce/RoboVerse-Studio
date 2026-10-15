# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic PPO implementation for task plugins.

The trainer only depends on the VectorTask contract.  Robot-specific code stays
in the task package, so changing a reward or swapping a MuJoCo XML does not
require forking the PPO loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


@dataclass
class PPOConfig:
    horizon: int = 256
    iterations: int = 1000
    epochs: int = 5
    minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 3e-4
    entropy_coef: float = 0.005
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    hidden: tuple[int, ...] = (256, 256)
    device: str = "auto"
    seed: int = 0
    checkpoint_every: int = 25

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PPOConfig":
        values = dict(raw)
        if "hidden" in values:
            values["hidden"] = tuple(int(x) for x in values["hidden"])
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def validate(self) -> None:
        if self.horizon < 2 or self.iterations < 1 or self.epochs < 1 or self.minibatches < 1:
            raise ValueError("horizon, iterations, epochs and minibatches must be positive")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma must be in (0, 1] and gae_lambda in [0, 1]")
        if self.clip_ratio <= 0 or self.learning_rate <= 0:
            raise ValueError("clip_ratio and learning_rate must be positive")
        if not self.hidden or any(width < 8 for width in self.hidden):
            raise ValueError("hidden must contain layers with at least 8 units")

    def device_name(self) -> str:
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...]):
        super().__init__()
        def mlp(out_dim: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            width = obs_dim
            for size in hidden:
                layers.extend((nn.Linear(width, size), nn.Tanh()))
                width = size
            layers.append(nn.Linear(width, out_dim))
            return nn.Sequential(*layers)
        self.actor = mlp(act_dim)
        self.critic = mlp(1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        return Normal(mean, self.log_std.exp().clamp(1e-3, 2.0))

    def sample(self, obs: torch.Tensor):
        distribution = self.distribution(obs)
        raw = distribution.rsample()
        action = torch.tanh(raw)
        log_prob = distribution.log_prob(raw).sum(-1)
        log_prob -= torch.log(1.0 - action.square() + 1e-6).sum(-1)
        return action, log_prob, self.critic(obs).squeeze(-1)

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        distribution = self.distribution(obs)
        action = action.clamp(-0.999999, 0.999999)
        raw = torch.atanh(action)
        log_prob = distribution.log_prob(raw).sum(-1)
        log_prob -= torch.log(1.0 - action.square() + 1e-6).sum(-1)
        entropy = distribution.entropy().sum(-1)
        return log_prob, entropy, self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor(obs))


def _tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def train(
    task,
    config: PPOConfig,
    checkpoint: str | Path,
    on_iteration: Callable[[dict[str, Any]], None] | None = None,
    stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    config.validate()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device_name())
    obs = np.asarray(task.reset(), dtype=np.float32)
    if obs.ndim != 2:
        raise ValueError("task.reset() must return (num_envs, obs_dim)")
    obs_dim, act_dim = int(obs.shape[1]), int(task.act_dim)
    network = ActorCritic(obs_dim, act_dim, config.hidden).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    checkpoint = Path(checkpoint).expanduser().resolve()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    last: dict[str, Any] = {}

    for iteration in range(1, config.iterations + 1):
        if stop and stop():
            break
        nenv = task.num_envs
        observations = np.empty((config.horizon, nenv, obs_dim), np.float32)
        actions = np.empty((config.horizon, nenv, act_dim), np.float32)
        log_probs = np.empty((config.horizon, nenv), np.float32)
        values = np.empty((config.horizon, nenv), np.float32)
        rewards = np.empty((config.horizon, nenv), np.float32)
        dones = np.empty((config.horizon, nenv), np.float32)
        terms: dict[str, list[float]] = {}

        for step in range(config.horizon):
            observations[step] = obs
            with torch.no_grad():
                action, log_prob, value = network.sample(_tensor(obs, device))
            action_np = action.cpu().numpy()
            next_obs, reward, done, info = task.step(action_np)
            actions[step] = action_np
            log_probs[step] = log_prob.cpu().numpy()
            values[step] = value.cpu().numpy()
            rewards[step] = reward
            dones[step] = done.astype(np.float32)
            for name, value_array in info.items():
                terms.setdefault(name, []).append(float(np.asarray(value_array).mean()))
            obs = np.asarray(next_obs, dtype=np.float32)

        with torch.no_grad():
            last_value = network.critic(_tensor(obs, device)).squeeze(-1).cpu().numpy()
        advantages = np.zeros_like(rewards)
        gae = np.zeros(nenv, np.float32)
        for step in range(config.horizon - 1, -1, -1):
            next_value = last_value if step == config.horizon - 1 else values[step + 1]
            not_done = 1.0 - dones[step]
            delta = rewards[step] + config.gamma * next_value * not_done - values[step]
            gae = delta + config.gamma * config.gae_lambda * not_done * gae
            advantages[step] = gae
        returns = advantages + values

        flat_obs = _tensor(observations.reshape(-1, obs_dim), device)
        flat_actions = _tensor(actions.reshape(-1, act_dim), device)
        flat_old_log_probs = _tensor(log_probs.reshape(-1), device)
        flat_advantages = _tensor(advantages.reshape(-1), device)
        flat_returns = _tensor(returns.reshape(-1), device)
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
        count = flat_obs.shape[0]
        batch_size = max(1, count // config.minibatches)
        losses: list[float] = []
        for _ in range(config.epochs):
            permutation = torch.randperm(count, device=device)
            for start_index in range(0, count, batch_size):
                indices = permutation[start_index:start_index + batch_size]
                new_log_prob, entropy, new_value = network.evaluate(flat_obs[indices], flat_actions[indices])
                ratio = (new_log_prob - flat_old_log_probs[indices]).exp()
                unclipped = ratio * flat_advantages[indices]
                clipped = ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * flat_advantages[indices]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * (new_value - flat_returns[indices]).square().mean()
                loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), config.max_grad_norm)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

        elapsed = time.perf_counter() - start
        last = {
            "iteration": iteration,
            "reward": float(rewards.mean()),
            "success": float(np.mean(terms.get("success", [0.0]))),
            "steps_per_second": float(iteration * config.horizon * nenv / max(elapsed, 1e-9)),
            "loss": float(np.mean(losses)) if losses else 0.0,
            "terms": {name: float(np.mean(values_)) for name, values_ in terms.items()},
        }
        if iteration % config.checkpoint_every == 0 or iteration == config.iterations:
            torch.save({
                "task": task.metadata(),
                "config": asdict(config),
                "iteration": iteration,
                "model": network.state_dict(),
            }, checkpoint)
        if on_iteration:
            on_iteration(last)

    return {"checkpoint": str(checkpoint), **last}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("training config must be a JSON object")
    return value
