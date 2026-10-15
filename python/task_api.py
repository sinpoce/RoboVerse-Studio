# SPDX-License-Identifier: Apache-2.0
"""Stable task contract shared by the simulator and training backends.

The workbench owns MuJoCo physics; a training task owns the learning problem.
Keeping the two contracts separate lets a user use the same XML in the editor,
an offline trainer, or a custom controller without teaching the UI about every
robot and reward function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class TaskSpec:
    """Declarative part of a task package.

    The JSON form intentionally contains only serialisable values.  Dynamic
    observations and rewards belong in the task plugin referenced by
    ``plugin``; the spec controls the safe, reproducible training settings.
    """

    task_id: str
    label: str
    model: str
    plugin: str
    action_mode: str = "position"
    action_scale: float = 1.0
    control_dt: float = 0.02
    episode_seconds: float = 5.0
    num_envs: int = 64
    observation: tuple[str, ...] = ()
    reward: Mapping[str, float] = field(default_factory=dict)
    randomization: Mapping[str, Any] = field(default_factory=dict)
    ppo: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, base: Path | None = None) -> "TaskSpec":
        required = ("task_id", "label", "model", "plugin")
        missing = [key for key in required if not str(raw.get(key, "")).strip()]
        if missing:
            raise ValueError(f"task spec missing required fields: {', '.join(missing)}")
        model = Path(str(raw["model"]))
        if base is not None and not model.is_absolute():
            model = (base / model).resolve()
        spec = cls(
            task_id=str(raw["task_id"]),
            label=str(raw["label"]),
            model=str(model),
            plugin=str(raw["plugin"]),
            action_mode=str(raw.get("action_mode", "position")),
            action_scale=float(raw.get("action_scale", 1.0)),
            control_dt=float(raw.get("control_dt", 0.02)),
            episode_seconds=float(raw.get("episode_seconds", 5.0)),
            num_envs=int(raw.get("num_envs", 64)),
            observation=tuple(str(x) for x in raw.get("observation", ())),
            reward={str(k): float(v) for k, v in dict(raw.get("reward", {})).items()},
            randomization=dict(raw.get("randomization", {})),
            ppo=dict(raw.get("ppo", {})),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if not self.task_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError("task_id may contain only letters, numbers, '_' and '-'")
        if self.action_mode not in {"position", "velocity", "torque", "raw"}:
            raise ValueError("action_mode must be position, velocity, torque, or raw")
        if self.action_scale <= 0:
            raise ValueError("action_scale must be positive")
        if self.control_dt <= 0 or self.episode_seconds <= 0:
            raise ValueError("control_dt and episode_seconds must be positive")
        if self.num_envs < 1:
            raise ValueError("num_envs must be at least one")
        if any(not key.strip() for key in self.reward):
            raise ValueError("reward term names cannot be empty")


def load_task_spec(path: str | Path) -> TaskSpec:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("task spec root must be a JSON object")
    return TaskSpec.from_dict(raw, base=path.parent)


class VectorTask(Protocol):
    """Runtime interface consumed by PPO and the training UI."""

    task_id: str
    obs_dim: int
    act_dim: int
    num_envs: int

    def reset(self, ids: Sequence[int] | None = None): ...
    def step(self, action): ...
    def metadata(self) -> Mapping[str, Any]: ...
