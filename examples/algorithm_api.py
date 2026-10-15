# SPDX-License-Identifier: Apache-2.0
"""Algorithm plug-in contract used by RoboVerse Studio.

An algorithm is deliberately a small Python adapter around a task's vector
environment.  This keeps the simulator independent from the learning method:
PPO, evolutionary search, imitation learning, MPC or a user's own trainer can
all expose the same ``train`` function and appear in the app's algorithm menu.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import importlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable

TrainFunction = Callable[..., dict[str, Any]]


def json_configuration(value):
    """Keep upstream infinite critic bounds valid in standard JSON and Swift."""
    if isinstance(value, dict): return {key: json_configuration(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [json_configuration(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value): raise ValueError("算法参数不能包含 NaN")
        return "np.inf" if value > 0 else "-np.inf"
    return value


@dataclass(frozen=True)
class AlgorithmDescriptor:
    """Metadata and entry point for one selectable training algorithm."""

    key: str
    algorithm_id: str
    label: str
    train: TrainFunction
    config_kind: str = "generic"
    spec: str = ""
    options: dict[str, Any] | None = None


def _ppo_adapter(task, config, checkpoint, on_iteration=None, stop=None):
    from ppo import PPOConfig, train as ppo_train

    ppo = PPOConfig.from_dict(dict(config or {}))
    ppo.validate()
    return ppo_train(task, ppo, checkpoint, on_iteration=on_iteration, stop=stop)


PPO = AlgorithmDescriptor(
    key="ppo",
    algorithm_id="ppo_train",
    label="PPO · configurable task",
    train=_ppo_adapter,
    config_kind="ppo",
    spec="ppo",
)


def _safe_id(module_name: str, function_name: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", f"{module_name}_{function_name}").strip("_").lower()
    return value if value.startswith("train_") else f"train_{value}"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON experiment file without importing a learning framework."""

    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("training config must be a JSON object")
    return value


def resolve_algorithm(spec: str | dict[str, Any]) -> AlgorithmDescriptor:
    """Resolve ``ppo`` or a user ``module:function`` into a descriptor.

    A mapping can override the id/label while keeping the implementation in a
    normal importable module, which is useful when the same trainer is offered
    for several task configurations.
    """

    overrides = spec if isinstance(spec, dict) else {}
    value = str(overrides.get("spec", overrides.get("path", spec))).strip()
    if value in {"ppo", "ppo_train", "builtin:ppo"}:
        descriptor = PPO
    else:
        if ":" not in value:
            raise ValueError(f"algorithm must be 'ppo' or module:function, got {value!r}")
        module_name, function_name = value.split(":", 1)
        if not module_name or not function_name:
            raise ValueError(f"invalid algorithm specification: {value!r}")
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)
        if not callable(function):
            raise TypeError(f"algorithm entry point is not callable: {value}")
        algorithm_id = str(
            overrides.get("id")
            or getattr(function, "ALGORITHM_ID", "")
            or getattr(module, "ALGORITHM_ID", "")
            or _safe_id(module_name, function_name)
        )
        if not algorithm_id.startswith("train_"):
            algorithm_id = f"train_{algorithm_id}"
        label = str(
            overrides.get("label")
            or getattr(function, "LABEL", "")
            or getattr(module, "LABEL", "")
            or algorithm_id.replace("train_", "").replace("_", " ").title()
        )
        descriptor = AlgorithmDescriptor(
            key=value,
            algorithm_id=algorithm_id,
            label=label,
            train=function,
            config_kind=str(overrides.get("config_kind", getattr(module, "CONFIG_KIND", "generic"))),
            spec=value,
            options=dict(overrides),
        )
    if overrides:
        descriptor = replace(
            descriptor,
            algorithm_id=str(overrides.get("id", descriptor.algorithm_id)),
            label=str(overrides.get("label", descriptor.label)),
            options=dict(overrides),
        )
        if not descriptor.algorithm_id.startswith("train_"):
            descriptor = replace(descriptor, algorithm_id=f"train_{descriptor.algorithm_id}")
    return descriptor


def configured_algorithms(raw: dict[str, Any]) -> list[AlgorithmDescriptor]:
    """Load the configured algorithm menu, de-duplicated by UI id."""

    specs = raw.get("algorithms", [])
    if isinstance(specs, (str, dict)):
        specs = [specs]
    specs = list(specs)
    selected = raw.get("algorithm")
    if not specs:
        specs = [selected or "ppo"]
    elif selected and selected not in specs:
        # The selected value may be a menu id, not a Python entry point.
        descriptors = [resolve_algorithm(item) for item in specs]
        if not any(selected in (item.algorithm_id, item.key, (item.options or {}).get("id"))
                   for item in descriptors):
            specs.insert(0, selected)
    result: list[AlgorithmDescriptor] = []
    seen: set[str] = set()
    for item in specs:
        descriptor = resolve_algorithm(item)
        if descriptor.algorithm_id not in seen:
            result.append(descriptor)
            seen.add(descriptor.algorithm_id)
    if not result:
        raise ValueError("no algorithms configured")
    return result


def config_for_algorithm(raw: dict[str, Any], descriptor: AlgorithmDescriptor) -> dict[str, Any]:
    """Return the algorithm-specific JSON object passed to a plug-in."""

    if descriptor.config_kind == "ppo" or descriptor.algorithm_id == PPO.algorithm_id:
        return dict(raw.get("ppo", {}))
    if descriptor.config_kind == "project":
        value = dict(descriptor.options or {})
        configs = raw.get("algorithm_config", {})
        if isinstance(configs, dict):
            custom = configs.get(descriptor.algorithm_id)
            if isinstance(custom, dict):
                value.update(custom)
        return value
    configs = raw.get("algorithm_config", {})
    if isinstance(configs, dict):
        for key in (descriptor.algorithm_id, descriptor.key, descriptor.spec):
            value = configs.get(key)
            if isinstance(value, dict):
                return dict(value)
    return dict(configs) if isinstance(configs, dict) else {}


def run_algorithm(
    descriptor: AlgorithmDescriptor,
    task,
    config: dict[str, Any],
    checkpoint: str | Path,
    on_iteration=None,
    stop=None,
) -> dict[str, Any]:
    """Run one descriptor using the stable plug-in function signature."""

    return descriptor.train(
        task,
        dict(config),
        checkpoint,
        on_iteration=on_iteration,
        stop=stop,
    )
