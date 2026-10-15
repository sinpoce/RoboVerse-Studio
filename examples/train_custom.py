# SPDX-License-Identifier: Apache-2.0
"""Train any task plugin with a selectable algorithm plug-in.

Example:
  examples/.venv/bin/python examples/train_custom.py \
    --task tasks.g1_backflip:create_task \
    --model models/unitree_g1/scene.xml \
    --config examples/configs/g1_backflip.json \
    --algorithm ppo
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
for path in (ROOT, EXAMPLES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from algorithm_api import config_for_algorithm, configured_algorithms, load_config, resolve_algorithm, run_algorithm  # noqa: E402
from import_project import SCHEMA, scan_project  # noqa: E402
from python.task_api import TaskSpec  # noqa: E402


def load_factory(path: str):
    try:
        module_name, function_name = path.split(":", 1)
    except ValueError as exc:
        raise ValueError("--task must use module:function syntax") from exc
    return getattr(importlib.import_module(module_name), function_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Configurable MuJoCo task and algorithm trainer")
    parser.add_argument("--task", default=None, help="task factory, e.g. tasks.g1_backflip:create_task")
    parser.add_argument("--model", default=None, help="MuJoCo XML/MJCF entry file")
    parser.add_argument("--config", required=True, help="JSON with task and ppo sections")
    parser.add_argument("--algorithm", default=None, help="ppo or a module:function algorithm plug-in")
    parser.add_argument("--checkpoint", default="runs/policy.pt")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    raw = scan_project(config_path) if config_path.is_dir() else load_config(config_path)
    if raw.get("schema") == SCHEMA:
        algorithms = configured_algorithms(raw)
        selected = args.algorithm or raw.get("algorithm")
        descriptor = next((a for a in algorithms if selected in (a.algorithm_id, a.key, (a.options or {}).get("id"))), None) if selected else algorithms[0]
        if descriptor is None: raise ValueError("Choose an imported algorithm id: " + ", ".join(a.algorithm_id for a in algorithms))
        config = config_for_algorithm(raw, descriptor)
        if args.num_envs is not None: config["num_envs"] = args.num_envs
        if args.iterations is not None: config["iterations"] = args.iterations
        config["preview"] = False
        result = run_algorithm(descriptor, None, config, args.checkpoint,
                               on_iteration=lambda value: print(json.dumps(value), flush=True))
        print(json.dumps(result, indent=2), flush=True)
        return
    task_config = dict(raw.get("task", {}))
    # The JSON file can be self-contained.  CLI values override it for quick
    # experiments and for Makefile-driven CI runs.
    task_config["plugin"] = args.task or task_config.get("plugin", "")
    model_value = args.model or task_config.get("model", "")
    model_path = Path(str(model_value)).expanduser()
    if not model_path.is_absolute():
        cwd_path = (Path.cwd() / model_path).resolve()
        model_path = cwd_path if cwd_path.exists() else model_path
    task_config["model"] = str(model_path)
    spec = TaskSpec.from_dict(task_config, base=Path(args.config).expanduser().resolve().parent)
    task_config["model"] = spec.model
    task_config["plugin"] = spec.plugin
    if args.num_envs is not None:
        task_config["num_envs"] = args.num_envs
    algorithm_spec = args.algorithm or raw.get("algorithm", "ppo")
    descriptor = next((a for a in configured_algorithms(raw)
                       if algorithm_spec in (a.algorithm_id, a.key, (a.options or {}).get("id"))), None)
    if descriptor is None: descriptor = resolve_algorithm(algorithm_spec)
    algorithm_config = config_for_algorithm(raw, descriptor)
    if args.iterations is not None:
        algorithm_config["iterations"] = args.iterations
    num_envs = int(task_config.get("num_envs", 64))
    factory = load_factory(spec.plugin)
    task = factory(spec.model, num_envs, task_config)
    print(json.dumps({
        "task": task.metadata(),
        "algorithm": descriptor.algorithm_id,
        "algorithm_label": descriptor.label,
        "config": algorithm_config,
    }, ensure_ascii=False, indent=2), flush=True)

    def report(value: dict) -> None:
        print(json.dumps(value, ensure_ascii=False), flush=True)

    result = run_algorithm(descriptor, task, algorithm_config, args.checkpoint, on_iteration=report)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
