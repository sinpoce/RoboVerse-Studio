# SPDX-License-Identifier: Apache-2.0
"""Discover trainable entry points in an existing robotics project.

The importer is intentionally dependency-light: it scans source and config
files without importing the project's framework. That lets RoboVerse Studio
inspect Isaac Gym projects on macOS even when Isaac Gym itself is unavailable.
Known projects use explicit local physics adapters; discovery never implies
that arbitrary Isaac Gym code is portable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


SCHEMA = "roboverse.project.v1"


def _label(value: str) -> str:
    return re.sub(r"[_-]+", " ", value).strip().title()


def _module(root: Path, file: Path) -> str:
    return ".".join(file.relative_to(root).with_suffix("").parts)


def _yaml_files(root: Path, folder: str) -> list[Path]:
    base = root / folder
    if not base.exists():
        return []
    return sorted(path for path in base.rglob("*.yaml") if path.is_file())


def _python_files(root: Path, folder: str) -> list[Path]:
    base = root / folder
    if not base.exists():
        return []
    return sorted(
        path for path in base.rglob("*.py")
        if path.is_file() and path.name != "__init__.py" and "__pycache__" not in path.parts
    )


def scan_project(project: str | Path) -> dict[str, Any]:
    root = Path(project).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"project directory does not exist: {root}")
    stagewise = all((root / name).is_file() for name in (
        "main_teacher.py", "algos/comoppo/agent.py", "tasks/go1_backflip.py", "assets/go1/xml/go1.xml"))

    models = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in {".git", "__pycache__", "runs", "logs"} for part in path.parts):
            continue
        if path.suffix.lower() in {".xml", ".mjcf", ".urdf", ".sdf"}:
            models.append({
                "id": str(path.relative_to(root)),
                "label": path.stem,
                "path": str(path),
                "format": path.suffix.lower().lstrip("."),
            })

    tasks = []
    task_yaml = _yaml_files(root, "tasks")
    for path in task_yaml:
        tasks.append({
            "id": path.stem,
            "label": _label(path.stem),
            "config": str(path),
            "relative_config": str(path.relative_to(root)),
        })

    algorithms = []
    algo_yaml = _yaml_files(root, "algos")
    for path in algo_yaml:
        relative = path.relative_to(root)
        if relative.parts[0] == "algos" and len(relative.parts) > 1:
            family = relative.parts[1]
        else:
            family = path.stem
        task_name = path.stem
        algorithm_id = f"project_{family}_{task_name}"
        # The native project runner uses these paths without importing the
        # project's framework into the RoboVerse process.
        entry = {
            "id": algorithm_id,
            "label": f"{_label(family)} · {_label(task_name)} · MuJoCo" if stagewise else f"{_label(family)} · {_label(task_name)}",
            "spec": "project_runner:train",
            "task_config": str(root / "tasks" / f"{task_name}.yaml"),
            "algorithm_config": str(path),
            "project_root": str(root),
            "project_algorithm": family,
            "config_kind": "project",
            "adapter": "stagewise_mujoco" if stagewise else "unavailable",
            "num_envs": 8,
            "horizon": 128,
            "iterations": 1000,
            "model_num": 0,
            "checkpoint_every": 25,
        }
        if (root / "tasks" / f"{task_name}.yaml").exists():
            if family == "student": entry["teacher_run"] = ""
            algorithms.append(entry)

    # Also expose Python algorithm modules for projects whose launch command
    # is not represented by a YAML file (users can wire them manually).
    python_algorithms = []
    for path in _python_files(root, "algos"):
        python_algorithms.append({
            "module": _module(root, path),
            "path": str(path),
            "label": _label(path.stem),
        })

    runtime = "MuJoCo · Stage-Wise adapter" if stagewise else "unadapted Python project"
    return {
        "schema": SCHEMA,
        "project": {
            "name": root.name,
            "root": str(root),
            "runtime": runtime,
            "compatible_on_macos": stagewise,
        },
        "models": models,
        "tasks": tasks,
        "algorithms": algorithms,
        "python_algorithms": python_algorithms,
        "warnings": (["使用原项目算法与任务逻辑，物理后端为 MuJoCo / CPU；结果不等同于原 PhysX 实验。"] if stagewise else
                     ["项目已扫描；尚无已验证的物理后端适配器。请提供自定义任务/算法配置。"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Import an existing robotics project into RoboVerse Studio")
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    manifest = scan_project(args.project)
    output = Path(args.output).expanduser() if args.output else Path(args.project).expanduser() / "roboverse.project.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output.resolve()),
        "project": manifest["project"],
        "models": len(manifest["models"]),
        "tasks": len(manifest["tasks"]),
        "algorithms": len(manifest["algorithms"]),
        "warnings": manifest["warnings"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
