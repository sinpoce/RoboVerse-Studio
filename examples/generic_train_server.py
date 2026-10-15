# SPDX-License-Identifier: Apache-2.0
"""WebSocket training sidecar for user-defined task and algorithm plugins.

The workbench loads task and algorithm plugins, streams training telemetry and
previews, and manages checkpoints and independent policy evaluation.

Environment variables:
  MJBATCH_STUDIO_TASK_CONFIG  JSON task/algorithm file
  MJBATCH_STUDIO_ALGORITHM    optional algorithm id or module:function
  MJBATCH_STUDIO_TASK          optional module:function override
  MJBATCH_STUDIO_MODEL         optional MJCF/XML override
  MJBATCH_STUDIO_CHECKPOINT    output policy path
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time

import mujoco

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
PYTHON = ROOT / "python"
for path in (ROOT, EXAMPLES, PYTHON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from algorithm_api import (  # noqa: E402
    AlgorithmDescriptor,
    PPO,
    config_for_algorithm,
    configured_algorithms,
    resolve_algorithm,
    run_algorithm,
    json_configuration,
)
from import_project import SCHEMA, scan_project  # noqa: E402
from ppo import PPOConfig, load_config  # noqa: E402
from python.task_api import TaskSpec  # noqa: E402
import training_transport  # noqa: E402


def _factory(path: str):
    module, function = path.split(":", 1)
    return getattr(importlib.import_module(module), function)


class Worker(threading.Thread):
    def __init__(self, loop, outbox) -> None:
        super().__init__(daemon=True)
        self.loop, self.outbox = loop, outbox
        self.cmd_q: "queue.Queue[dict]" = queue.Queue()
        self.running = False
        self.configured = False
        self.task = None
        self.task_spec: TaskSpec | None = None
        self.project_mode = False
        self.project_manifest: dict = {}
        self.algorithms: list[AlgorithmDescriptor] = [PPO]
        self.algorithm_by_id: dict[str, AlgorithmDescriptor] = {PPO.algorithm_id: PPO}
        self.selected_algorithm = PPO
        self.ppo_config = PPOConfig()
        self.train_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.iteration = 0
        self.iterations = 0
        self.elapsed = 0.0
        self.last = {}
        self.metric = "reward"
        self.activity = "training"
        self.evaluation_result = None
        self.config_path = Path(os.environ.get(
            "MJBATCH_STUDIO_TASK_CONFIG",
            str(ROOT / "examples" / "configs" / "g1_backflip.json"),
        )).expanduser().resolve()
        self._raw = scan_project(self.config_path) if self.config_path.is_dir() else load_config(self.config_path)
        self._load_spec()

    def emit(self, message: dict) -> None:
        self.loop.call_soon_threadsafe(self.outbox.put_nowait, ("text", message))

    def _err(self, message: str) -> None:
        self.emit({"type": "error", "message": message})

    def _load_spec(self, model_override: str | None = None, use_environment: bool = True) -> None:
        self.project_mode = self._raw.get("schema") == SCHEMA
        self.project_manifest = dict(self._raw) if self.project_mode else {}
        algorithm_raw = dict(self._raw)
        environment_algorithm = os.environ.get("MJBATCH_STUDIO_ALGORITHM") if use_environment else None
        if environment_algorithm:
            algorithm_raw["algorithm"] = environment_algorithm
        self.algorithms = configured_algorithms(algorithm_raw)
        self.algorithm_by_id = {item.algorithm_id: item for item in self.algorithms}
        selected_entry = environment_algorithm or algorithm_raw.get("algorithm")
        if isinstance(selected_entry, dict):
            requested = resolve_algorithm(selected_entry)
            selected = self.algorithm_by_id.get(requested.algorithm_id)
        else:
            selected_spec = str(selected_entry or self.algorithms[0].spec or self.algorithms[0].key)
            selected = next((item for item in self.algorithms
                             if selected_spec in (item.key, (item.options or {}).get("id"))), None)
            if selected is None:
                selected = self.algorithm_by_id.get(selected_spec)
        self.selected_algorithm = selected or self.algorithms[0]
        self.ppo_config = PPOConfig.from_dict(dict(self._raw.get("ppo", {})))
        self.ppo_config.validate()
        algorithm_config = config_for_algorithm(self._raw, self.selected_algorithm)
        self.iterations = int(algorithm_config.get("iterations", self.ppo_config.iterations))
        self.checkpoint = Path(os.environ.get("MJBATCH_STUDIO_CHECKPOINT", "runs/policy.pt")).expanduser().resolve()
        if self.project_mode:
            self.task_spec = None
            self.task_config = {}
            return
        task = dict(self._raw.get("task", {}))
        task["plugin"] = (os.environ.get("MJBATCH_STUDIO_TASK", task.get("plugin", ""))
                          if use_environment else task.get("plugin", ""))
        model_value = model_override or (os.environ.get("MJBATCH_STUDIO_MODEL", task.get("model", ""))
                                         if use_environment else task.get("model", ""))
        model_path = Path(str(model_value)).expanduser()
        if not model_path.is_absolute():
            cwd_path = (Path.cwd() / model_path).resolve()
            model_path = cwd_path if cwd_path.exists() else model_path
        task["model"] = str(model_path)
        self.task_spec = TaskSpec.from_dict(task, base=self.config_path.parent)
        self.task_config = task

    def _catalogue(self) -> dict:
        if self.project_mode:
            # Imported experiments bind a task to its own robot asset; choosing
            # an unrelated XML must not silently leave the trained robot unchanged.
            models = [{"id": "imported_project", "label": self.project_manifest["project"]["name"]}]
            return {
                "type": "model_list",
                "models": models or [{"id": "imported_project", "label": self.project_manifest.get("project", {}).get("name", "Imported project")}],
                "metrics": ["reward", "fail_rate", "steps/s (throughput)"],
            }
        assert self.task_spec is not None
        return {
            "type": "model_list",
            "models": [{"id": self.task_spec.task_id, "label": self.task_spec.label}],
            "metrics": ["reward", "success", "steps/s (throughput)"],
        }

    def _model_info(self) -> None:
        if self.project_mode:
            project = self.project_manifest.get("project", {})
            self.emit({
                "type": "model_info",
                "name": f"{project.get('name', 'Imported project')} · {project.get('runtime', 'external')} project",
                "nq": 0, "nv": 0, "nu": 0, "nbody": 0, "njnt": 0, "ngeom": 0,
                "algorithms": [{"id": item.algorithm_id, "label": item.label} for item in self.algorithms],
                "metrics": ["reward", "fail_rate", "steps/s (throughput)"],
            })
            return
        assert self.task_spec is not None
        if self.task is not None:
            info = self.task.metadata()
        else:
            model = mujoco.MjModel.from_xml_path(self.task_spec.model)
            info = {
                "nq": model.nq, "nv": model.nv, "nu": model.nu,
                "nbody": model.nbody, "njnt": model.njnt, "ngeom": model.ngeom,
                "reward_terms": list(dict(self.task_config.get("reward", {}))),
            }
        self.emit({
            "type": "model_info",
            "name": f"{self.task_spec.label} · configurable training task",
            **{key: int(info.get(key, 0)) for key in ("nq", "nv", "nu", "nbody", "njnt", "ngeom")},
            "algorithms": [
                {"id": item.algorithm_id, "label": item.label} for item in self.algorithms
            ],
            "metrics": ["reward", "success", "steps/s (throughput)"] + list(info.get("reward_terms", [])),
        })

    def _status(self, force: bool = False) -> None:
        values = {
            "reward": float(self.last.get("reward", 0.0)),
            "steps/s (throughput)": float(self.last.get("steps_per_second", 0.0)),
        }
        for key in ("success", "fail_rate"):
            if key in self.last: values[key] = float(self.last[key])
        values.update({str(k): float(v) for k, v in self.last.get("terms", {}).items()})
        terms = [{"id": key, "value": value} for key, value in values.items() if key not in {"steps/s (throughput)"}]
        self.emit({
            "type": "status", "running": self.running, "configured": self.configured,
            "activity": self.activity,
            "algo": self.selected_algorithm.algorithm_id,
            "iteration": self.iteration, "iterations": self.iterations,
            "elapsed": self.elapsed, "steps_per_s": values["steps/s (throughput)"],
            "metric_value": values.get(self.metric, values["reward"]), "metric": self.metric,
            "values": values, "terms": terms,
        })

    def _emit_iteration(self, value: dict) -> None:
        if value.get("evaluation_progress"):
            result = value["evaluation_progress"]
            self.emit({"type": "log", "message": f"评估回合 {result['episode']}：{'成功' if result['success'] else '未完成后空翻'}，后向转角 {result['backward_rotation_deg']:.1f}°"})
            return
        if value.get("model_info"):
            self.emit({"type": "model_info", **value["model_info"],
                       "algorithms": [{"id": item.algorithm_id, "label": item.label} for item in self.algorithms]})
            self._status(force=True)
            return
        if value.get("frame_path"):
            data = Path(value["frame_path"]).read_bytes()
            self.loop.call_soon_threadsafe(self.outbox.put_nowait, ("binary", data))
            return
        self.last = value
        self.iteration = int(value.get("iteration", self.iteration))
        self.elapsed = float(value.get("elapsed", self.elapsed))
        self._status(force=True)

    def _ensure_idle(self):
        if self.train_thread and self.train_thread.is_alive():
            raise RuntimeError("训练尚未停止；请先停止并等待检查点保存。")

    def _settings(self):
        config = config_for_algorithm(self._raw, self.selected_algorithm)
        self.emit({"type": "training_config", "config_json": json.dumps(json_configuration(config), indent=2, ensure_ascii=False, allow_nan=False),
                   "checkpoint": str(self.checkpoint), "config_path": str(self.config_path),
                   "algo": self.selected_algorithm.algorithm_id,
                   "num_sims": config.get("num_envs", self.task_spec.num_envs if self.task_spec else 8)})

    def _checkpoint_config(self, path):
        from stagewise_evaluate import checkpoint_metadata
        checkpoint, _, metadata = checkpoint_metadata(path)
        config = config_for_algorithm(self._raw, self.selected_algorithm)
        if not self.project_mode or config.get("adapter") != "stagewise_mujoco":
            raise ValueError("请先导入 Stage-Wise 项目并选择对应任务算法")
        task = Path(config["task_config"])
        import yaml
        if yaml.safe_load(task.read_text())["name"] != metadata["task"]["name"]:
            raise ValueError("检查点的机器人任务与当前选择不一致")
        if config.get("project_algorithm") != metadata["algorithm_config"]["name"].lower():
            raise ValueError("检查点算法与当前选择不一致")
        return checkpoint, metadata, config

    def _evaluate(self, config, checkpoint):
        try:
            from project_runner import evaluate
            result = evaluate(config, checkpoint, self._emit_iteration, self.stop_event.is_set)
            self.evaluation_result = result
            self.emit({"type": "evaluation_result", "success_rate": result["success_rate"],
                       "episodes": result["episodes"], "recording_path": result.get("recording_path", ""),
                       "report_path": str(Path(result["output_dir"]) / "evaluation.json")})
            self.emit({"type": "log", "message": f"{'评估已停止' if result.get('stopped') else '评估完成'}：{sum(r['success'] for r in result['results'])}/{result['episodes']} 回合成功；报告：{result['output_dir']}"})
        except Exception as exc:
            self._err(f"评估失败：{type(exc).__name__}: {exc}")
        finally:
            self.running = False
            self.activity = "training"
            self._status(force=True)

    def _train(self) -> None:
        try:
            config = config_for_algorithm(self._raw, self.selected_algorithm)
            if self.selected_algorithm.algorithm_id == PPO.algorithm_id:
                config = dict(self._raw.get("ppo", {}))
                config.update({
                    key: value for key, value in {
                        "horizon": self.ppo_config.horizon,
                        "iterations": self.ppo_config.iterations,
                        "learning_rate": self.ppo_config.learning_rate,
                        "entropy_coef": self.ppo_config.entropy_coef,
                        "gamma": self.ppo_config.gamma,
                        "epochs": self.ppo_config.epochs,
                        "minibatches": self.ppo_config.minibatches,
                        "clip_ratio": self.ppo_config.clip_ratio,
                        "gae_lambda": self.ppo_config.gae_lambda,
                        "value_coef": self.ppo_config.value_coef,
                        "hidden": list(self.ppo_config.hidden),
                    }.items()
                })
            result = run_algorithm(
                self.selected_algorithm,
                self.task,
                config,
                self.checkpoint,
                on_iteration=self._emit_iteration,
                stop=lambda: self.stop_event.is_set(),
            )
            self.last = result
            if self.project_mode and result.get("total_steps") and result.get("checkpoint"):
                config["model_num"] = result["total_steps"]
                self._raw.setdefault("algorithm_config", {})[self.selected_algorithm.algorithm_id] = config
                self._settings()
            if result.get("checkpoint"):
                self.emit({"type": "log", "message": f"{'已停止' if self.stop_event.is_set() else '训练完成'}，检查点：{result['checkpoint']}"})
        except Exception as exc:
            self._err(f"training failed: {type(exc).__name__}: {exc}")
        finally:
            self.running = False
            self._status(force=True)

    def _configure(self, num_envs: int) -> None:
        self._ensure_idle()
        if self.project_mode:
            config = config_for_algorithm(self._raw, self.selected_algorithm)
            if config.get("adapter") != "stagewise_mujoco":
                raise ValueError("项目尚无本地仿真适配器")
            config["num_envs"] = num_envs
            self._raw.setdefault("algorithm_config", {})[self.selected_algorithm.algorithm_id] = config
            self.configured = True
            self.running = False
            self.stop_event.clear()
            self.iteration = 0
            self.elapsed = 0.0
            self.last = {}
            self._model_info()
            self.emit({"type": "configured", "num_sims": num_envs})
            return
        assert self.task_spec is not None
        self.task_config["num_envs"] = num_envs
        factory = _factory(self.task_spec.plugin)
        self.task = factory(self.task_spec.model, num_envs, self.task_config)
        self.configured = True
        self.running = False
        self.stop_event.clear()
        self.iteration = 0
        self.elapsed = 0.0
        self.last = {}
        self._model_info()
        self.emit({"type": "configured", "num_sims": num_envs})

    def _handle(self, cmd: dict) -> None:
        kind = cmd.get("cmd")
        if kind == "hello":
            self.emit({"type": "hello", "version": 2})
            self.emit(self._catalogue())
            self._model_info()
            self._settings()
        elif kind == "list_models":
            self.emit(self._catalogue())
        elif kind == "task_config":
            self._ensure_idle()
            path = Path(str(cmd.get("path", ""))).expanduser().resolve()
            self.config_path = path
            self._raw = scan_project(path) if path.is_dir() else load_config(path)
            self._load_spec(use_environment=False)
            self.task = None
            self.configured = False
            self.emit(self._catalogue())
            self._model_info()
            self._settings()
            for warning in self.project_manifest.get("warnings", []):
                self.emit({"type": "log", "message": warning})
        elif kind == "load_checkpoint":
            self._ensure_idle()
            checkpoint, metadata, config = self._checkpoint_config(cmd.get("path", ""))
            # Restore the saved algorithm hyperparameters along with optimizer state.
            config["algorithm_overrides"] = metadata["algorithm_config"]
            config["model_num"] = int(metadata["total_steps"])
            self.checkpoint = checkpoint
            self._raw.setdefault("algorithm_config", {})[self.selected_algorithm.algorithm_id] = config
            self._settings()
            self.emit({"type": "log", "message": f"已加载检查点，续训起点 {metadata['total_steps']} 步：{checkpoint}"})
        elif kind == "evaluate_policy":
            self._ensure_idle()
            checkpoint, _, config = self._checkpoint_config(cmd.get("checkpoint", self.checkpoint))
            episodes = int(cmd.get("episodes", 5))
            if not 1 <= episodes <= 100: raise ValueError("评估回合数必须为 1–100")
            config.update(evaluation_episodes=episodes, evaluation_seed=int(cmd.get("seed", 1001)),
                          realtime=bool(cmd.get("realtime", True)))
            self.stop_event.clear()
            self.activity = "evaluation"
            self.running = True
            self.train_thread = threading.Thread(target=self._evaluate, args=(config, checkpoint), daemon=True)
            self.train_thread.start()
        elif kind == "algorithm_config":
            self._ensure_idle()
            config = json.loads(str(cmd.get("config_json", "{}")))
            if not isinstance(config, dict): raise ValueError("参数必须是 JSON 对象")
            if self.selected_algorithm.config_kind == "ppo":
                self.ppo_config = PPOConfig.from_dict(config)
                self.ppo_config.validate()
                self._raw["ppo"] = config
            else:
                self._raw.setdefault("algorithm_config", {})[self.selected_algorithm.algorithm_id] = config
            self.iterations = int(config.get("iterations", 1000))
            if cmd.get("checkpoint"): self.checkpoint = Path(cmd["checkpoint"]).expanduser().resolve()
            self.configured = False
            self._settings()
            self.emit({"type": "log", "message": "算法参数已更新。"})
        elif kind == "algorithm":
            self._ensure_idle()
            algorithm_id = str(cmd.get("name", ""))
            if algorithm_id not in self.algorithm_by_id:
                raise ValueError(f"algorithm is not configured: {algorithm_id}")
            self.selected_algorithm = self.algorithm_by_id[algorithm_id]
            config = config_for_algorithm(self._raw, self.selected_algorithm)
            self.iterations = int(config.get("iterations", self.ppo_config.iterations))
            self.iteration, self.elapsed, self.last = 0, 0.0, {}
            self.configured = False
            self._settings()
            self.emit({
                "type": "log",
                "message": f"Selected algorithm: {self.selected_algorithm.label}. Configure to create a fresh run.",
            })
        elif kind == "configure":
            default_sims = self.task_spec.num_envs if self.task_spec is not None else 0
            self._configure(max(1, int(cmd.get("num_sims", default_sims))))
        elif kind == "start_training":
            # A single workbench action applies parameters, configures physics,
            # then starts the rollout. Do not race three UI round trips.
            self._ensure_idle()
            config = json.loads(str(cmd.get("config_json", "{}")))
            if not isinstance(config, dict): raise ValueError("参数必须是 JSON 对象")
            num_envs = int(config.get("num_envs", cmd.get("num_sims", 8)))
            if num_envs < 1 or (self.project_mode and num_envs > 256):
                raise ValueError("并行环境数超出范围")
            if int(config.get("iterations", 1000)) < 1 or int(config.get("horizon", 128)) < 2:
                raise ValueError("迭代次数必须 >= 1，采样步数必须 >= 2")
            self._handle({**cmd, "cmd": "algorithm_config"})
            self._configure(num_envs)
            self._handle({"cmd": "run"})
        elif kind == "run":
            if not self.configured:
                raise RuntimeError("configure a task before running")
            if not self.train_thread or not self.train_thread.is_alive():
                self.stop_event.clear()
                self.activity = "training"
                self.running = True
                self.train_thread = threading.Thread(target=self._train, daemon=True)
                self.train_thread.start()
        elif kind == "pause":
            self.stop_event.set()
            self.emit({"type": "log", "message": "正在停止评估…" if self.activity == "evaluation" else "正在停止并保存检查点…"})
        elif kind == "reset":
            self._ensure_idle()
            self.running = False
            self.stop_event.set()
            if self.task is not None:
                self.task.reset()
            self.iteration, self.elapsed, self.last = 0, 0.0, {}
        elif kind == "metric":
            self.metric = str(cmd.get("name", "reward"))
        else:
            self._err(f"unknown command: {kind}")

    def run(self) -> None:
        last_status = 0.0
        while True:
            while True:
                try:
                    command = self.cmd_q.get_nowait()
                except queue.Empty:
                    break
                try:
                    self._handle(command)
                except Exception as exc:
                    self._err(f"{type(exc).__name__}: {exc}")
            now = time.perf_counter()
            if now - last_status >= 0.25:
                self._status()
                last_status = now
            time.sleep(0.01)


def main() -> None:
    training_transport.main(make_worker=Worker)


if __name__ == "__main__":
    main()
