# SPDX-License-Identifier: Apache-2.0
"""Run imported algorithms in an isolated local process with real telemetry."""
from __future__ import annotations
import json
from pathlib import Path
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time

ALGORITHM_ID = "train_imported_project"
LABEL = "Imported project algorithm"
CONFIG_KIND = "project"


def train(task, config, checkpoint, on_iteration=None, stop=None):
    return _run("stagewise_train.py", config, checkpoint, on_iteration, stop)


def evaluate(config, checkpoint, on_iteration=None, stop=None):
    return _run("stagewise_evaluate.py", config, checkpoint, on_iteration, stop)


def _run(script_name, config, checkpoint, on_iteration=None, stop=None):
    if config.get("adapter") != "stagewise_mujoco":
        raise ValueError("此项目尚无本地适配器；不能把文件扫描当作可运行的训练。")
    script = Path(__file__).with_name(script_name)
    checkpoint = Path(checkpoint).expanduser().resolve()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="roboverse-training-") as directory:
        config = dict(config)
        if config.get("preview", True):
            config["preview_path"] = str(Path(directory) / "preview.jpg")
        path = Path(directory) / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "-u", str(script), "--config", str(path), "--checkpoint", str(checkpoint)],
            cwd=config["project_root"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lines = queue.Queue()
        def read():
            try:
                for line in process.stdout: lines.put(line)
            finally: lines.put(None)
        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        tail, result, ended, stopping = [], None, False, None
        try:
            while not ended:
                if stop and stop() and stopping is None:
                    process.send_signal(signal.SIGINT)
                    stopping = time.monotonic()
                if stopping is not None and time.monotonic()-stopping > 10 and process.poll() is None:
                    process.kill()
                try: line = lines.get(timeout=0.1)
                except queue.Empty: continue
                if line is None:
                    ended = True
                    continue
                tail = (tail + [line.rstrip()])[-25:]
                try: value = json.loads(line)
                except (ValueError, TypeError): continue
                if not isinstance(value, dict): continue
                if "result" in value:
                    result = value["result"]
                elif on_iteration:
                    on_iteration(value)
            code = process.wait(timeout=5)
            if code != 0 and stopping is None:
                raise RuntimeError("本地训练失败：\n" + "\n".join(tail))
            if result is None:
                if stopping is not None: return {"stopped": True}
                raise RuntimeError("训练进程未返回检查点结果：\n" + "\n".join(tail))
            return result
        finally:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()
            reader.join(timeout=1)
