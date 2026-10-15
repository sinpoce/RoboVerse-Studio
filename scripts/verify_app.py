"""Verify a relocated bundle, its physics service and signature without using source paths."""
import asyncio
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image
import websockets


async def verify_training(resources, directory, env):
    project = Path(os.environ.get("ROBOVERSE_STAGEWISE_PROJECT", Path.home()/"Downloads/Stage-Wise-CMORL-main"))
    if not (project/"algos/comoppo/agent.py").is_file():
        return {"status": "skipped", "reason": "Set ROBOVERSE_STAGEWISE_PROJECT to a Stage-Wise checkout"}
    process = await asyncio.create_subprocess_exec(
        str(resources/"runtime/bin/python3"), "-B", str(resources/"examples/generic_train_server.py"),
        "--port", "0", cwd=directory, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    frames = []
    try:
        line = await asyncio.wait_for(process.stdout.readline(), 30)
        if not line: raise RuntimeError((await process.stderr.read()).decode())
        port = int(line.decode().split()[-1])
        async with websockets.connect(f"ws://127.0.0.1:{port}") as socket:
            async def until(kind, predicate=lambda m: True):
                async def receive():
                    while True:
                        raw = await socket.recv()
                        if isinstance(raw, bytes):
                            frames.append(raw)
                            continue
                        message = json.loads(raw)
                        if message["type"] == "error": raise RuntimeError(message)
                        if message["type"] == kind and predicate(message): return message
                return await asyncio.wait_for(receive(), 60)

            await until("training_config")
            await socket.send(json.dumps({"cmd": "task_config", "path": str(project)}))
            settings = await until("training_config", lambda m: "project_comoppo" in m["algo"])
            config = json.loads(settings["config_json"])
            config.update(num_envs=2, horizon=32, iterations=2)
            checkpoint = Path(directory)/"training/policy.pt"
            await socket.send(json.dumps({"cmd": "start_training", "config_json": json.dumps(config),
                                          "checkpoint": str(checkpoint)}))
            await until("log", lambda m: "训练完成" in m.get("message", ""))
            await until("status", lambda m: not m.get("running"))
            metadata = json.loads((checkpoint.parent/"policy.run/experiment.json").read_text())
            assert checkpoint.is_file() and metadata["total_steps"] == 128
            assert metadata["actor_parameter_delta"] > 0
            assert frames
            frame = Image.open(io.BytesIO(frames[0]))
            frame.load()
            return {"status": "passed", "backend": metadata["backend"], "steps": metadata["total_steps"],
                    "actor_parameter_delta": metadata["actor_parameter_delta"], "preview": list(frame.size)}
    finally:
        if process.returncode is None:
            process.terminate()
            await asyncio.wait_for(process.wait(), 10)


async def main():
    source = Path(__file__).resolve().parents[1] / 'dist/RoboVerse Studio.app'
    with tempfile.TemporaryDirectory(prefix='mjstudio-relocation-') as directory:
        bundle = Path(directory) / 'Relocated Studio.app'
        shutil.copytree(source, bundle, symlinks=True)
        resources = bundle / 'Contents/Resources'
        verify = ['codesign', '--verify', '--deep', '--strict', str(bundle)]
        subprocess.run(verify, check=True)
        env = {key: value for key, value in os.environ.items() if key not in
               ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'MJBATCH_STUDIO_HOME', 'MJBATCH_STUDIO_PYTHON', 'MJBATCH_STUDIO_SERVER')}
        env.update(PATH='/usr/bin:/bin:/usr/sbin:/sbin', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
        process = await asyncio.create_subprocess_exec(
            str(resources/'runtime/bin/python3'), '-B', str(resources/'python/server.py'), '--port', '0',
            cwd=directory, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            line = await asyncio.wait_for(process.stdout.readline(), 20)
            port = int(line.decode().split()[-1])
            async with websockets.connect(f'ws://127.0.0.1:{port}') as socket:
                async def until(kind, request=None):
                    async def receive():
                        while True:
                            raw = await socket.recv()
                            if isinstance(raw, bytes):
                                if kind == 'frame': return raw
                                continue
                            message = json.loads(raw)
                            if message['type'] == 'error': raise RuntimeError(message)
                            if message['type'] == kind and (request is None or request == message.get('request_id')):
                                return message
                    return await asyncio.wait_for(receive(), 20)
                await until('hello')
                await socket.send(json.dumps({'cmd': 'open_project', 'path': str(resources/'projects/机器人实验场景.mjstudio'), 'request_id': 'open'}))
                scene = await until('scene')
                while scene['nu'] != 29: scene = await until('scene')
                await until('ack', 'open')
                image = Image.open(io.BytesIO(await until('frame')))
                image.load()
                await socket.send(json.dumps({'cmd': 'step', 'count': 10, 'request_id': 'step'}))
                await until('ack', 'step')
                saved = str(Path(directory)/'relocated.mjstudio')
                await socket.send(json.dumps({'cmd': 'save_project', 'path': saved, 'request_id': 'save'}))
                await until('ack', 'save')
                assert Path(saved).is_file()
                assert len(scene['sensors']) == 6
        finally:
            if process.returncode is None:
                process.terminate()
                await asyncio.wait_for(process.wait(), 5)
        training = await verify_training(resources, directory, env)
        subprocess.run(verify, check=True)
        print(json.dumps({'relocation': 'passed', 'robot_actuators': scene['nu'],
                          'sensors': len(scene['sensors']), 'render': list(image.size),
                          'save_project': 'passed', 'training': training,
                          'signature_after_running': 'valid'}, ensure_ascii=False))


if __name__ == '__main__': asyncio.run(main())
