"""Local workbench service. Request IDs, bounded delivery, real-time scheduling."""
from __future__ import annotations

import argparse
import asyncio
import json
import queue
import threading
import time
import traceback
from pathlib import Path

import mujoco
import numpy as np
import websockets

from studio_world import World, number


class Worker(threading.Thread):
    def __init__(self, loop, deliver):
        super().__init__(daemon=True)
        self.loop, self.deliver = loop, deliver
        self.commands = queue.Queue(maxsize=256)
        self.stop_event = threading.Event()
        self.world = None
        self.render_dirty = True

    def emit(self, payload):
        self.loop.call_soon_threadsafe(self.deliver, payload)

    def handle(self, cmd):
        w = self.world
        kind = cmd.get('cmd')
        scene_changed = False
        message = ''
        if kind == 'hello':
            self.emit({'type': 'hello', 'version': 2, 'engine': f'MuJoCo {mujoco.__version__}'})
            scene_changed = True
        elif kind == 'load':
            w.load(cmd.get('path'), cmd.get('builtin'))
            scene_changed = True
            message = '场景已载入'
        elif kind == 'open_project':
            w.open(cmd['path'])
            scene_changed = True
            message = '项目已打开'
        elif kind == 'save_project':
            w.save(cmd['path'])
            message = '项目已保存'
        elif kind == 'run':
            if w.replay_states is not None and w.replay_index == len(w.replay_times)-1:
                w.seek(0)
            w.running = True
        elif kind == 'pause':
            w.running = False
        elif kind == 'step':
            w.running = False
            w.step(int(number(cmd.get('count', 1), 1, 100)))
        elif kind == 'reset':
            w.reset(int(cmd.get('keyframe', -1)))
            scene_changed = True
        elif kind == 'speed':
            w.speed = number(cmd['value'], .1, 10)
        elif kind == 'select':
            w.selected = int(number(cmd['id'], 0, w.model.nbody-1))
        elif kind == 'pick':
            w.pick(cmd['x'], cmd['y'])
        elif kind == 'camera':
            w.camera_command(cmd)
        elif kind == 'view':
            if 'width' in cmd:
                w.resize(cmd['width'], cmd['height'])
            if 'contacts' in cmd:
                w.options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = bool(cmd['contacts'])
                w.options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = bool(cmd['contacts'])
            if 'joints' in cmd:
                w.options.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = bool(cmd['joints'])
            if 'collision' in cmd:
                w.options.geomgroup[2] = not bool(cmd['collision'])
                w.options.geomgroup[3] = True
        elif kind == 'add':
            w.add(cmd)
            scene_changed = True
        elif kind == 'insert_model':
            w.insert_model(cmd)
            scene_changed = True
            message = '模型已插入当前世界'
        elif kind == 'remove':
            w.remove_selected()
            scene_changed = True
        elif kind == 'add_imu':
            w.add_imu(cmd)
            scene_changed = True
        elif kind == 'body_edit':
            w.body_edit(cmd)
            scene_changed = True
        elif kind == 'physics':
            w.physics(cmd)
            scene_changed = True
        elif kind in ('undo', 'redo'):
            w.undo(kind == 'redo')
            scene_changed = True
        elif kind == 'joint':
            w.set_joint(cmd)
        elif kind == 'control':
            w.control(cmd)
        elif kind == 'load_trajectory':
            w.load_trajectory(cmd['path'])
            message = '轨迹控制已加载；运行后按仿真时间执行，结束时保持最后输入'
        elif kind == 'stop_trajectory':
            w.trajectory = None
            w.data.ctrl[:] = 0
            w.dirty = True
        elif kind == 'force':
            w.force(cmd)
        elif kind == 'record_start':
            if w.recording: raise ValueError('录制已在进行')
            w.start_recording()
            message = '开始录制物理状态（50 Hz）'
        elif kind == 'record_stop':
            w.recording = False
            message = f'录制已停止：{len(w.record_times)} 帧'
        elif kind == 'replay':
            w.replay()
            scene_changed = True
        elif kind == 'seek':
            w.running = False
            w.seek(int(cmd['index']))
        elif kind == 'replay_exit':
            w.running = False
            w.replay_states = w.replay_times = None
            w.data.xfrc_applied[:] = 0
            w.dirty = True
        elif kind == 'open_recording':
            w.open(cmd['path'], recording=True)
            scene_changed = True
        elif kind == 'export_recording':
            w.export_recording(cmd['path'])
            message = '录制文件已导出'
        elif kind == 'export_csv':
            w.export_csv(cmd['path'])
            message = 'CSV 数据已导出'
        else:
            raise ValueError(f'未知命令：{kind}')
        if scene_changed:
            self.emit(w.scene())
        self.render_dirty = True
        self.emit(w.status())
        if cmd.get('request_id'):
            self.emit({'type': 'ack', 'request_id': cmd['request_id'], 'command': kind,
                       'message': message, 'path': cmd.get('path', '')})

    def run(self):
        try:
            self.world = World()
        except Exception as exc:
            self.emit({'type': 'error', 'message': f'无法启动物理引擎：{exc}'})
            return
        w = self.world
        last_clock = time.perf_counter()
        last_render = last_status = last_stats = last_clock
        last_sim = w.data.time
        accumulator = 0.0
        try:
            while not self.stop_event.is_set():
                for _ in range(24):
                    try:
                        cmd = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        self.handle(cmd)
                        if cmd.get('cmd') not in ('camera', 'view', 'select', 'pick', 'control'):
                            accumulator = 0
                            last_clock = time.perf_counter()
                    except Exception as exc:
                        w.running = False
                        self.emit({'type': 'error', 'request_id': cmd.get('request_id'),
                                   'message': str(exc), 'command': cmd.get('cmd')})
                now = time.perf_counter()
                elapsed = min(now-last_clock, .1)
                last_clock = now
                if w.running:
                    try:
                        if w.replay_states is not None:
                            accumulator += elapsed*w.speed
                            target = w.replay_times[w.replay_index] + accumulator
                            index = min(len(w.replay_times)-1, int(np.searchsorted(w.replay_times, target, side='right')-1))
                            if index > w.replay_index:
                                accumulator -= w.replay_times[index]-w.replay_times[w.replay_index]
                                w.seek(index)
                            if w.replay_index == len(w.replay_times)-1:
                                w.running = False
                        else:
                            accumulator = min(.1*w.speed, accumulator + elapsed*w.speed)
                            deadline = time.perf_counter() + .008
                            while accumulator >= w.model.opt.timestep and time.perf_counter() < deadline:
                                w.step()
                                accumulator -= w.model.opt.timestep
                        self.render_dirty = True
                    except Exception as exc:
                        w.running = False
                        self.emit({'type': 'error', 'message': f'仿真停止：{exc}'})
                else:
                    accumulator = 0
                if now-last_stats >= .5:
                    w.actual_rtf = max(0, (w.data.time-last_sim)/(now-last_stats)) if w.running else 0
                    last_stats, last_sim = now, w.data.time
                if self.render_dirty and now-last_render >= 1/30:
                    try:
                        self.emit(w.render())
                        self.render_dirty = False
                    except Exception as exc:
                        self.emit({'type': 'error', 'message': f'渲染失败：{exc}'})
                    last_render = now
                if now-last_status >= .1:
                    self.emit(w.status())
                    last_status = now
                self.stop_event.wait(.002)
        finally:
            w.close()


async def amain(port):
    loop = asyncio.get_running_loop()
    clients = {}
    def deliver(payload):
        encoded = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False, allow_nan=False)
        for ws, outbox in list(clients.items()):
            if outbox.full():
                # Slow consumers must not accumulate old video frames or stall physics.
                if isinstance(payload, bytes) or payload.get('type') == 'status':
                    continue
                asyncio.create_task(ws.close(code=1013, reason='Client cannot keep up'))
                continue
            outbox.put_nowait(encoded)
    worker = Worker(loop, deliver)
    worker.start()
    async def handler(ws):
        outbox = asyncio.Queue(maxsize=64)
        clients[ws] = outbox
        async def sender():
            while True:
                await ws.send(await outbox.get())
        task = asyncio.create_task(sender())
        try:
            worker.commands.put_nowait({'cmd': 'hello'})
            async for raw in ws:
                if not isinstance(raw, str): continue
                try:
                    cmd = json.loads(raw)
                    if not isinstance(cmd, dict) or not isinstance(cmd.get('cmd'), str):
                        raise ValueError('命令必须是含 cmd 的 JSON 对象')
                    worker.commands.put_nowait(cmd)
                except (ValueError, queue.Full) as exc:
                    await ws.send(json.dumps({'type': 'error', 'message': str(exc) or '命令队列已满'}))
        except websockets.ConnectionClosed:
            pass
        finally:
            clients.pop(ws, None)
            task.cancel()
            # A disconnected GUI must not leave an unattended simulation running.
            if not clients:
                try: worker.commands.put_nowait({'cmd': 'pause'})
                except queue.Full: pass
    try:
        async with websockets.serve(handler, '127.0.0.1', port, max_size=65536, origins=[None]) as server:
            print(f'MJBATCH_STUDIO_PORT {server.sockets[0].getsockname()[1]}', flush=True)
            await asyncio.Future()
    finally:
        worker.stop_event.set()
        await asyncio.to_thread(worker.join, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=0)
    args = parser.parse_args()
    try:
        asyncio.run(amain(args.port))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
