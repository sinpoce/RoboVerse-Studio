# SPDX-License-Identifier: Apache-2.0
"""Loopback WebSocket transport for the task training service."""

from __future__ import annotations

import argparse
import asyncio
import json

import websockets


async def amain(host: str, port: int, make_worker) -> None:
    """Serve the protocol. `make_worker(loop, outbox)` must return a started-able
    thread with a `cmd_q` and emit text or binary events through outbox."""
    loop = asyncio.get_running_loop()
    outbox: "asyncio.Queue" = asyncio.Queue()
    clients: set = set()
    worker = make_worker(loop, outbox)
    worker.start()

    async def broadcaster() -> None:
        while True:
            kind, payload = await outbox.get()
            if not clients:
                continue
            data = json.dumps(payload) if kind == "text" else payload
            dead = []
            for ws in list(clients):
                try:
                    await ws.send(data)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                clients.discard(ws)

    async def handler(ws) -> None:
        clients.add(ws)
        worker.cmd_q.put({"cmd": "hello"})  # let a fresh client resync
        try:
            async for message in ws:
                if isinstance(message, str):
                    try:
                        worker.cmd_q.put(json.loads(message))
                    except json.JSONDecodeError:
                        continue
        except websockets.ConnectionClosed:
            pass
        finally:
            clients.discard(ws)

    asyncio.create_task(broadcaster())
    # Native clients omit Origin. A web page must not be able to launch local
    # training or change imported project paths through a localhost socket.
    async with websockets.serve(handler, host, port, origins=[None]) as server:
        bound = server.sockets[0].getsockname()[1]
        # SidecarManager parses this line to discover the chosen port.
        print(f"MJBATCH_STUDIO_PORT {bound}", flush=True)
        await asyncio.Future()  # serve forever


def main(make_worker) -> None:
    parser = argparse.ArgumentParser(description="RoboVerse Studio sidecar")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 picks a free port")
    args = parser.parse_args()
    try:
        asyncio.run(amain(args.host, args.port, make_worker))
    except KeyboardInterrupt:
        pass
