import asyncio
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image
import websockets

ROOT = Path(__file__).resolve().parents[1]


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.process = await asyncio.create_subprocess_exec(sys.executable, str(ROOT/'python/server.py'), '--port', '0',
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        line = await asyncio.wait_for(self.process.stdout.readline(), 15)
        self.port = int(line.decode().split()[-1])
        self.ws = await websockets.connect(f'ws://127.0.0.1:{self.port}')
        self.latest = None
        self.frame = None
        self.scene = None
        await self.until('scene')
        self.sequence = 0

    async def asyncTearDown(self):
        await self.ws.close()
        if self.process.returncode is None:
            self.process.terminate()
            await asyncio.wait_for(self.process.wait(), 5)

    async def until(self, kind, request=None):
        async def receive():
            while True:
                raw = await self.ws.recv()
                if isinstance(raw, bytes):
                    self.frame = raw
                    if kind == 'frame': return raw
                    continue
                message = json.loads(raw)
                if message['type'] == 'status': self.latest = message
                if message['type'] == 'scene': self.scene = message
                if message['type'] == kind and (request is None or message.get('request_id') == request): return message
                if message['type'] == 'error' and kind != 'error': raise AssertionError(message)
        return await asyncio.wait_for(receive(), 15)

    async def command(self, command, **fields):
        self.sequence += 1
        request = str(self.sequence)
        await self.ws.send(json.dumps({'cmd':command, 'request_id':request, **fields}))
        return await self.until('ack', request)

    async def test_render_run_pause_and_single_step(self):
        image = Image.open(io.BytesIO(await self.until('frame')))
        self.assertEqual(image.size, (960,640))
        await self.command('step')
        self.assertAlmostEqual(self.latest['time'], .002)
        await self.command('run')
        await asyncio.sleep(.3)
        await self.command('pause')
        elapsed = self.latest['time']
        self.assertGreater(elapsed, .1)
        self.assertLess(elapsed, .7)
        await asyncio.sleep(.15)
        await self.command('hello')
        self.assertEqual(elapsed, self.latest['time'])
        await self.command('reset')
        self.assertEqual(self.latest['time'], 0)

    async def test_invalid_command_ack_and_recovery(self):
        await self.ws.send(json.dumps({'cmd':'physics','timestep':-1,'request_id':'bad'}))
        error = await self.until('error','bad')
        self.assertIn('数值', error['message'])
        await self.command('add',shape='box')
        self.assertEqual(len(self.scene['bodies']),3)
        await self.command('undo')
        self.assertEqual(len(self.scene['bodies']),2)

    async def test_record_save_reopen_and_csv_over_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            project = str(Path(directory)/'test.mjstudio')
            record = str(Path(directory)/'test.mjrec')
            csv = str(Path(directory)/'test.csv')
            await self.command('record_start')
            await self.command('step',count=100)
            await self.command('record_stop')
            self.assertGreater(self.latest['record_count'],5)
            await self.command('export_recording',path=record)
            await self.command('export_csv',path=csv)
            await self.command('save_project',path=project)
            saved = self.latest['qpos']
            await self.command('reset')
            await self.command('open_project',path=project)
            self.assertEqual(saved,self.latest['qpos'])
            await self.command('open_recording',path=record)
            await self.command('seek',index=3)
            self.assertEqual(self.latest['replay_index'],3)
            self.assertTrue(Path(csv).read_text().startswith('time_s,'))


if __name__ == '__main__': unittest.main(verbosity=2)
