"""
ROS 2 -> browser bridge for the 3D viewer (the same viewer as web3d_drones_edge/server_edge.py).

It subscribes to the drone's topics, turns them into the messages the viewer understands and streams them over a WebSocket, and it serves the viewer
page. What the viewer sends back (weather, light, sensors, pause, speed, new goal) is published on /edge_residual/control, so the browser is just another
ROS client. Only the static obstacles, vehicles and trees are regenerated here from the world seed (they are the simulated world, not sensor data).

    http://<host>:<port>/          the viewer         ws://<host>:<port>/ws     the stream

Subscribes: /edge_residual/status, /rays, /kpi, /detections, /camera/annotated, /gcs.  Publishes: /edge_residual/control.  Parameters: port (8010), host, world_seed, log_dir (the *.jsonl replay files shown in the replay drawer).
"""

import asyncio
import base64
import json
import math
import mimetypes
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

from .spin_util import spin
from . import aerial
from . import flyers as FL                                                 # the roaming birds and drones: positions come from the step number (see flyers.py)
from .core import CHUNK, CONDITIONS, DroneEnv
from . import city_world as CW
from .replay_lib import Replay, list_replays

try:                                                  # orjson (optional) encodes / decodes the 30 messages per second several times faster than json
    import orjson
    _dumps = lambda o: orjson.dumps(o).decode()           # the viewer reads text frames, so the bytes are decoded
    _loads = orjson.loads
except ImportError:
    _dumps, _loads = json.dumps, json.loads

VIEW_CHUNKS = 4
BRAIN_NAME = 'adaptive-weight residual SAC on PID'


class WebBridge(Node):

    def __init__(self):
        super().__init__('web_bridge', start_parameter_services=False)
        self.declare_parameter('port', 8019)
        self.declare_parameter('host', '0.0.0.0')
        self.declare_parameter('world_seed', 1234)
        self.declare_parameter('log_dir', '/tmp/edge_residual')
        self.port = int(self.get_parameter('port').value)
        self.host = str(self.get_parameter('host').value)
        self.log_dir = str(self.get_parameter('log_dir').value)
        self.world_seed = int(self.get_parameter('world_seed').value)
        self.world = DroneEnv(seed=self.world_seed)
        self.web_dir = os.path.join(get_package_share_directory('edge_residual_adaptive'), 'web')
        self._status, self._status_raw, self._status_seen, self._status_count = {}, None, None, 0       # the status text is parsed only when the viewer is sent something (30 times a second, not 60)
        self.rays, self.kpi, self.det, self.gcs = [], {}, {}, {}
        self.traffic, self._traffic_count = None, 0                       # the latest /traffic message, and how many arrived
        self.camera_png, self.camera_version = None, 0
        self.light, self.sensors = 'day', 'fused'
        self.last_every = {'weather': 30.0, 'light': 20.0, 'mode': 'cycle'}      # what the page shows while an automatic mode is off
        self.create_subscription(String, '/edge_residual/status_slow', self.on_status, 10)
        self.create_subscription(String, '/edge_residual/kpi', lambda m: self.load('kpi', m), 10)
        self.create_subscription(String, '/edge_residual/detections', lambda m: self.load('det', m), 5)
        self.create_subscription(String, '/edge_residual/gcs', lambda m: self.load('gcs', m), 5)
        self.create_subscription(Float32MultiArray, '/edge_residual/rays', self.on_rays, 10)
        self.create_subscription(Float32MultiArray, '/edge_residual/traffic', self.on_traffic, 2)
        self.create_subscription(Float32MultiArray, '/edge_residual/flyers', self.on_flyers, 2)
        self._flyers_count = 0
        self.create_subscription(Image, '/edge_residual/camera/annotated', self.on_image, 5)
        self.control = self.create_publisher(String, '/edge_residual/control', 10)
        self.create_subscription(String, '/edge_residual/control', self.on_control, 10)   # the light and the sensor set can be changed from anywhere
        threading.Thread(target=self.run_server, daemon=True).start()
        self.get_logger().info(f'viewer: http://localhost:{self.port}/')

    # ---- ROS callbacks ----------------------------------------------------------------
    def on_status(self, m):
        self._status_raw = m.data
        self._status_count += 1

    @property
    def status(self):
        raw = self._status_raw
        if raw is not self._status_seen:
            self._status_seen = raw
            try:
                self._status = _loads(raw)
            except ValueError:
                pass
        return self._status

    def load(self, name, m):
        try:
            setattr(self, name, _loads(m.data))
        except ValueError:
            pass

    def on_control(self, m):
        try:
            c = json.loads(m.data)
        except ValueError:
            return
        if c.get('cmd') == 'light' and c.get('value') in aerial.LIGHT:
            self.light = c['value']
        elif c.get('cmd') == 'sensors':
            self.sensors = c.get('value', self.sensors)

    def on_traffic(self, m):
        self.traffic = m.data                       # turned into a message only when it is sent (15 a second)
        self._traffic_count += 1

    def on_flyers(self, m):
        self.world.flyers_on, self.world.flyer_t = bool(m.data[1]), int(m.data[0])
        self._flyers_count += 1

    def flyers_message(self, x, y):
        """The roaming birds and patrol drones (each a real moving obstacle): spot, member, kind, hash, size, centre, velocity per second (60 steps a second)."""
        rows = FL.flyers_near(self.world, x, y, VIEW_CHUNKS, self.world.flyer_t)
        return {'type': 'flyers', 'f': [[round(v, 1) for v in r[:12]] + [round(r[12] * 60.0, 1), round(r[13] * 60.0, 1), round(r[14] * 60.0, 1)] for r in rows]}

    def cars_message(self):
        d = self.traffic
        n = int(d[1])
        return {'type': 'cars', 't': round(d[0], 2),
                'cars': [[round(d[2 + 7 * i], 1), round(d[3 + 7 * i], 1), round(d[4 + 7 * i], 2), int(d[5 + 7 * i]), int(d[6 + 7 * i]), int(d[7 + 7 * i]), round(d[8 + 7 * i], 1)] for i in range(n)]}

    def on_rays(self, m):
        self.rays = m.data                        # turned into numbers only when it is sent

    def on_image(self, m):
        img = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, 3)
        ok, png = cv2.imencode('.png', img)
        if ok:
            self.camera_png = base64.b64encode(png.tobytes()).decode()
            self.camera_version += 1

    def send_control(self, cmd, value=None):
        m = String()
        m.data = json.dumps({'cmd': cmd, 'value': value})
        self.control.publish(m)

    # ---- messages for the viewer ---------------------------------------------------------
    def state_message(self):
        s, k, g = self.status, self.kpi, self.gcs
        r = self.rays
        return {
            'type': 'state',
            'drone': {'x': s['x'], 'y': s['y'], 'z': s['z'], 'yaw': s['yaw'], 'vz': s['vz'], 'colliding': s['colliding']},
            'sensors': [[round(r[i], 1), round(r[i + 1], 1), round(r[i + 2], 1), round(r[i + 3], 3)] for i in range(0, len(r) - 3, 4)],
            'goal': {'x': s['goal'][0], 'y': s['goal'][1], 'z': s['goal'][2]}, 'goalRadius': s['goal_radius'],
            'wind': s['wind'], 'weather': s['weather'], 'brain': BRAIN_NAME, 'light': self.light, 'sensorSet': self.sensors, 'battery': s['battery'],
            'goals': k.get('goals', s['goals']), 'totalGoals': s['goals'], 'steps': k.get('steps', s['step']), 'collisions': k.get('collisions', s['collisions']),
            'totalCollisions': s['collisions'], 'respawns': s['respawns'],             # goals / steps / collisions since the last weather change (the KPI node), totals since the start
            'collPer1k': k.get('collisions_per_1k', 0), 'goalsPer1k': k.get('goals_per_1k', 0), 'energyPerGoal': k.get('energy_per_goal'),
            'groundSpeed': k.get('ground_speed', 0),
            'telemetry': {'frames': g.get('frames', 0), 'bytes': g.get('bytes', 0), 'crcErrors': g.get('crc_errors', 0), 'lost': g.get('lost', 0)},
            'paused': s['paused'], 'brains': [BRAIN_NAME], 'weathers': list(CONDITIONS), 'hasDetector': bool(self.det), 'auto': self.auto_message(s),
        }

    def auto_message(self, s):
        a = s.get('auto', {'weather': {}, 'light': {}})
        w, l = a['weather'], a['light']
        if w.get('seconds', 0) > 0:
            self.last_every['weather'], self.last_every['mode'] = w['seconds'], w.get('mode', 'cycle')
        if l.get('seconds', 0) > 0:
            self.last_every['light'] = l['seconds']
        return {'weather': {'on': w.get('seconds', 0) > 0, 'every': self.last_every['weather'], 'mode': self.last_every['mode']},
                'light': {'on': l.get('seconds', 0) > 0, 'every': self.last_every['light']}}

    def world_messages(self, x, y):
        boxes = self.world.chunk_boxes_around(x, y, VIEW_CHUNKS)
        t = aerial.targets_near(self.world, x, y, VIEW_CHUNKS * CHUNK)
        return [{'type': 'obstacles', 'boxes': [[round(float(v), 1) for v in b] for b in boxes]},
                {'type': 'targets',
                 'vehicles': [],                                  # the sim's random cars are replaced by the city's cars: parked (here) and driving (the 'cars' message)
                 'parked': [[round(px, 1), round(py, 1), round(h, 2), ty, CW.COLORS.index(col)] for px, py, h, ty, col in CW.parked_near(self.world, x, y, VIEW_CHUNKS)],
                 'people': [[round(a, 1), round(b, 1)] for a, b, _, _ in t['people']],
                 'trees': [[round(a, 1), round(b, 1), round(c, 1)] for a, b, c in t['trees']],
                 'rocks': [[round(a, 1), round(b, 1), round(c, 1)] for a, b, c, _, _ in t['rocks']]}]

    def camera_message(self):
        d = self.det
        return {'type': 'camera', 'png': self.camera_png, 'found': d.get('found', [0, 0, 0]), 'truth': d.get('truth', [0, 0, 0]),
                'names': d.get('names', aerial.CLASSES), 'alt': d.get('alt', 0), 'light': d.get('light', self.light)}

    # ---- what the browser asks for ---------------------------------------------------------------
    def on_browser(self, raw):
        try:
            msg = json.loads(raw)
        except ValueError:
            return None
        if msg.get('type') != 'control':
            return None
        a, v = msg.get('action'), msg.get('value')
        if a in ('pause', 'resume', 'new_goal'):
            self.send_control(a)
        elif a == 'speed':
            self.send_control('speed', float(v))
        elif a == 'weather' and v in CONDITIONS:
            self.send_control('weather', v)
        elif a == 'light':
            self.light = v
            self.send_control('light', v)
        elif a == 'sensors':
            self.sensors = v
            self.send_control('sensors', v)
        elif a == 'auto_weather' and isinstance(v, dict):
            self.send_control('auto_weather', {'seconds': max(5.0, float(v.get('every', 30))) if v.get('on') else 0, 'mode': v.get('mode', 'cycle')})
        elif a == 'auto_light' and isinstance(v, dict):
            self.send_control('auto_light', {'seconds': max(5.0, float(v.get('every', 20))) if v.get('on') else 0})
        return None

    async def on_replay(self, sess, raw, ws):
        """Replay commands of the drawer. Returns a reply for the browser, or None (also when the message is not about replays)."""
        try:
            msg = json.loads(raw)
        except ValueError:
            return None
        if msg.get('type') != 'control':
            return None
        a, v = msg.get('action'), msg.get('value')
        r = sess['replay']
        if a == 'list_replays':
            return {'type': 'replays', 'files': list_replays(self.log_dir)}
        if a == 'replay_start':
            await ws.send(_dumps({'type': 'replays', 'files': list_replays(self.log_dir), 'error': f'Loading {v} ... the first time takes a few seconds'}))
            try:
                sess['replay'] = await asyncio.get_running_loop().run_in_executor(None, Replay, self.log_dir, str(v or ''), self.world_seed)
            except (ValueError, OSError):
                return {'type': 'replays', 'files': list_replays(self.log_dir), 'error': 'could not load that replay'}
        elif r is not None and a == 'replay_pause':
            r.toggle_pause()
        elif r is not None and a == 'replay_seek':
            r.seek(float(v or 0))
        elif r is not None and a == 'speed':
            r.speed = min(max(float(v or 1.0), 0.1), 16.0)
            return {}
        elif a == 'replay_stop':
            sess['replay'] = None
        elif r is not None and a in ('pause', 'resume', 'new_goal', 'weather'):
            return {}                                     # the live flight is not changed while a replay is on the screen
        return None

    # ---- web server (its own asyncio loop in a thread) --------------------------------------------
    def serve_file(self, path):
        rel = 'index.html' if path in ('', '/') else path.lstrip('/')
        if rel.startswith('static/'):
            rel = rel[len('static/'):]
        full = os.path.normpath(os.path.join(self.web_dir, rel))
        if not full.startswith(os.path.normpath(self.web_dir)) or not os.path.isfile(full):
            return Response(404, 'Not Found', Headers([('Content-Type', 'text/plain')]), b'not found')
        with open(full, 'rb') as fh:
            body = fh.read()
        return Response(200, 'OK', Headers([('Content-Type', mimetypes.guess_type(full)[0] or 'application/octet-stream'),
                                            ('Content-Length', str(len(body))), ('Cache-Control', 'no-cache')]), body)

    def run_server(self):
        async def process_request(connection, request):
            if request.path.split('?')[0] == '/ws':
                return None
            return self.serve_file(request.path.split('?')[0])

        async def handler(ws):
            sess = {'replay': None}                       # the replay this browser is watching, if any

            async def send_replay(r, last_t, last_push, last_cam):
                now = time.perf_counter()
                r.advance(now - last_t)
                if now - last_push >= 1.0 / 30.0:
                    await ws.send(_dumps(r.state_message(self.light, self.sensors, [BRAIN_NAME])))
                    if r.chunk_changed():
                        for m in r.world_messages():
                            await ws.send(_dumps(m))
                    last_push = now
                if now - last_cam >= 0.5:
                    await ws.send(_dumps(r.camera_message(self.light)))
                    last_cam = now
                return now, last_push, last_cam

            async def writer():
                last_chunk, last_cam = None, -1
                sent_count, sent_at = -1, 0.0
                cars_count, cars_at = -1, 0.0
                flyers_count, flyers_at = -1, 0.0
                rt, rpush, rcam = time.perf_counter(), 0.0, 0.0
                while True:
                    r = sess['replay']
                    if r is not None:
                        rt, rpush, rcam = await send_replay(r, rt, rpush, rcam)
                        last_chunk, last_cam, sent_count = None, -1, -1      # the live view starts again from scratch afterwards
                    elif self.status and len(self.rays):
                        rt = time.perf_counter()
                        now = time.perf_counter()
                        # send a state when a new one has arrived (not on a separate 30 Hz clock: two clocks of the same rate beat against each other and the
                        # drone seemed to hop), but not more than about 70 times a second
                        if self._status_count != sent_count and now - sent_at >= 0.014:
                            sent_count, sent_at = self._status_count, now
                            await ws.send(_dumps(self.state_message()))
                            chunk = (int(math.floor(self.status['x'] / CHUNK)), int(math.floor(self.status['y'] / CHUNK)))
                            if chunk != last_chunk:
                                last_chunk = chunk
                                for m in self.world_messages(self.status['x'], self.status['y']):
                                    await ws.send(_dumps(m))
                        if self.traffic is not None and self._traffic_count != cars_count and now - cars_at >= 1.0 / 15.0:
                            cars_count, cars_at = self._traffic_count, now
                            await ws.send(_dumps(self.cars_message()))
                        if self.world.flyers_on and self._flyers_count != flyers_count and now - flyers_at >= 1.0 / 15.0:
                            flyers_count, flyers_at = self._flyers_count, now
                            await ws.send(_dumps(self.flyers_message(self.status['x'], self.status['y'])))
                        if self.camera_version != last_cam and self.camera_png:
                            last_cam = self.camera_version
                            await ws.send(_dumps(self.camera_message()))
                    await asyncio.sleep(0.004)

            async def reader():
                async for raw in ws:
                    reply = await self.on_replay(sess, raw, ws)
                    if reply is None and sess['replay'] is None:
                        reply = self.on_browser(raw)
                    if reply:
                        await ws.send(_dumps(reply))

            tasks = [asyncio.ensure_future(writer()), asyncio.ensure_future(reader())]
            try:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for t in tasks:
                    t.cancel()

        async def main():
            async with serve(handler, self.host, self.port, process_request=process_request, max_size=None):
                await asyncio.Future()

        try:
            asyncio.run(main())
        except OSError as e:
            self.get_logger().error(f'cannot serve the viewer on {self.host}:{self.port}: {e}. Stop the other program on that port, or pass port:=<another port>.')
            rclpy.try_shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = WebBridge()
    try:
        spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
