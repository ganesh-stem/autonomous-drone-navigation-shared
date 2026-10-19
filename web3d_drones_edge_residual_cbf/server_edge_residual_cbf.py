"""
Live 3D demo of the edge drone flown by a RESIDUAL SAC brain (PID bias) with a real CONTROL-BARRIER-FUNCTION shield
stacked on top -- see edge_v3_14_residual_cbf_sac.py. The actor still gets PID_WEIGHT * pid_bias(s) added to its
logits, exactly as in web3d_drones_edge_residual; on top of that, every chosen action (both the greedy one here and
random exploration) is filtered through the CBF shield_mask() below before it is used.

    python server_edge_residual_cbf.py              then open http://localhost:8021

This folder reuses the shared environment / rendering modules (aerial, drone_env, env_edge_v2, detector,
telemetry) straight from ../web3d_drones_edge/src, so nothing there needs to be duplicated.
"""

import argparse
import asyncio
import base64
import json
import math
import random
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import sys
from pathlib import Path as _P
HERE = Path(__file__).resolve().parent
EDGE_SRC = HERE.parent / 'web3d_drones_edge' / 'src'          # reuse the original project's modules
sys.path.insert(0, str(EDGE_SRC))

import aerial
import drone_env as D
from detector import MODALITIES, Detector, decode
from env_edge import CONDITIONS
from env_edge_v2 import EdgeDroneEnvV2                         # 9 actions (this also extends drone_env.ACTIONS in place)
from telemetry import GroundStation, Telemetry

TICK_HZ = 60.0
PUSH_HZ = 30.0
VIEW_CHUNKS = 4
CAMERA_EVERY_S = 0.5
RANDOM_STEPS_AFTER_RESPAWN = 90

YAW_DEADBAND = 0.08
ELEV_DEADBAND = 0.15
PID_BEARING_IDX, PID_ELEV_IDX = 28, 30
PID_WEIGHT = 1.5
SHIELD_MARGIN = 8.0
CBF_ALPHA = 0.6

BRAINS = {
    'residual SAC on PID + CBF shield': HERE / 'results' / 'models' / 'sac_residual_cbf' / 'brain_final.pth',
}
LOG_DIR = HERE / 'logs'
try:
    from orjson import loads as _loads
except ImportError:
    _loads = json.loads

REPLAY_NAME_RE = re.compile(r'^[\w\-. ]+\.jsonl$')
LIGHT_CYCLE = ['day', 'dusk', 'night', 'dusk']
COLORS = [(0, 0, 255), (0, 255, 255), (255, 128, 0)]


def pid_bias(s):
    """Which of the 9 actions a classical (heuristic, proportional-with-deadband) controller would prefer right
    now, from the same bearing/elevation-to-goal signals already in the state. See edge_v3_9_residual_pid_sac.py."""
    bearing, elev = float(s[PID_BEARING_IDX]), float(s[PID_ELEV_IDX])
    idx = 3 if elev > ELEV_DEADBAND else 4 if elev < -ELEV_DEADBAND else \
        1 if bearing > YAW_DEADBAND else 2 if bearing < -YAW_DEADBAND else 0
    bias = np.zeros(len(D.ACTIONS), dtype=np.float32)
    bias[idx] = 1.0
    return bias


def shield_mask(env, margin=SHIELD_MARGIN, alpha=CBF_ALPHA):
    """CONTROL-BARRIER-FUNCTION safety filter, stacked on top of the residual-PID actor. See
    edge_v3_10_cbf_shield.py / edge_v3_14_residual_cbf_sac.py for the full derivation. Returns a boolean array over
    the 9 actions; if none pass, the one with the most room is allowed."""
    est = env.wind_estimate if getattr(env, 'wind_aware', False) else np.zeros(3)
    dist = (1.0 - np.asarray(env.signals)) * D.SENSOR_MAX_DIST
    ok = np.zeros(len(D.ACTIONS), dtype=bool)
    room = np.zeros(len(D.ACTIONS))
    for a, (yaw_d, fwd, vdir) in enumerate(D.ACTIONS):
        move = env.speed * fwd
        yr = math.radians(yaw_d)
        d = np.array([move * math.cos(yr) + est[0], move * math.sin(yr) + est[1], vdir * D.CLIMB_RATE * (env.speed / D.SPEED_FAST) + est[2]])
        length = float(np.linalg.norm(d))
        if length < 1e-6:
            ok[a], room[a] = True, D.SENSOR_MAX_DIST
            continue
        dots = D._LOCAL_DIRS @ (d / length)
        near = dots > 0.85
        ray = float(dist[near].min()) if near.any() else float(dist[int(np.argmax(dots))])
        h_now = ray - margin
        h_next = (ray - length) - margin
        if env.z + d[2] - D.DRONE_RADIUS < 3.0:
            h_next = min(h_next, -1.0)
        room[a] = h_next
        ok[a] = h_next >= (1.0 - alpha) * h_now
    if not ok.any():
        ok[int(np.argmax(room))] = True
    return ok


class SacNumpyBrain:
    """The actor half of a trained residual-SAC checkpoint: PID_WEIGHT * pid_bias(s) + the network's own logits,
    exactly as it was trained. A finished SAC brain is tested with its most probable action (greedy)."""

    def __init__(self, path):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        sd = ck['state_dict'] if 'state_dict' in ck else ck
        self.w = [(sd[f'actor.{i}.weight'].numpy().T.astype(np.float32).copy(), sd[f'actor.{i}.bias'].numpy().astype(np.float32).copy())
                  for i in (0, 2, 4)]
        self.n_inputs = self.w[0][0].shape[0]
        self.n_actions = self.w[2][0].shape[1]

    def logits(self, s):
        (w1, b1), (w2, b2), (w3, b3) = self.w
        h = np.maximum(s @ w1 + b1, 0.0)
        h = np.maximum(h @ w2 + b2, 0.0)
        return h @ w3 + b3 + PID_WEIGHT * pid_bias(s)

    def act(self, s, mask=None):
        z = self.logits(np.asarray(s, dtype=np.float32))
        if mask is not None:
            z = np.where(mask, z, -1e9)
        return int(np.argmax(z))


class Sim:
    def __init__(self, seed=1234):
        self.seed = seed
        self.brains = {n: SacNumpyBrain(p) for n, p in BRAINS.items() if p.is_file()}
        self.brain_name = next(iter(self.brains)) if self.brains else None
        self.weather = 'windy'
        self.light = 'day'
        self.auto_weather = dict(on=True, every=30.0, mode='cycle')
        self.auto_light = dict(on=True, every=20.0)
        self.weather_steps = self.light_steps = 0
        self.light_phase = 0
        self.pick = random.Random(2)
        self.sensors = 'fused'
        self.paused = False
        self.speed = 1.0
        self.rng = random.Random(1)
        self.detector = None
        det_path = HERE.parent / 'web3d_drones_edge' / 'results' / 'models' / 'detector.pt'
        if det_path.is_file():
            self.detector = Detector()
            self.detector.load_state_dict(torch.load(det_path, map_location='cpu'))
            self.detector.eval()
        self.new_env()

    def new_env(self):
        self.env = EdgeDroneEnvV2(seed=self.seed, rng_seed=5, weather=self.weather, wind_aware=True)
        self.random_steps = 0
        self.steps = 0
        self.weather_steps = 0
        self.collisions = 0
        self.respawns = 0
        self.tm = Telemetry()
        self.gs = GroundStation()
        self._energy_base = 0.0          # energyPerGoal is measured since the last weather change; battery/energy itself is never reset
        self._goals_base = 0             # env.goals_reached is lifetime; this is the snapshot the per-weather 'goals' KPI subtracts
        self._collisions_total = 0       # lifetime collisions, never reset by change_weather (unlike self.collisions)

    def step(self):
        if self.brain_name is None:
            return
        brain = self.brains[self.brain_name]
        env = self.env
        mask = shield_mask(env)
        a = self.rng.choice([i for i in range(brain.n_actions) if mask[i]]) if self.random_steps > 0 else brain.act(env.get_state(), mask)
        self.random_steps = max(0, self.random_steps - 1)
        resp = env.step(a)
        self.steps += 1
        self.weather_steps += 1
        self.light_steps += 1
        if self.auto_weather['on'] and self.weather_steps >= self.auto_weather['every'] * 60:
            self.change_weather(self.next_weather())
        if self.auto_light['on'] and self.light_steps >= self.auto_light['every'] * 60:
            self.light_phase = (self.light_phase + 1) % len(LIGHT_CYCLE)
            self.light, self.light_steps = LIGHT_CYCLE[self.light_phase], 0
        self.collisions += int(env.is_colliding)
        self._collisions_total += int(env.is_colliding)
        if resp:
            self.respawns += 1
            self.random_steps = RANDOM_STEPS_AFTER_RESPAWN
        for f in self.tm.step_frames(env, self.steps):
            self.gs.feed(f)

    def change_weather(self, name):
        self.weather = name
        self.env.set_weather(name)
        self.steps = self.weather_steps = self.collisions = self.respawns = 0
        self._goals_base = self.env.goals_reached   # env.goals_reached is never reset -- 'goals' below is measured since this point
        self._energy_base = self.env.energy   # battery is never reset -- only energyPerGoal is measured since this point
        self.env.ground_dist = 0.0

    def next_weather(self):
        names = list(CONDITIONS)
        if self.auto_weather['mode'] == 'random':
            return self.pick.choice([n for n in names if n != self.weather])
        return names[(names.index(self.weather) + 1) % len(names)]

    def set_light(self, name):
        self.light, self.light_steps = name, 0
        self.light_phase = LIGHT_CYCLE.index(name)

    # ── messages ────────────────────────────────────────────────────────────────
    def state_message(self):
        e = self.env
        ends = e.sensor_end
        goals = e.goals_reached - self._goals_base   # since the last weather change (matches the KPI drawer's label)
        return {
            'type': 'state',
            'drone': {'x': e.x, 'y': e.y, 'z': e.z, 'yaw': e.yaw, 'vz': e.vz, 'colliding': bool(e.is_colliding)},
            'sensors': [[round(float(ends[i, 0]), 1), round(float(ends[i, 1]), 1), round(float(ends[i, 2]), 1), round(e.signals[i], 3)]
                        for i in range(D.N_SENSORS)],
            'goal': {'x': float(e.goal[0]), 'y': float(e.goal[1]), 'z': float(e.goal[2])}, 'goalRadius': D.GOAL_RADIUS,
            'wind': [round(float(v), 3) for v in e.wind], 'weather': self.weather, 'brain': self.brain_name, 'light': self.light,
            'sensorSet': self.sensors, 'battery': round(e.battery_percent, 1),
            'goals': goals, 'totalGoals': e.goals_reached, 'steps': self.steps,
            'collisions': self.collisions, 'totalCollisions': self._collisions_total, 'respawns': self.respawns,
            'collPer1k': round(1000.0 * self.collisions / max(self.steps, 1), 1),
            'goalsPer1k': round(1000.0 * goals / max(self.steps, 1), 2),
            'energyPerGoal': round((e.energy - self._energy_base) / goals, 1) if goals else None,
            'groundSpeed': round(e.ground_dist / max(self.steps, 1), 2),
            'telemetry': {'frames': self.tm.frames_sent, 'bytes': self.tm.bytes_sent, 'crcErrors': self.gs.crc_errors, 'lost': self.gs.lost},
            'paused': self.paused, 'brains': list(self.brains), 'weathers': list(CONDITIONS), 'hasDetector': self.detector is not None,
            'auto': {'weather': self.auto_weather, 'light': self.auto_light},
        }

    def obstacles_message(self, e=None):
        e = e or self.env
        boxes = e.chunk_boxes_around(e.x, e.y, VIEW_CHUNKS)
        return {'type': 'obstacles', 'boxes': [[round(float(v), 1) for v in b] for b in boxes]}

    def targets_message(self, e=None):
        e = e or self.env
        t = aerial.targets_near(e, e.x, e.y, VIEW_CHUNKS * D.CHUNK)
        return {'type': 'targets',
                'vehicles': [[round(x, 1), round(y, 1), round(a, 2)] for x, y, a, _, _ in t['vehicles']],
                'people': [[round(x, 1), round(y, 1)] for x, y, _, _ in t['people']],
                'trees': [[round(x, 1), round(y, 1), round(r, 1)] for x, y, r in t['trees']],
                'rocks': [[round(x, 1), round(y, 1), round(r, 1)] for x, y, r, _, _ in t['rocks']]}

    def camera_message(self, e=None, seed_step=None):
        e = e or self.env
        alt = float(np.clip(e.z, 35.0, 100.0))
        rng = np.random.default_rng(self.steps if seed_step is None else seed_step)
        vib = min(2.0, 0.5 * float(np.linalg.norm(e.wind)))
        radius = aerial.FOOTPRINT_PER_ALT * alt
        img, labels = aerial.render(rng, e.boxes_near(e.x, e.y, radius), e.x, e.y, alt, self.light, vib,
                                    scene=aerial.targets_near(e, e.x, e.y, radius))
        x = np.zeros_like(img)
        chans = MODALITIES[self.sensors]
        x[chans] = img[chans]
        dets = []
        if self.detector is not None:
            with torch.no_grad():
                dets = decode(self.detector(torch.from_numpy(x[None])), 0.4)[0]
        rgb = (x[:3].transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8).copy()
        thermal = cv2.applyColorMap((x[3] * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        height = cv2.cvtColor((x[4] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        panel = []
        for im in (rgb, thermal, height):
            im = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            for c, s, x0, y0, x1, y1 in dets:
                cv2.rectangle(im, (int(x0 * 2), int(y0 * 2)), (int(x1 * 2), int(y1 * 2)), COLORS[c], 1)
            panel.append(im)
        ok, png = cv2.imencode('.png', np.concatenate(panel, axis=1))
        truth = [0, 0, 0]
        for c, *_ in labels:
            truth[c] += 1
        found = [0, 0, 0]
        for c, *_ in dets:
            found[c] += 1
        return {'type': 'camera', 'png': base64.b64encode(png.tobytes()).decode(), 'found': found, 'truth': truth, 'names': aerial.CLASSES,
                'alt': round(alt), 'light': self.light}


replay = None


def list_replays():
    if not LOG_DIR.is_dir():
        return []
    return [{'name': f.name, 'mb': round(f.stat().st_size / 1e6, 1)} for f in sorted(LOG_DIR.glob('*.jsonl'), reverse=True) if REPLAY_NAME_RE.match(f.name)]


def replay_load(name):
    """Parse a replay.jsonl written by edge_v3_4_sac.py (20 numbers per step, 9 actions)."""
    global replay
    if not REPLAY_NAME_RE.match(name or ''):
        return None
    path = LOG_DIR / name
    if not path.is_file():
        return None
    chunks, buf, meta = [], [], {}
    block = []

    def flush():
        if buf:
            chunks.append(np.array(buf, dtype=np.float32))
            buf.clear()

    def parse_block():
        if not block:
            return
        for line in block:
            try:
                fr = _loads(line)
            except ValueError:
                continue
            if len(fr) == 15:
                fr = fr + [0.0, 0.0, 0.0, 100.0, 0]
            buf.append(fr)
        flush()
        block.clear()

    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if line[0:1] == '{':
                try:
                    m = _loads(line)
                    if m.get('t') == 'meta':
                        meta = m
                except ValueError:
                    pass
            elif line.strip():
                block.append(line if line.endswith('\n') else line + '\n')
                if len(block) >= 100_000:
                    parse_block()
    parse_block()
    flush()
    if not chunks:
        return None
    frames = np.concatenate(chunks)
    renv = EdgeDroneEnvV2(seed=int(meta.get('world_seed', 1234)), weather='calm', wind_aware=False)
    renv.x, renv.y, renv.z, renv.yaw = (float(v) for v in frames[0][1:5])
    replay = dict(name=name, frames=frames, env=renv, pos=0.0, paused=False, weathers=meta.get('weathers', list(CONDITIONS)),
                  hits=np.cumsum(frames[:, 7], dtype=np.float64), last_chunk=None)
    return replay


def replay_state_message():
    r = replay
    idx = min(int(r['pos']), len(r['frames']) - 1)
    step, x, y, z, yaw, act, rew, hit, gx, gy, gz, goals, score, tsteps, vz, wx, wy, wz, batt, wi = r['frames'][idx].tolist()
    e = r['env']
    e.x, e.y, e.z, e.yaw = x, y, z, yaw
    e._update_sensors()
    ends = e.sensor_end
    n = idx + 1
    coll = float(r['hits'][idx])
    return {
        'type': 'state',
        'drone': {'x': x, 'y': y, 'z': z, 'yaw': yaw, 'vz': vz, 'colliding': bool(hit)},
        'sensors': [[round(float(ends[i, 0]), 1), round(float(ends[i, 1]), 1), round(float(ends[i, 2]), 1), round(e.signals[i], 3)] for i in range(D.N_SENSORS)],
        'goal': {'x': gx, 'y': gy, 'z': gz}, 'goalRadius': D.GOAL_RADIUS,
        'wind': [wx, wy, wz], 'weather': r['weathers'][int(wi)] if int(wi) < len(r['weathers']) else 'calm', 'brain': 'replay', 'light': sim.light,
        'sensorSet': sim.sensors, 'battery': round(batt, 1), 'goals': int(goals), 'steps': int(step), 'collisions': int(coll), 'respawns': 0,
        'collPer1k': round(1000.0 * coll / n, 1), 'goalsPer1k': round(1000.0 * goals / n, 2), 'energyPerGoal': None, 'groundSpeed': 0,
        'telemetry': {'frames': 0, 'bytes': 0, 'crcErrors': 0, 'lost': 0},
        'paused': bool(r['paused']), 'brains': list(sim.brains), 'weathers': list(CONDITIONS), 'hasDetector': sim.detector is not None,
        'replay': {'name': r['name'], 'pos': idx, 'n': len(r['frames']), 'paused': bool(r['paused'])},
    }


sim = Sim()
app = FastAPI()
clients = set()
lock = asyncio.Lock()


class NoCacheStatic(StaticFiles):
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers['Cache-Control'] = 'no-cache, must-revalidate'
        return resp


app.mount('/static', NoCacheStatic(directory=str(HERE / 'static')), name='static')


@app.get('/')
async def index():
    return FileResponse(str(HERE / 'static' / 'index.html'), headers={'Cache-Control': 'no-cache'})


async def loop():
    last_tick = time.perf_counter()
    debt = 0.0
    last_push = last_cam = 0.0
    last_chunk = None
    while True:
        now_t = time.perf_counter()
        debt += (now_t - last_tick) * TICK_HZ * sim.speed
        last_tick = now_t
        ticks = min(int(debt), max(4, int(TICK_HZ * max(sim.speed, 1.0) * 0.1)))
        debt = 0.0 if ticks >= max(4, int(TICK_HZ * max(sim.speed, 1.0) * 0.1)) else debt - ticks
        async with lock:
            if replay is not None:
                if not replay['paused']:
                    replay['pos'] += ticks
                    if replay['pos'] >= len(replay['frames']) - 1:
                        replay['pos'], replay['paused'] = float(len(replay['frames']) - 1), True
            elif not sim.paused:
                for _ in range(ticks):
                    sim.step()
            now = asyncio.get_event_loop().time()
            msgs = []
            live = replay['env'] if replay is not None else sim.env
            if now - last_push >= 1.0 / PUSH_HZ:
                last_push = now
                msgs.append(replay_state_message() if replay is not None else sim.state_message())
                ch = (int(math.floor(live.x / D.CHUNK)), int(math.floor(live.y / D.CHUNK)), live.seed, replay is not None)
                if ch != last_chunk:
                    last_chunk = ch
                    msgs.append(sim.obstacles_message(live))
                    msgs.append(sim.targets_message(live))
            if now - last_cam >= CAMERA_EVERY_S:
                last_cam = now
                msgs.append(sim.camera_message(live, int(replay['pos']) if replay is not None else None))
        for ws in list(clients):
            try:
                for m in msgs:
                    await ws.send_text(json.dumps(m))
            except Exception:
                clients.discard(ws)
        await asyncio.sleep(0.004)


@app.on_event('startup')
async def _startup():
    asyncio.create_task(loop())


@app.websocket('/ws')
async def ws_endpoint(ws: WebSocket):
    global replay
    await ws.accept()
    clients.add(ws)
    async with lock:
        await ws.send_text(json.dumps(sim.obstacles_message()))
        await ws.send_text(json.dumps(sim.targets_message()))
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if msg.get('type') != 'control':
                continue
            a, v = msg.get('action'), msg.get('value')
            async with lock:
                if a == 'pause':
                    sim.paused = True
                elif a == 'resume':
                    sim.paused = False
                elif a == 'speed':
                    sim.speed = float(v)
                elif a == 'weather' and v in CONDITIONS:
                    sim.change_weather(v)
                elif a == 'auto_weather' and isinstance(v, dict):
                    sim.auto_weather.update(on=bool(v.get('on')), every=max(5.0, min(600.0, float(v.get('every', 30)))),
                                            mode='random' if v.get('mode') == 'random' else 'cycle')
                    sim.weather_steps = 0
                elif a == 'auto_light' and isinstance(v, dict):
                    sim.auto_light.update(on=bool(v.get('on')), every=max(5.0, min(600.0, float(v.get('every', 20)))))
                    sim.light_steps = 0
                elif a == 'brain' and v in sim.brains:
                    sim.brain_name = v
                    sim.new_env()
                elif a == 'light' and v in aerial.LIGHT:
                    sim.set_light(v)
                elif a == 'sensors' and v in MODALITIES:
                    sim.sensors = v
                elif a == 'new_goal' and replay is None:
                    sim.env.new_goal()
                    sim.env._update_nav()
                elif a == 'list_replays':
                    await ws.send_text(json.dumps({'type': 'replays', 'files': list_replays()}))
                elif a == 'replay_start':
                    if replay_load(str(v or '')) is None:
                        await ws.send_text(json.dumps({'type': 'replays', 'files': list_replays(), 'error': 'could not load that replay'}))
                elif a == 'replay_pause' and replay is not None:
                    replay['paused'] = not replay['paused']
                    if not replay['paused'] and replay['pos'] >= len(replay['frames']) - 1:
                        replay['pos'] = 0.0
                elif a == 'replay_seek' and replay is not None:
                    replay['pos'] = float(min(max(float(v or 0), 0.0), 1.0) * (len(replay['frames']) - 1))
                elif a == 'replay_stop':
                    replay = None
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8021)
    ap.add_argument('--auto-weather', type=float, default=30.0, metavar='SECONDS')
    ap.add_argument('--auto-weather-mode', choices=['cycle', 'random'], default='cycle')
    ap.add_argument('--auto-light', type=float, default=20.0, metavar='SECONDS')
    args = ap.parse_args()
    sim.auto_weather.update(on=args.auto_weather > 0, every=max(5.0, args.auto_weather or 30.0), mode=args.auto_weather_mode)
    sim.auto_light.update(on=args.auto_light > 0, every=max(5.0, args.auto_light or 20.0))
    LOG_DIR.mkdir(exist_ok=True)
    print(f'replays go in {LOG_DIR}   brains found: {list(sim.brains)}   detector: {"yes" if sim.detector else "not trained yet"}   http://localhost:{args.port}')
    uvicorn.run(app, host='0.0.0.0', port=args.port, log_level='warning')
