"""
Replay files for the ROS viewer: lists the *.jsonl files in the log folder and plays one back in the browser (the same files the training project
and recorder_node write: a meta line, then one line of 20 numbers per step, or 15 numbers in the old format).

A replay belongs to one browser connection. It does not touch the ROS topics, so the live flight keeps running underneath.
"""

import base64
import json
import math
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np

from . import aerial
from . import city_world as CW
from .core import CHUNK, CONDITIONS, DroneEnv, GOAL_RADIUS, N_SENSORS

NAME_RE = re.compile(r'^[\w\-. ]+\.jsonl$')
VIEW_CHUNKS = 4
TICK_HZ = 60.0
try:
    from orjson import loads as _loads
except ImportError:
    _loads = json.loads


def list_replays(log_dir):
    d = Path(log_dir)
    if not d.is_dir():
        return []
    return [{'name': f.name, 'mb': round(f.stat().st_size / 1e6, 1)} for f in sorted(d.glob('*.jsonl'), reverse=True) if NAME_RE.match(f.name)]


class Replay:

    def __init__(self, log_dir, name, default_seed):
        path = Path(log_dir) / name
        if not NAME_RE.match(name or '') or not path.is_file():
            raise ValueError('no such replay file')
        st = path.stat()
        cache = Path(tempfile.gettempdir()) / 'edge_residual_replay_cache' / f'{name}_{st.st_size}_{int(st.st_mtime)}'      # the parsed numbers, so a second play is instant
        try:
            frames = np.load(str(cache) + '.npy')
            meta = json.loads(Path(str(cache) + '.json').read_text())
        except (OSError, ValueError):
            frames, meta = self._parse(path)
            try:
                cache.parent.mkdir(exist_ok=True)
                np.save(str(cache) + '.npy', frames)
                Path(str(cache) + '.json').write_text(json.dumps(meta))
            except OSError:
                pass
        self.name = name
        self.frames = frames
        self.weathers = meta.get('weathers', list(CONDITIONS))
        self.hits = np.cumsum(self.frames[:, 7], dtype=np.float64)
        self.env = DroneEnv(seed=int(meta.get('world_seed', default_seed)))
        self.env.x, self.env.y, self.env.z, self.env.yaw = (float(v) for v in self.frames[0][1:5])
        self.pos, self.paused, self.speed = 0.0, False, 1.0
        self.last_chunk, self.last_cam = None, -1.0

    @staticmethod
    def _parse(path):
        rows, meta = [], {}
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
                    try:
                        fr = _loads(line)
                    except ValueError:
                        continue                                    # a torn last line after a crash is fine
                    if len(fr) == 15:
                        fr = list(fr) + [0.0, 0.0, 0.0, 100.0, 0]
                    if len(fr) == 20:
                        rows.append(fr)
        if not rows:
            raise ValueError('the file has no steps')
        return np.array(rows, dtype=np.float32), {k: meta[k] for k in ('weathers', 'world_seed') if k in meta}

    @property
    def index(self):
        return min(int(self.pos), len(self.frames) - 1)

    def advance(self, seconds):
        if self.paused:
            return
        self.pos += seconds * TICK_HZ * self.speed
        if self.pos >= len(self.frames) - 1:
            self.pos, self.paused = float(len(self.frames) - 1), True

    def toggle_pause(self):
        self.paused = not self.paused
        if not self.paused and self.pos >= len(self.frames) - 1:
            self.pos = 0.0

    def seek(self, fraction):
        self.pos = float(min(max(float(fraction), 0.0), 1.0) * (len(self.frames) - 1))

    def state_message(self, light, sensors, brains):
        idx = self.index
        step, x, y, z, yaw, act, rew, hit, gx, gy, gz, goals, score, tsteps, vz, wx, wy, wz, batt, wi = self.frames[idx].tolist()
        e = self.env
        e.x, e.y, e.z, e.yaw = x, y, z, yaw
        e._update_sensors()
        ends = e.sensor_end
        n = idx + 1
        coll = float(self.hits[idx])
        return {
            'type': 'state',
            'drone': {'x': x, 'y': y, 'z': z, 'yaw': yaw, 'vz': vz, 'colliding': bool(hit)},
            'sensors': [[round(float(ends[i, 0]), 1), round(float(ends[i, 1]), 1), round(float(ends[i, 2]), 1), round(float(e.signals[i]), 3)] for i in range(N_SENSORS)],
            'goal': {'x': gx, 'y': gy, 'z': gz}, 'goalRadius': GOAL_RADIUS,
            'wind': [wx, wy, wz], 'weather': self.weathers[int(wi)] if int(wi) < len(self.weathers) else 'calm', 'brain': 'replay', 'light': light,
            'sensorSet': sensors, 'battery': round(batt, 1), 'goals': int(goals), 'totalGoals': int(goals), 'steps': int(step), 'collisions': int(coll), 'totalCollisions': int(coll), 'respawns': 0,
            'collPer1k': round(1000.0 * coll / n, 1), 'goalsPer1k': round(1000.0 * goals / n, 2), 'energyPerGoal': None, 'groundSpeed': 0,
            'telemetry': {'frames': 0, 'bytes': 0, 'crcErrors': 0, 'lost': 0},
            'paused': self.paused, 'brains': brains, 'weathers': list(CONDITIONS), 'hasDetector': False,
            'replay': {'name': self.name, 'pos': idx, 'n': len(self.frames), 'paused': self.paused},
        }

    def world_messages(self):
        e = self.env
        boxes = e.chunk_boxes_around(e.x, e.y, VIEW_CHUNKS)
        t = aerial.targets_near(e, e.x, e.y, VIEW_CHUNKS * CHUNK)
        return [{'type': 'obstacles', 'boxes': [[round(float(v), 1) for v in b] for b in boxes]},
                {'type': 'targets',
                 'vehicles': [],
                 'parked': [[round(px, 1), round(py, 1), round(h, 2), ty, CW.COLORS.index(col)] for px, py, h, ty, col in CW.parked_near(e, e.x, e.y, VIEW_CHUNKS)],
                 'people': [[round(a, 1), round(b, 1)] for a, b, _, _ in t['people']],
                 'trees': [[round(a, 1), round(b, 1), round(c, 1)] for a, b, c in t['trees']],
                 'rocks': [[round(a, 1), round(b, 1), round(c, 1)] for a, b, c, _, _ in t['rocks']]}]

    def chunk_changed(self):
        e = self.env
        ch = (int(math.floor(e.x / CHUNK)), int(math.floor(e.y / CHUNK)))
        if ch != self.last_chunk:
            self.last_chunk = ch
            return True
        return False

    def camera_message(self, light):
        """The camera picture for this step (the detector is a live ROS node, so a replay shows the pictures without detection boxes)."""
        e = self.env
        alt = float(np.clip(e.z, 35.0, 100.0))
        rng = np.random.default_rng(self.index)
        vib = min(2.0, 0.5 * float(np.linalg.norm([self.frames[self.index][15], self.frames[self.index][16], self.frames[self.index][17]])))
        radius = aerial.FOOTPRINT_PER_ALT * alt
        img, labels = aerial.render(rng, e.boxes_near(e.x, e.y, radius), e.x, e.y, alt, light, vib, scene=aerial.targets_near(e, e.x, e.y, radius))
        rgb = (img[:3].transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8).copy()
        thermal = cv2.applyColorMap((img[3] * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        height = cv2.cvtColor((img[4] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        panel = np.concatenate([cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST) for im in (rgb, thermal, height)], axis=1)
        ok, png = cv2.imencode('.png', panel)
        truth = [0, 0, 0]
        for c, *_ in labels:
            truth[c] += 1
        return {'type': 'camera', 'png': base64.b64encode(png.tobytes()).decode(), 'found': [0, 0, 0], 'truth': truth, 'names': aerial.CLASSES,
                'alt': round(alt), 'light': light}
