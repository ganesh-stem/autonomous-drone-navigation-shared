"""
The cars of the city: parked cars in the car parks and traffic that really drives the road lattice. It is part of the SIMULATION, so the onboard camera, the detector and
the 3D viewer all see the same cars (the viewer only draws what it is sent). Pure Python + numpy, no ROS and no torch: it is used by server_edge_residual_adaptive.py
and by the ROS 2 sim_node (tools/build_core.py copies this file into the ROS package, do not edit the copy).

The road lattice is the one of the viewer (static/js/city.js): roads run along the borders of the 250 x 250 blocks (= the sim's chunks), 11 units either side of the border
line plus a 6 unit sidewalk.

  Traffic       right-hand traffic, one lane per direction, safe following gaps and queues, a traffic light at every junction (staggered phases), quarter-turn curves at junctions.
                How many cars are out depends on the light: dusk = rush hour (170), day = 85, night = about 15.
  parked_cars   the car park block (theme 6 of the viewer) holds up to 52 parked cars in its painted stalls; the cars are the same every time for the same (seed, block).
  aerial_*      the cars in the format of aerial.render(), so the camera image and the detector's "really there" count include them.
"""

import math
import random
from collections import defaultdict

import numpy as np

BLOCK = 250.0
ROAD_HALF = 11.0
WALK = 6.0
LANE = ROAD_HALF / 2
STOP = ROAD_HALF + 2.5
MAX_CARS = 170
RANGE = 900.0                                         # cars live within this distance of the drone and are recycled at the far edge
ACCEL, BRAKE, COMFORT = 9.0, 26.0, 12.0
TURN_P, TURN_SPEED = 0.27, 11.0
DENSITY = {'day': 0.5, 'dusk': 1.0, 'night': 0.09}    # share of MAX_CARS on the road in each light (the sim's light goes day, dusk, night, dusk: dusk twice = both rush hours)
TYPE_LEN = [9.4, 9.8, 10.2]                           # sedan, suv, van (the viewer's cars.js)
TYPE_WEIGHTS = [0.55, 0.30, 0.15]
COLORS = [0xd23c3c, 0x2f6fd6, 0xe8c23a, 0xf0f0f0, 0x38a86a, 0x2b2f38, 0xd9772b, 0x8a93a6, 0x7a3fb0, 0x1f9aa8]
CAR_PARK_R = 93.0                                     # a tower closer than this to the block centre turns the car park into a plain lawn (as in city.js)


# ───────────────────────── which block is a car park ─────────────────────────
def theme_for(bx, by):
    """The viewer's themeFor(): 0 = plain lawn, 1..7 = the scenery kinds (6 = car park). Same arithmetic as city.js (32-bit hash, 1 + (2bx + 3by) mod 7)."""
    h = ((bx * 73856093) & 0xFFFFFFFF) ^ ((by * 19349663) & 0xFFFFFFFF)
    if (h % 1000) / 1000.0 < 0.2:
        return 0
    return 1 + (2 * bx + 3 * by) % 7


def _tower_in_range(boxes, bx, by, radius):
    cx, cy = (bx + 0.5) * BLOCK, (by + 0.5) * BLOCK
    for b in boxes:
        if b[2] <= 0.5:
            dx, dy = max(b[0] - cx, 0.0, cx - b[3]), max(b[1] - cy, 0.0, cy - b[4])
            if dx * dx + dy * dy < radius * radius:
                return True
    return False


_parked_cache = {}


def parked_cars(seed, bx, by, boxes):
    """Parked cars of block (bx, by) as [(x, y, heading, type, color), ...]. boxes = the block's own boxes (env._chunk(bx, by)), for the tower fallback."""
    key = (seed, bx, by)
    hit = _parked_cache.get(key)
    if hit is not None:
        return hit
    cars = []
    if theme_for(bx, by) == 6 and not _tower_in_range(boxes, bx, by, CAR_PARK_R):
        rng = np.random.default_rng([seed & 0xFFFFFFFF, bx + 9_000_011, by + 9_000_041])
        cx, cy = (bx + 0.5) * BLOCK, (by + 0.5) * BLOCK
        for row in (-1, 1):
            for k in range(26):
                draw = rng.random(3)                                                    # always three draws per stall, so the cars do not shift when one is added
                if draw[0] > 0.55:
                    continue
                t = 0 if draw[1] < TYPE_WEIGHTS[0] else 1 if draw[1] < TYPE_WEIGHTS[0] + TYPE_WEIGHTS[1] else 2
                cars.append((cx - 81.25 + k * 6.5, cy + row * 17.0, math.pi / 2 * row, t, COLORS[int(draw[2] * len(COLORS)) % len(COLORS)]))    # nose first into the stall
    if len(_parked_cache) > 4000:
        _parked_cache.clear()
    _parked_cache[key] = cars
    return cars


def parked_near(env, x, y, chunks):
    """Parked cars in the blocks within `chunks` blocks of (x, y): world coordinates, as a list of (x, y, heading, type, color)."""
    out = []
    bx0, by0 = int(math.floor(x / BLOCK)), int(math.floor(y / BLOCK))
    for bx in range(bx0 - chunks, bx0 + chunks + 1):
        for by in range(by0 - chunks, by0 + chunks + 1):
            out.extend(parked_cars(env.seed, bx, by, env._chunk(bx, by)))
    return out


# ───────────────────────── traffic ─────────────────────────
def signal(axis, t, kx, ky):
    """Traffic light of the junction (kx, ky) for roads along `axis` (0 = along x, 1 = along y): 0 green, 1 yellow, 2 red. A 24 s cycle, every junction with its own phase."""
    p = (t + (kx * 7 + ky * 13) % 24) % 24
    if axis == 0:
        return 0 if p < 10 else 1 if p < 12 else 2
    return 2 if p < 13 else 0 if p < 22 else 1


class Car:
    __slots__ = ('active', 'retire', 'axis', 'dir', 'line', 's', 'v', 'vmax', 'type', 'color', 'heading', 'len', 'braking', 'turn')

    def __init__(self):
        self.active = self.retire = self.braking = False
        self.axis, self.dir, self.line = 0, 1, 0
        self.s = self.v = self.heading = 0.0
        self.vmax, self.type, self.color, self.len, self.turn = 22.0, 0, COLORS[0], TYPE_LEN[0], None

    def lateral(self):
        """World coordinate of the centre of this car's lane (right-hand traffic)."""
        if self.axis == 0:
            return self.line * BLOCK + (-LANE if self.dir > 0 else LANE)
        return self.line * BLOCK + (LANE if self.dir > 0 else -LANE)

    def xy(self):
        if self.turn:
            return self.turn['pos']
        return (self.s, self.lateral()) if self.axis == 0 else (self.lateral(), self.s)


def _heading_of(axis, d):
    return (0.0 if d > 0 else math.pi) if axis == 0 else (math.pi / 2 if d > 0 else -math.pi / 2)


def _angle_diff(a, b):
    return (b - a + 3 * math.pi) % (2 * math.pi) - math.pi


def _brake_limit(gap, keep):
    return math.sqrt(max(0.0, 2 * COMFORT * (gap - keep)))


class Traffic:

    def __init__(self, seed=1, max_cars=MAX_CARS):
        self.rng = random.Random(int(seed) & 0xFFFFFFFF)
        self.cars = [Car() for _ in range(max_cars)]
        self.max = max_cars
        self.density = 0.5
        self.clock = 0.0
        self.started = False
        self.lanes = {}

    # ---- placing cars
    def _place(self, c, fill, dx, dy):
        r = self.rng
        free = False
        for _ in range(10):
            c.axis, c.dir, c.turn = (0 if r.random() < 0.5 else 1), (1 if r.random() < 0.5 else -1), None
            across, along = (dy, dx) if c.axis == 0 else (dx, dy)
            c.line = round(across / BLOCK) + r.randrange(7) - 3
            c.s = along + (r.random() * 2 - 1) * RANGE * 0.9 if fill else along - c.dir * RANGE * 0.88 + (r.random() - 0.5) * 60
            if not any(o is not c and o.active and o.axis == c.axis and o.line == c.line and o.dir == c.dir and abs(o.s - c.s) < 40 for o in self.cars):
                free = True
                break
        if not free:
            c.active = False
            return False
        u, i = r.random(), 0
        while i < 2 and u > TYPE_WEIGHTS[i]:
            u -= TYPE_WEIGHTS[i]
            i += 1
        c.type, c.len, c.color = i, TYPE_LEN[i], COLORS[r.randrange(len(COLORS))]
        c.vmax = 19 + r.random() * 7
        c.v, c.braking, c.retire, c.active = c.vmax * 0.7, False, False, True
        c.heading = _heading_of(c.axis, c.dir)
        return True

    # ---- the stop line ahead
    @staticmethod
    def _next_stop(c, t):
        front = c.s + c.dir * c.len / 2
        k = math.ceil((front - 0.5 + STOP) / BLOCK) if c.dir > 0 else math.floor((front + 0.5 - STOP) / BLOCK)
        sp = k * BLOCK - c.dir * STOP
        sig = signal(0, t, k, c.line) if c.axis == 0 else signal(1, t, c.line, k)
        return (sp - front) * c.dir, sig

    def _start_turn(self, c, k):
        r = self.rng
        left = r.random() < 0.5
        hx, hy = (c.dir, 0) if c.axis == 0 else (0, c.dir)
        nx, ny = (-hy, hx) if left else (hy, -hx)
        n_axis = 1 - c.axis
        n_dir = nx if n_axis == 0 else ny
        cl = k                                                                          # the road we turn onto is the cross road k
        lat_old = c.lateral()
        lat_new = cl * BLOCK + ((-LANE if n_dir > 0 else LANE) if n_axis == 0 else (LANE if n_dir > 0 else -LANE))
        cxi, cyi = (k * BLOCK, c.line * BLOCK) if c.axis == 0 else (c.line * BLOCK, k * BLOCK)       # centre of the junction
        exit_s = (cxi if n_axis == 0 else cyi) + n_dir * ROAD_HALF
        for o in self.lanes.get((n_axis, cl, n_dir), ()):                                # the lane we would turn into must be free near its exit
            if o is not c and -45 < (o.s - exit_s) * n_dir < 16:
                return
        p0 = (k * BLOCK - c.dir * ROAD_HALF, lat_old) if c.axis == 0 else (lat_old, k * BLOCK - c.dir * ROAD_HALF)
        p2 = (cxi + n_dir * ROAD_HALF, lat_new) if n_axis == 0 else (lat_new, cyi + n_dir * ROAD_HALF)
        p1 = (lat_new, lat_old) if c.axis == 0 else (lat_old, lat_new)                  # where the old lane line meets the new one: the corner of the curve
        length = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) + math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        c.turn = {'p0': p0, 'p1': p1, 'p2': p2, 't': 0.0, 'len': length, 'axis': n_axis, 'dir': n_dir, 'line': cl, 'pos': p0}

    def _step_turn(self, c, dt):
        tr = c.turn
        c.v += max(-BRAKE * dt, min(ACCEL * dt, TURN_SPEED - c.v))
        tr['t'] = min(1.0, tr['t'] + c.v * dt / tr['len'])
        t = tr['t']
        u = 1 - t
        p0, p1, p2 = tr['p0'], tr['p1'], tr['p2']
        tr['pos'] = (u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0], u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1])
        tx, ty = 2 * u * (p1[0] - p0[0]) + 2 * t * (p2[0] - p1[0]), 2 * u * (p1[1] - p0[1]) + 2 * t * (p2[1] - p1[1])
        if tx or ty:
            c.heading = math.atan2(ty, tx)
        if t >= 1.0:
            c.axis, c.dir, c.line = tr['axis'], tr['dir'], tr['line']
            c.s = p2[0] if c.axis == 0 else p2[1]
            c.turn = None
            c.heading = _heading_of(c.axis, c.dir)

    # ---- one step of the traffic
    def step(self, dt, dx, dy, light):
        """dt in seconds; (dx, dy) = the drone's position (the traffic lives around it); light = 'day' | 'dusk' | 'night'."""
        self.clock += dt
        self.density += (DENSITY.get(light, DENSITY['day']) - self.density) * min(1.0, dt * 0.35)
        target = round(self.max * self.density)
        cars = self.cars
        live = sum(1 for c in cars if c.active and not c.retire)
        if not self.started:
            self.started = True
            for c in cars:
                if live < target and self._place(c, True, dx, dy):
                    live += 1
        elif live < target:
            spawned = 0
            for c in cars:
                if not c.active:
                    if self._place(c, False, dx, dy):
                        spawned += 1
                    if spawned >= 4 or live + spawned >= target:
                        break
        elif live > target:
            far, fd = None, -1.0
            for c in cars:
                if c.active and not c.retire:
                    x, y = c.xy()
                    d = math.hypot(x - dx, y - dy)
                    if d > fd:
                        fd, far = d, c
            if far:
                far.retire = True

        lanes = defaultdict(list)                                                        # cars of one lane, in driving order, so the gap to the car ahead is the next entry
        for c in cars:
            if c.active and not c.turn:
                lanes[(c.axis, c.line, c.dir)].append(c)
        for lane in lanes.values():
            lane.sort(key=lambda o: o.s * o.dir)
        self.lanes = lanes
        ahead = {}
        for lane in lanes.values():
            for i in range(len(lane) - 1):
                ahead[id(lane[i])] = lane[i + 1]

        t = self.clock
        for c in cars:
            if not c.active:
                continue
            if c.turn:
                self._step_turn(c, dt)
            else:
                gap = float('inf')
                o = ahead.get(id(c))
                if o is not None:
                    gap = (o.s - c.s) * c.dir - (o.len + c.len) / 2
                sd, sig = self._next_stop(c, t)
                vt = min(c.vmax, _brake_limit(gap, 2.5))
                if sig == 2 or (sig == 1 and sd > c.v * c.v / 32 + 1):                   # yellow: stop only if it can be done comfortably
                    vt = min(vt, _brake_limit(sd, 0.4))
                dv = vt - c.v
                c.v += max(-BRAKE * dt, min(ACCEL * dt, dv))
                c.braking = dv < -1.5
                before = c.s
                c.s += c.dir * c.v * dt
                kt = math.ceil((before + ROAD_HALF) / BLOCK) if c.dir > 0 else math.floor((before - ROAD_HALF) / BLOCK)
                entry = kt * BLOCK - c.dir * ROAD_HALF
                if (before - entry) * c.dir < 0 <= (c.s - entry) * c.dir and not c.retire and self.rng.random() < TURN_P:   # the car's centre enters a junction: some turn
                    self._start_turn(c, kt)
                c.heading += _angle_diff(c.heading, _heading_of(c.axis, c.dir)) * min(1.0, dt * 8)
            x, y = c.xy()
            dist = math.hypot(x - dx, y - dy)
            if dist > RANGE or (c.retire and dist > 520):
                if c.retire:
                    c.active = False
                else:
                    self._place(c, False, dx, dy)

    # ---- what the others are told
    def snapshot(self):
        """Active cars as [x, y, heading, type, colour index, braking, speed] (rounded, ready for a JSON message)."""
        out = []
        for c in self.cars:
            if c.active:
                x, y = c.xy()
                out.append([round(x, 1), round(y, 1), round(c.heading, 2), c.type, COLORS.index(c.color), 1 if c.braking else 0, round(c.v, 1)])
        return out

    def aerial_vehicles(self, x, y, radius):
        """The moving cars within `radius` of (x, y) in the format of aerial.render(): (x, y, angle, heat, (r, g, b))."""
        out = []
        for c in self.cars:
            if c.active:
                cx, cy = c.xy()
                if abs(cx - x) <= radius and abs(cy - y) <= radius:
                    out.append((cx, cy, c.heading, 0.9, _rgb(c.color)))
        return out


def _rgb(hex_color):
    return (((hex_color >> 16) & 255) / 255.0, ((hex_color >> 8) & 255) / 255.0, (hex_color & 255) / 255.0)


def aerial_from_rows(rows, x, y, radius):
    """Like Traffic.aerial_vehicles(), for cars received as snapshot rows [x, y, heading, type, colour index, braking, speed]."""
    return [(r[0], r[1], r[2], 0.9, _rgb(COLORS[int(r[4]) % len(COLORS)])) for r in rows if abs(r[0] - x) <= radius and abs(r[1] - y) <= radius]


def parked_aerial(parked, x, y, radius):
    """Parked cars within `radius` of (x, y) in aerial.render()'s vehicle format. They are a little cooler than a running car in the thermal view."""
    return [(px, py, h, 0.8, _rgb(col)) for px, py, h, _t, col in parked if abs(px - x) <= radius and abs(py - y) <= radius]
