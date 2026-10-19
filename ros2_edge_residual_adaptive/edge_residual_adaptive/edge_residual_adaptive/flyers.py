"""
Birds and patrol drones as real, moving obstacles of the simulation.

The floating blocks of the world (the boxes the viewer marks with colour-changing gates) stay exactly where they are. In addition, every such spot sends out its birds
and patrol drones, and each of them is a small obstacle of its own that ROAMS: it follows a long, smooth wander path (hundreds of units wide, nothing like a circle around
the gate) and is a small box of about its own size (a bird 9 x 9 x 4, a patrol drone 10 x 10 x 3 units). The rays, the collisions, the safety filter and the aerial camera all
see these boxes, in addition to the static ones.

The position of a flyer is a pure function of its spot (the base corner x0, y0, z0 of the floating block), its number there and the step number t:

    offset(t) = rotate( ax*sin(w t + p1),  ay*sin(r w t + p2) )  in x, y      and      az*sin(0.6 w t + p1)  in z

Every number (how many flyers, species, path size, speed, phases, tilt) comes from a hash of the spot, so the simulator, the onboard camera and the 3D viewer all compute the same
positions and only have to share the step number. Peak horizontal speed is 1.0 to 2.2 units per step for a bird and 0.8 to 1.6 for a patrol drone (the drone flies 5.0).

install(cls) wraps boxes_near() and step() of an environment class: for an env whose  flyers_on  is True, boxes_near() also returns the flyers that are near, and step() moves the
clock and shoves the drone aside when a flyer has moved onto it (that counts as a collision). Replays and older variants keep their static world (flyers_on False).

This file is shared: web3d_drones_edge_residual_adaptive/server_edge_residual_adaptive.py imports it, ros2_edge_residual_adaptive/tools/build_core.py copies it into the ROS 2 package
(like city_world.py), and the standalone Kaggle trainer has the same code pasted in (a self-test checks that the copies are identical).
"""
import math

import numpy as np

AMP_XY = (150.0, 420.0)        # how far a flyer wanders from its spot, along each of its two (tilted) path axes, in world units
AMP_Z = (12.0, 40.0)           # how far it wanders up and down
SPEED_BIRD = (1.0, 2.2)        # PEAK speed of a bird's path, units per simulation step (the drone flies 5.0 per step)
SPEED_DRONE = (0.8, 1.6)       # PEAK speed of a patrol drone's path
BIRD_BOX = (9.0, 9.0, 4.0)     # collision box of a bird (width, depth, height; its wings span about 13 units, its body 6); the drone's own radius of 3 is added by the environment
DRONE_BOX = (10.0, 10.0, 3.0)  # collision box of a patrol drone
Z_RANGE = (20.0, 195.0)        # the flight band the flyers stay in
MARGIN = math.hypot(AMP_XY[1], AMP_XY[1]) + 15.0    # the window of floating blocks looks this far out: a flyer may be this far from its spot (diagonally) and still be near the drone
DRONE_RADIUS = 3.0             # the same number as drone_env.DRONE_RADIUS
CHUNK = 250.0                  # the size of a chunk of the world (the same number as drone_env.CHUNK)
FLOOR = 0.5                    # a box with z0 above this is a floating block (towers start at z0 = 0)


def _hash(x0, y0, z0, k):
    """A stable pseudo-random number in [0, 1) from a spot's base corner; k picks one of several independent numbers."""
    t = np.sin(x0 * 12.9898 + y0 * 78.233 + z0 * 37.719 + k * 4.1414) * 43758.5453
    return t - np.floor(t)


def _members(base):
    """The flyers of the floating blocks `base` (n, 6): arrays with one entry per flyer."""
    x0, y0, z0 = base[:, 0], base[:, 1], base[:, 2]
    h0, h1 = _hash(x0, y0, z0, 20), _hash(x0, y0, z0, 21)
    n_birds = np.where(h0 < 0.35, 2 + np.floor(h0 * 977) % 2, 1).astype(int)       # most spots send one bird, about a third a small flock of 2 or 3
    n_drones = np.where(h1 < 0.28, 2, 1).astype(int)                                 # one patrol drone, a quarter of the spots two
    parent, member, kind = [], [], []
    for i in range(len(base)):
        for m in range(n_birds[i]):
            parent.append(i); member.append(m); kind.append(int(_hash(x0[i], y0[i], z0[i], 30 + m) * 3) % 3)      # bird species 0, 1 or 2
        for m in range(n_drones[i]):
            parent.append(i); member.append(10 + m); kind.append(3)                                                  # 3 = patrol drone
    return np.array(parent, dtype=int), np.array(member, dtype=int), np.array(kind, dtype=int)


def _consts(base):
    """Everything about the flyers of the floating blocks `base` that does not change with time (computed once per window, then kept)."""
    parent, member, kind = _members(base)
    b = base[parent]
    x0, y0, z0 = b[:, 0], b[:, 1], b[:, 2]
    h = [_hash(x0, y0, z0, 40 + member * 8 + j) for j in range(8)]
    drone = kind == 3
    ax = AMP_XY[0] + (AMP_XY[1] - AMP_XY[0]) * h[0]
    ay = AMP_XY[0] + (AMP_XY[1] - AMP_XY[0]) * h[1]
    az = AMP_Z[0] + (AMP_Z[1] - AMP_Z[0]) * h[2]
    v = np.where(drone, SPEED_DRONE[0] + (SPEED_DRONE[1] - SPEED_DRONE[0]) * h[3], SPEED_BIRD[0] + (SPEED_BIRD[1] - SPEED_BIRD[0]) * h[3])
    r = 0.55 + 0.6 * h[7]                                                             # the two axes have different rates, so the path is a long loop, not an ellipse
    w = v / np.hypot(ax, r * ay)                                                      # rad per step: the peak speed of the path is v
    rot = 2 * math.pi * h[6]
    return dict(parent=parent, member=member, kind=kind, ax=ax, ay=ay, az=az, r=r, w=w, p1=2 * math.pi * h[4], p2=2 * math.pi * h[5], c=np.cos(rot), s=np.sin(rot),
                cx=(b[:, 0] + b[:, 3]) / 2, cy=(b[:, 1] + b[:, 4]) / 2, cz=(b[:, 2] + b[:, 5]) / 2, base=b[:, :3],
                size=np.where(drone[:, None], np.array(DRONE_BOX), np.array(BIRD_BOX)))


def _at(k, t):
    """Positions (n, 3) of the flyers' centres and their velocities (n, 3, per step) at step t."""
    w = k['w']
    a1, a2, a3 = w * t + k['p1'], k['r'] * w * t + k['p2'], 0.6 * w * t + k['p1']
    ex, ey = k['ax'] * np.sin(a1), k['ay'] * np.sin(a2)
    evx, evy = k['ax'] * w * np.cos(a1), k['ay'] * k['r'] * w * np.cos(a2)
    c, s = k['c'], k['s']
    z = k['cz'] + k['az'] * np.sin(a3)
    vz = k['az'] * 0.6 * w * np.cos(a3)
    inside = (z > Z_RANGE[0]) & (z < Z_RANGE[1])
    z, vz = np.clip(z, Z_RANGE[0], Z_RANGE[1]), np.where(inside, vz, 0.0)
    pos = np.stack([k['cx'] + c * ex - s * ey, k['cy'] + s * ex + c * ey, z], axis=1)
    vel = np.stack([c * evx - s * evy, s * evx + c * evy, vz], axis=1)
    return pos, vel


def flyer_boxes(env, window, x, y, reach, t):
    """The boxes of the roaming flyers that are within `reach` of (x, y) at step t, as rows (x0, y0, z0, x1, y1, z1). `window(x, y, r)` returns the static boxes of the chunks
    around (x, y) within r. The flyers of a window (their constants) are worked out once and kept on the env; their positions once per step, however often boxes_near() is
    called within the step."""
    r = reach + MARGIN
    key = (int(math.floor((x - r) / CHUNK)), int(math.floor((x + r) / CHUNK)), int(math.floor((y - r) / CHUNK)), int(math.floor((y + r) / CHUNK)))
    cache = env.__dict__.setdefault('_flyer_cache', {})
    ent = cache.get(key)
    if ent is None:
        wide = window(x, y, r)
        fl = wide[wide[:, 2] > FLOOR] if len(wide) else wide
        if len(cache) > 64:
            cache.clear()
        ent = cache[key] = [_consts(fl) if len(fl) else None, None, None]
    k = ent[0]
    if k is None:
        return np.zeros((0, 6))
    if ent[1] != t:
        ent[1], ent[2] = t, _at(k, t)[0]
    pos, size = ent[2], k['size']
    near = (np.abs(pos[:, 0] - x) < reach + size[:, 0]) & (np.abs(pos[:, 1] - y) < reach + size[:, 1])
    pos, size = pos[near], size[near]
    return np.concatenate([pos - size / 2, pos + size / 2], axis=1)


def flyers_near(env, x, y, chunks, t):
    """Rows for the viewer: [x0, y0, z0 (the spot), member, kind (0-2 a bird species, 3 a patrol drone), hash, w, d, h, cx, cy, cz, vx, vy, vz] of every flyer within
    `chunks` chunks of (x, y) at step t; the velocity is per step."""
    base = env.chunk_boxes_around(x, y, chunks + int(math.ceil(MARGIN / 250.0)))
    base = base[base[:, 2] > FLOOR] if len(base) else base
    if len(base) == 0:
        return []
    k = _consts(base)
    pos, vel = _at(k, t)
    size, b = k['size'], k['base']
    reach = chunks * 250.0
    near = (np.abs(pos[:, 0] - x) < reach) & (np.abs(pos[:, 1] - y) < reach)
    mh = _hash(b[:, 0], b[:, 1], b[:, 2], 90 + k['member'])
    rows = np.stack([b[:, 0], b[:, 1], b[:, 2], k['member'], k['kind'], mh, size[:, 0], size[:, 1], size[:, 2], pos[:, 0], pos[:, 1], pos[:, 2], vel[:, 0], vel[:, 1], vel[:, 2]], axis=1)[near]
    return [[float(v) for v in r] for r in rows]


def push_out(env, radius):
    """If a moving flyer now overlaps the drone (a sphere of `radius` at env.x, env.y, env.z), shove the drone out of it by the shortest way and return True. Without this a
    flyer that moves onto the drone would trap it: the environment rejects every move that touches a box, so the drone would sit inside the box until the stuck timer respawns it."""
    boxes = env.boxes_near(env.x, env.y, radius + 2)
    if len(boxes) == 0:
        return False
    p = np.array([env.x, env.y, env.z])
    hit = False
    for _ in range(3):                                                         # a few passes, in case two boxes overlap the drone at once
        moved_now = False
        for b in boxes:
            lo, hi = b[:3], b[3:]
            q = np.minimum(np.maximum(p, lo), hi)                             # the point of the box nearest to the drone
            d = p - q
            dist = float(np.linalg.norm(d))
            if dist >= radius:
                continue
            if dist > 1e-9:                                                    # the centre is outside the box: push along the line from the box to the drone
                p = q + d / dist * (radius + 0.5)
            else:                                                              # the centre is inside the box: leave through the nearest face
                gaps = np.concatenate([p - lo, hi - p])
                k = int(np.argmin(gaps))
                axis, sign = k % 3, (-1.0 if k < 3 else 1.0)
                p[axis] = (lo[axis] - radius - 0.5) if sign < 0 else (hi[axis] + radius + 0.5)
            moved_now = hit = True
        if not moved_now:
            break
    if hit:
        env.x, env.y, env.z = float(p[0]), float(p[1]), float(max(p[2], radius))
    return hit


def install(cls):
    """Make environment class `cls` add the roaming birds and patrol drones to its boxes when an instance has flyers_on = True. Safe to call on a class and on its subclasses:
    boxes_near and step are each wrapped only once along the inheritance chain, and a step that reaches another wrapped step does not count time twice."""
    cls.flyers_on, cls.flyer_t = False, 0
    if not getattr(cls.boxes_near, '_flyers', False):
        orig_near = cls.boxes_near

        def boxes_near(self, x, y, radius):
            if not self.flyers_on:
                return orig_near(self, x, y, radius)
            mine = orig_near(self, x, y, radius)                                   # the static boxes, as before
            fl = flyer_boxes(self, lambda cx, cy, r: orig_near(self, cx, cy, r), x, y, radius, self.flyer_t)
            return np.concatenate([mine, fl]) if len(fl) else mine                 # plus the roaming flyers that are near

        boxes_near._flyers = True
        cls.boxes_near = boxes_near
    if not getattr(cls.step, '_flyers', False):
        orig_step = cls.step

        def step(self, action):
            if not self.flyers_on or getattr(self, '_in_flyer_step', False):
                return orig_step(self, action)
            self._in_flyer_step = True
            try:
                self.flyer_t += 1                                                  # time advances first, so the whole step (move, collision, rays) sees one moment
                hit = push_out(self, DRONE_RADIUS)                                 # a flyer that moved onto the drone shoves it aside
                result = orig_step(self, action)
                if hit:
                    self.is_colliding, self.collision_kind = True, "obstacle"      # and it counts as a collision
                return result
            finally:
                self._in_flyer_step = False

        step._flyers = True
        cls.step = step
