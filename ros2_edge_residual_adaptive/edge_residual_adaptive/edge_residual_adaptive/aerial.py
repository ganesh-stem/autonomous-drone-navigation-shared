"""
Synthetic aerial images with labels, from the same infinite sky the drone flies in.

A nadir (straight down) camera with 5 channels, 96 x 96 pixels:
    0-2  RGB       colour camera. Gets dark at dusk and night, noisy, hazy, blurred by wind vibration.
    3    thermal   works in the dark. Warm things (engines, people, but also sun-heated rocks) are bright.
    4    height    a LiDAR-like height map: the top of each building or block (0 = ground).
The image footprint grows with altitude (1.2 x altitude across), so the same object is smaller from higher up.

Classes:  0 vehicle   1 person   2 building (the roof of a tower or floating block from the environment)
Decoys that must NOT be detected: trees (dark green, cold) and hot rocks (grey, warm).
A hot rock fools the thermal camera and a tree fools nobody at night, so fusing the channels matters.

    img, labels = render(rng, boxes, cx, cy, alt, light='day', vibration=0.0)
    labels: list of (class, x0, y0, x1, y1) in pixels
"""

import math
from pathlib import Path

import cv2
import numpy as np

SIZE = 96
CLASSES = ['vehicle', 'person', 'building']
LIGHT = {'day': 1.0, 'dusk': 0.35, 'night': 0.06}       # brightness of the RGB camera
FOOTPRINT_PER_ALT = 1.2
VEHICLE_SIZE = (10.0, 4.5)                               # world units (length, width)
PERSON_RADIUS = 1.3


def _lowfreq(rng, n, amp):
    small = rng.normal(0.0, amp, (n, n)).astype(np.float32)
    return cv2.resize(small, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)


def _poly(cx, cy, length, width, ang):
    c, s = math.cos(ang), math.sin(ang)
    pts = []
    for u, v in ((-length / 2, -width / 2), (length / 2, -width / 2), (length / 2, width / 2), (-length / 2, width / 2)):
        pts.append((cx + u * c - v * s, cy + u * s + v * c))
    return np.array(pts, dtype=np.float32)


def render(rng, boxes, cx, cy, alt, light='day', vibration=0.0, n_vehicles=None, n_people=None, n_decoys=None, scene=None):
    """scene: optional dict from targets_near() with the world positions of vehicles, people, trees and rocks. Without it they are placed at random."""
    foot = FOOTPRINT_PER_ALT * alt
    scale = SIZE / foot                                   # pixels per world unit

    def to_px(x, y):
        return (x - cx) * scale + SIZE / 2.0, (cy - y) * scale + SIZE / 2.0

    # ground: earth and grass, low-frequency variation plus fine grain
    base = np.array([0.36, 0.42, 0.27], dtype=np.float32) + rng.normal(0, 0.03, 3).astype(np.float32)
    rgb = np.empty((SIZE, SIZE, 3), dtype=np.float32)
    patch = _lowfreq(rng, 8, 0.06)
    for k in range(3):
        rgb[:, :, k] = base[k] + patch * (1.0 - 0.3 * k)
    thermal = 0.30 + _lowfreq(rng, 6, 0.03)
    height = np.zeros((SIZE, SIZE), dtype=np.float32)
    labels = []

    half = foot / 2.0
    occupied = []                                         # world rectangles where nothing else is placed

    # buildings: the roofs of towers and floating blocks
    inside = boxes[(boxes[:, 3] > cx - half) & (boxes[:, 0] < cx + half) & (boxes[:, 4] > cy - half) & (boxes[:, 1] < cy + half)] if len(boxes) else boxes
    for b in inside:
        x0, y0 = to_px(b[0], b[4])
        x1, y1 = to_px(b[3], b[1])
        ix0, iy0, ix1, iy1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
        shade = 0.45 + 0.35 * min(b[5], 300.0) / 300.0
        col = (shade, shade * 0.97, shade * 0.9)
        cv2.rectangle(rgb, (ix0, iy0), (ix1, iy1), col, -1)
        cv2.rectangle(rgb, (ix0, iy0), (ix1, iy1), (shade * 0.6,) * 3, 1)
        cv2.rectangle(thermal, (ix0, iy0), (ix1, iy1), 0.33, -1)
        cv2.rectangle(height, (ix0, iy0), (ix1, iy1), min(b[5], 300.0) / 300.0, -1)
        vx0, vy0, vx1, vy1 = max(ix0, 0), max(iy0, 0), min(ix1, SIZE - 1), min(iy1, SIZE - 1)
        full = max(1, (ix1 - ix0) * (iy1 - iy0))
        if vx1 - vx0 >= 3 and vy1 - vy0 >= 3 and (vx1 - vx0) * (vy1 - vy0) >= 0.3 * full:
            labels.append((2, vx0, vy0, vx1, vy1))
        occupied.append((b[0] - 6, b[1] - 6, b[3] + 6, b[4] + 6))

    def free_spot():
        for _ in range(30):
            x, y = cx + rng.uniform(-half * 0.9, half * 0.9), cy + rng.uniform(-half * 0.9, half * 0.9)
            if not any(o[0] < x < o[2] and o[1] < y < o[3] for o in occupied):
                return x, y
        return None

    if scene is None:                                     # random placement (training data)
        scene = dict(trees=[], rocks=[], vehicles=[], people=[])
        for _ in range(int(rng.integers(0, 6)) if n_decoys is None else n_decoys):
            p = free_spot()
            if p is None:
                continue
            if rng.random() < 0.5:
                scene['trees'].append((p[0], p[1], rng.uniform(4, 8)))
            else:
                scene['rocks'].append((p[0], p[1], rng.uniform(2.5, 4), rng.uniform(0.6, 0.8), rng.uniform(0.35, 0.5)))
        for _ in range(int(rng.integers(0, 4)) if n_vehicles is None else n_vehicles):
            p = free_spot()
            if p is not None:
                scene['vehicles'].append((p[0], p[1], rng.uniform(0, math.pi), rng.uniform(0.8, 1.0), tuple(float(v) for v in rng.uniform(0.1, 0.95, 3))))
        for _ in range(int(rng.integers(0, 5)) if n_people is None else n_people):
            p = free_spot()
            if p is not None:
                scene['people'].append((p[0], p[1], rng.uniform(0.55, 0.75), tuple(float(v) for v in rng.uniform(0.1, 0.9, 3))))

    for x, y, r_w in [(t[0], t[1], t[2]) for t in scene['trees']]:
        px, py = to_px(x, y)
        r = max(1.5, r_w * scale)
        cv2.circle(rgb, (int(px), int(py)), int(r), (0.10, 0.28, 0.10), -1)
        cv2.circle(thermal, (int(px), int(py)), int(r), 0.22, -1)
        cv2.circle(height, (int(px), int(py)), int(r), 0.02, -1)
    for x, y, r_w, heat, g in scene['rocks']:
        px, py = to_px(x, y)
        r = max(1.5, r_w * scale)
        cv2.circle(rgb, (int(px), int(py)), int(r), (g, g, g), -1)
        cv2.circle(thermal, (int(px), int(py)), int(r), heat, -1)

    for x, y, ang, heat, col in scene['vehicles']:
        px, py = to_px(x, y)
        poly = _poly(px, py, VEHICLE_SIZE[0] * scale, VEHICLE_SIZE[1] * scale, ang)
        cv2.fillConvexPoly(rgb, poly.astype(np.int32), col)
        cv2.fillConvexPoly(thermal, poly.astype(np.int32), float(heat))
        cv2.fillConvexPoly(height, poly.astype(np.int32), 0.01)
        x0, y0 = poly.min(axis=0)
        x1, y1 = poly.max(axis=0)
        x0, y0, x1, y1 = max(int(x0), 0), max(int(y0), 0), min(int(math.ceil(x1)), SIZE - 1), min(int(math.ceil(y1)), SIZE - 1)
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            labels.append((0, x0, y0, x1, y1))

    for x, y, heat, col in scene['people']:
        px, py = to_px(x, y)
        r = max(1.2, PERSON_RADIUS * scale)
        d = int(math.ceil(r))
        cv2.circle(rgb, (int(px), int(py)), d, col, -1)
        cv2.circle(thermal, (int(px), int(py)), d, float(heat), -1)
        x0, y0, x1, y1 = int(px) - d, int(py) - d, int(px) + d, int(py) + d
        if 0 <= x0 and 0 <= y0 and x1 < SIZE and y1 < SIZE:
            labels.append((1, x0, y0, x1, y1))

    # camera model: light level, haze, vibration blur, noise
    bright = LIGHT[light] if isinstance(light, str) else float(light)
    rgb = np.clip(rgb, 0, 1) * bright
    rgb += rng.normal(0.0, 0.01 + (1.0 - bright) * 0.06, rgb.shape).astype(np.float32)
    thermal = thermal + rng.normal(0.0, 0.02, thermal.shape).astype(np.float32)
    height = height + rng.normal(0.0, 0.005, height.shape).astype(np.float32)
    if vibration > 0:
        k = 2 * int(vibration * 2) + 1
        rgb = cv2.GaussianBlur(rgb, (k, k), vibration)
        thermal = cv2.GaussianBlur(thermal, (k, k), vibration)
    img = np.concatenate([np.clip(rgb, 0, 1).transpose(2, 0, 1), np.clip(thermal, 0, 1)[None], np.clip(height, 0, 1)[None]], axis=0)
    return img.astype(np.float32), labels


def chunk_targets(seed, cx, cy, boxes, chunk=250.0):
    """The targets of one world chunk, the same every time for the same (seed, chunk): vehicles, people, trees, hot rocks.
    Nothing is placed on or right beside a building."""
    rng = np.random.default_rng([seed & 0xFFFFFFFF, cx + 7_000_003, cy + 7_000_033])
    out = dict(vehicles=[], people=[], trees=[], rocks=[])

    def free():
        for _ in range(30):
            x, y = cx * chunk + rng.uniform(0, chunk), cy * chunk + rng.uniform(0, chunk)
            if not len(boxes) or not np.any((x > boxes[:, 0] - 6) & (x < boxes[:, 3] + 6) & (y > boxes[:, 1] - 6) & (y < boxes[:, 4] + 6)):
                if math.hypot(x, y) > 60:                          # keep the spawn clear
                    return x, y
        return None
    for _ in range(int(rng.integers(0, 3))):
        p = free()
        if p:
            out['vehicles'].append((p[0], p[1], float(rng.uniform(0, math.pi)), float(rng.uniform(0.8, 1.0)), tuple(float(v) for v in rng.uniform(0.1, 0.95, 3))))
    for _ in range(int(rng.integers(0, 4))):
        p = free()
        if p:
            out['people'].append((p[0], p[1], float(rng.uniform(0.55, 0.75)), tuple(float(v) for v in rng.uniform(0.1, 0.9, 3))))
    for _ in range(int(rng.integers(0, 5))):
        p = free()
        if p:
            out['trees'].append((p[0], p[1], float(rng.uniform(4, 8))))
    for _ in range(int(rng.integers(0, 3))):
        p = free()
        if p:
            out['rocks'].append((p[0], p[1], float(rng.uniform(2.5, 4)), float(rng.uniform(0.6, 0.8)), float(rng.uniform(0.35, 0.5))))
    return out


def targets_near(env, x, y, radius):
    """All targets in the chunks around (x, y) of a DroneEnv (they avoid its buildings)."""
    chunk = 250.0
    out = dict(vehicles=[], people=[], trees=[], rocks=[])
    for cx in range(int(math.floor((x - radius) / chunk)), int(math.floor((x + radius) / chunk)) + 1):
        for cy in range(int(math.floor((y - radius) / chunk)), int(math.floor((y + radius) / chunk)) + 1):
            t = chunk_targets(env.seed, cx, cy, env._chunk(cx, cy), chunk)
            for k in out:
                out[k].extend(t[k])
    return out


def preview(path, n=6, seed=1):
    """A contact sheet of the 3 sensors in day and night, for a quick look."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        alt = rng.uniform(35, 100)
        boxes = np.array([[-20 + 40 * rng.random(), -20 + 40 * rng.random(), 0, 20 + 40 * rng.random(), 20 + 40 * rng.random(), 120.0]])
        st = rng.bit_generator.state
        day, lab = render(rng, boxes, 0, 0, alt, 'day')
        rng.bit_generator.state = st
        night, _ = render(rng, boxes, 0, 0, alt, 'night')
        sheet = []
        for im in (day, night):
            rgbv = (im[:3].transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8).copy()
            for c, x0, y0, x1, y1 in lab:
                cv2.rectangle(rgbv, (x0, y0), (x1, y1), [(0, 0, 255), (0, 255, 255), (255, 128, 0)][c], 1)
            sheet.append(rgbv)
        th = cv2.applyColorMap((im[3] * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        hh = cv2.cvtColor((im[4] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        rows.append(np.concatenate(sheet + [th, hh], axis=1))
    sheet = cv2.resize(np.concatenate(rows, axis=0), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), sheet)


if __name__ == '__main__':
    out = Path(__file__).resolve().parents[1] / 'results' / 'images'
    out.mkdir(parents=True, exist_ok=True)
    preview(out / 'aerial_preview.png')
    print('wrote results/images/aerial_preview.png  (columns: RGB day, RGB night, thermal, height)')
