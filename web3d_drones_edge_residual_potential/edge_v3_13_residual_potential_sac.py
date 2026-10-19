"""
RESIDUAL SAC ON AN OBSTACLE-AWARE POTENTIAL-FIELD BASELINE

Builds on edge_v3_9_residual_pid_sac.py (residual SAC on a fixed PID baseline), which builds on plain discrete SAC
(edge_v3_4_sac.py). edge_v3_9's baseline is blind to obstacles -- it only looks at bearing/elevation to the goal,
so it is a one-hot vote for a single "best" action with no idea a wall is there. This file's one change: the
baseline is now a simple potential field. A goal-attraction vector (from bearing/elevation, same as edge_v3_9) is
combined with an obstacle-repulsion vector built from the same 28 rangefinder signals already in the state
(each ray pushes back along its own direction, weighted by how close it is), and every action's bias becomes how
well that action's own movement direction aligns with the resulting combined vector -- a continuous score per
action, not a single preferred one. Obstacle avoidance is still primarily the network's job (the repulsion term is
a cheap heuristic, not a real potential-field controller), but the baseline itself no longer actively works against
avoidance the way a purely goal-seeking one-hot vote can.

This targets the same measured weakness as edge_v3_12 (calm-air collisions, 13.3 vs plain SAC's 0.0 per 1000
steps) from a different angle: instead of trusting the baseline less near obstacles (edge_v3_12's adaptive weight),
this makes the baseline itself aware obstacles exist at all, at a fixed blend weight.

General idea from the residual-RL literature (Residual Reinforcement Learning for Robot Control, Johannink et al.,
arXiv:1812.03201): a residual policy is only as good as what it does not have to correct; giving the fixed part of
the system a *better* (but still simple, still non-learned) prior reduces the correction burden on the network.

Critics are unaffected (they still learn plain Q-values); only the actor's logits gain the potential-field bias.

No charging stations, no battery in the state or reward -- that mechanic was tried separately (see
../web3d_drones_edge_battery_archive/) and did not produce brains that reliably sought a charger, so this line of
variants goes back to the plain environment and isolates purely the algorithmic change named above.

Standalone: one file, no project modules, needs only numpy and torch. It flies a drone through an endless 3D sky of towers and floating blocks
while the weather changes (calm, breeze, windy, gale, gale with dusty sensors), records every step for the 3D viewer, and evaluates the result
on the same skies and wind as the earlier tables (goals and collisions per 1000 steps, mean over 6 world seeds).

HOW TO RUN: paste the whole file into one Kaggle cell (GPU on) and run it, or  !python edge_v3_13_residual_potential_sac.py
            No command-line options are needed: this run's settings are in the SETTINGS block right below.

Outputs (in /kaggle/working): replay.jsonl (play it in the 3D viewer's Replay drawer), train_log.csv, brain_final.pth, brain_autosave.pth,
summary.json and eval_v3.json (compare runs with compare_evals.py).

NOT auto-generated: hand-written from edge_v3_9_residual_pid_sac.py as a base -- there is no source module for
"residual RL on a potential-field baseline" in build/ to regenerate this from.
"""

# ══════════════════════════════ SETTINGS OF THIS RUN (edit here) ══════════════════════════════
RUN_DEFAULTS = dict(
    agent='sac',
    minutes=60.0,
    shield=False,
    wind_comp=0.0,
    stack=1,
    hidden=256,
    lr=0.0003,
    n_step=3,
    train_every=2,
    entropy_frac=0.5,
    potential_weight=1.5,
    seed=7,
    # more options you can add: replay_every=5 (a 5x smaller replay file), no_replay=True, no_eval=True, eval_steps=6000, eval_seeds=6, wind_aware=False
)
# ═══════════════════════════════════════════════════════════════════════════════════════════════

import argparse
import copy
import csv
import json
import math
import os
import random
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ════════ drone environment ════════

# ── World / obstacle generation ────────────────────────────────────────────────
CHUNK = 250.0                       # world units per chunk edge
OBSTACLE_CLEAR_RADIUS = 140.0       # keep this much free around the origin spawn
TOWER_PROB = 0.38                   # otherwise a floating block (was 0.5 -- more floating obstacles, fewer towers)
COUNT_PROBS = [0.10, 0.25, 0.35, 0.30]     # 0..3 obstacles per chunk (was [0.25, 0.35, 0.25, 0.15] -- denser on average)
XY_SIZE = (25.0, 90.0)
TOWER_HEIGHT = (80.0, 300.0)
FLOAT_Z = (20.0, 190.0)              # was (25.0, 260.0) -- overlaps the drone's actual flight band (spawn z=80, GOAL_Z=(30,200)) instead of sitting mostly above it
FLOAT_THICK = (15.0, 35.0)           # was (20.0, 70.0) -- the viewer's floating obstacles are small (bird/drone-scale) models, not a solid filled shape, so the true collision volume should be close to their own size
FLOAT_XY_SIZE = (14.0, 32.0)         # separate from XY_SIZE (towers keep the old 25-90 range): a floating hazard is roughly as big as a real bird/drone's wander range now, not a city block

# ── Drone ───────────────────────────────────────────────────────────────────────
DRONE_RADIUS = 3.0                  # was 6.0 -- matches the visual body's actual half-size instead of a buffer well past it (the body mesh is only 3 units tall, 6 wide)
SPEED_FAST = 5.0                    # horizontal units / step
SPEED_SLOW = 1.0                    # right after a collision
CLIMB_RATE = 4.0                    # vertical units / step when climbing or descending
CLIMB_FORWARD_FACTOR = 0.8          # forward speed multiplier while climbing / descending
YAW_FORWARD = 15.0                  # degrees / step when turning while flying
YAW_HOVER = 25.0                    # degrees / step when turning in place
HOVER_STEP_COST = 0.05              # hovering is allowed but not free

# actions: (yaw degrees, forward factor, vertical direction)
ACTIONS = [
    (0.0,           1.0, 0),        # 0 fly straight
    (YAW_FORWARD,   1.0, 0),        # 1 fly + yaw left
    (-YAW_FORWARD,  1.0, 0),        # 2 fly + yaw right
    (0.0, CLIMB_FORWARD_FACTOR, +1),  # 3 fly + climb
    (0.0, CLIMB_FORWARD_FACTOR, -1),  # 4 fly + descend
    (YAW_HOVER,     0.0, 0),        # 5 hover + yaw left
    (-YAW_HOVER,    0.0, 0),        # 6 hover + yaw right
]
ACTION_NAMES = ["straight", "left", "right", "climb", "descend", "hover-left", "hover-right"]
N_ACTIONS = len(ACTIONS)

# ── Sensors: rays in the drone's yaw frame (elevation is measured from horizontal) ──
SENSOR_MAX_DIST = 150.0
_H_YAWS = [0, 20, -20, 40, -40, 70, -70, 100, -100, 130, -130, 160, -160, 180]   # 14 horizontal
_V_YAWS = [0, 60, -60, 120, -120, 180]                                            # 6 per tilted ring
_TILT = 35.0
SENSOR_DIRS = (
    [(y, 0.0) for y in _H_YAWS]
    + [(y, +_TILT) for y in _V_YAWS]
    + [(y, -_TILT) for y in _V_YAWS]
    + [(0.0, +90.0), (0.0, -90.0)]
)                                                                                   # 28 rays
N_SENSORS = len(SENSOR_DIRS)
_SENS_YAW = np.radians(np.array([d[0] for d in SENSOR_DIRS]))
_SENS_ELEV = np.radians(np.array([d[1] for d in SENSOR_DIRS]))
_LOCAL_DIRS = np.stack([np.cos(_SENS_ELEV) * np.cos(_SENS_YAW),
                        np.cos(_SENS_ELEV) * np.sin(_SENS_YAW),
                        np.sin(_SENS_ELEV)], axis=1)                                # (28, 3), yaw = 0 frame
N_STATE_EXTRA = 6

# ── Reward shaping (same ideas as v2/v3) ────────────────────────────────────────
COLLISION_PENALTY = -2.0
GOAL_RADIUS = 30.0
GOAL_DIST = (300.0, 550.0)          # how far ahead a new goal appears
GOAL_Z = (30.0, 200.0)
SPEED_REF = 1500
STUCK_LIMIT = 480
CIRCLING_LIMIT = 480
PROGRESS_MIN = 20.0
SPIN_LIMIT = 25
SPIN_PENALTY = 0.3
PROXIMITY_THRESHOLD = 0.75
PROXIMITY_PENALTY_SCALE = 0.5
PROXIMITY_FADE_DIST = 120.0
DIST_NORM = 600.0


class DroneEnv:

    def __init__(self, seed=1234, rng_seed=None):
        self.seed = int(seed)
        self.rng = np.random.default_rng(rng_seed)
        self._chunks = {}                       # (cx, cy) -> (k, 6) float array of boxes
        self.x = self.y = 0.0
        self.z = 80.0
        self.yaw = 0.0                          # degrees, CCW from +x
        self.vz = 0.0
        self.speed = SPEED_FAST
        self.goal = np.array([400.0, 0.0, 80.0])
        self.signals = [0.0] * N_SENSORS
        self.sensor_end = np.zeros((N_SENSORS, 3))
        self.is_colliding = False
        self.collision_kind = ""                # "", "obstacle", "ground"
        self.reward = 0.0
        self.distance = 0.0
        self.goals_reached = 0
        self.last_action = 0
        self.respawns = 0
        self._steps_to_goal = 0
        self._stuck_steps = 0
        self._best_dist = float("inf")
        self._no_progress = 0
        self._spin_steps = 0
        self._last_yaw_sign = 0
        self._bearing = 0.0
        self._elev = 0.0
        self.reset()

    # ── infinite obstacle field ─────────────────────────────────────────────────
    def _chunk(self, cx, cy):
        key = (cx, cy)
        arr = self._chunks.get(key)
        if arr is not None:
            return arr
        rng = np.random.default_rng([self.seed & 0xFFFFFFFF, cx + 1_000_003, cy + 1_000_033])
        n = int(rng.choice(len(COUNT_PROBS), p=COUNT_PROBS))
        boxes = []
        for _ in range(n):
            is_tower = rng.random() < TOWER_PROB
            size_range = XY_SIZE if is_tower else FLOAT_XY_SIZE
            w = rng.uniform(*size_range)
            d = rng.uniform(*size_range)
            x0 = cx * CHUNK + rng.uniform(0, CHUNK - w)
            y0 = cy * CHUNK + rng.uniform(0, CHUNK - d)
            if is_tower:
                z0, z1 = 0.0, rng.uniform(*TOWER_HEIGHT)
            else:
                z0 = rng.uniform(*FLOAT_Z)
                z1 = z0 + rng.uniform(*FLOAT_THICK)
            # keep the spawn area free
            nearest = math.hypot(min(max(0.0, x0), x0 + w), min(max(0.0, y0), y0 + d))   # origin -> box (xy)
            if nearest < OBSTACLE_CLEAR_RADIUS:
                continue
            boxes.append((x0, y0, z0, x0 + w, y0 + d, z1))
        arr = np.array(boxes, dtype=np.float64).reshape(-1, 6)
        if len(self._chunks) > 6000:            # bounded memory on very long flights
            self._chunks.clear()
        self._chunks[key] = arr
        return arr

    def boxes_near(self, x, y, radius):
        """All boxes in chunks overlapping the square [x-r, x+r] x [y-r, y+r]."""
        cx0, cx1 = int(math.floor((x - radius) / CHUNK)), int(math.floor((x + radius) / CHUNK))
        cy0, cy1 = int(math.floor((y - radius) / CHUNK)), int(math.floor((y + radius) / CHUNK))
        parts = [self._chunk(cx, cy) for cx in range(cx0, cx1 + 1) for cy in range(cy0, cy1 + 1)]
        parts = [p for p in parts if len(p)]
        return np.concatenate(parts) if parts else np.zeros((0, 6))

    def chunk_boxes_around(self, x, y, chunks_radius):
        """Boxes in the (2r+1)^2 chunks around (x, y), for the viewer."""
        ccx, ccy = int(math.floor(x / CHUNK)), int(math.floor(y / CHUNK))
        parts = [self._chunk(ccx + i, ccy + j)
                 for i in range(-chunks_radius, chunks_radius + 1)
                 for j in range(-chunks_radius, chunks_radius + 1)]
        parts = [p for p in parts if len(p)]
        return np.concatenate(parts) if parts else np.zeros((0, 6))

    def _point_free(self, x, y, z, margin):
        boxes = self.boxes_near(x, y, margin + 5)
        if z - margin < 0:
            return False
        if len(boxes) == 0:
            return True
        cx = np.clip(x, boxes[:, 0], boxes[:, 3])
        cy = np.clip(y, boxes[:, 1], boxes[:, 4])
        cz = np.clip(z, boxes[:, 2], boxes[:, 5])
        return bool(np.all((cx - x) ** 2 + (cy - y) ** 2 + (cz - z) ** 2 > margin * margin))

    # ── lifecycle ───────────────────────────────────────────────────────────────
    def reset(self, x=0.0, y=0.0, z=80.0, yaw=0.0):
        self.x, self.y, self.z, self.yaw = float(x), float(y), float(z), float(yaw)
        self.vz = 0.0
        self.speed = SPEED_FAST
        self._best_dist = float("inf")
        self._no_progress = 0
        self._spin_steps = 0
        self._last_yaw_sign = 0
        self._stuck_steps = 0
        self.new_goal()
        self._update_sensors()
        self._update_nav()

    def new_goal(self, min_dist=GOAL_DIST[0], max_dist=GOAL_DIST[1]):
        """A free 3D waypoint at a random bearing, distance and altitude from the drone. About half the time the
        bearing is aimed through a nearby obstacle instead of picked completely freely, so the direct drone-to-goal
        line often actually crosses something -- otherwise obstacles and goal-seeking are two unrelated systems and
        the drone can solve most goals without ever needing to dodge anything."""
        aim_ang = None
        if self.rng.random() < 0.5:
            nearby = self.boxes_near(self.x, self.y, max_dist)
            if len(nearby):
                b = nearby[self.rng.integers(len(nearby))]
                ocx, ocy = (b[0] + b[3]) / 2, (b[1] + b[4]) / 2
                aim_ang = math.atan2(ocy - self.y, ocx - self.x) + self.rng.uniform(-0.12, 0.12)
        for _ in range(200):
            ang = aim_ang if aim_ang is not None else self.rng.uniform(0, 2 * math.pi)
            dist = self.rng.uniform(min_dist, max_dist)
            gx, gy = self.x + dist * math.cos(ang), self.y + dist * math.sin(ang)
            gz = self.rng.uniform(*GOAL_Z)
            if self._point_free(gx, gy, gz, GOAL_RADIUS + 20):
                break
        self.goal = np.array([gx, gy, gz])
        self._best_dist = float("inf")
        self._no_progress = 0
        self._steps_to_goal = 0

    def respawn(self):
        """Put the drone in a free spot not far from where it is, with a fresh goal (the sky has
        no edge to fall off, so 'teleport' just means 'get unstuck')."""
        self.respawns += 1
        for _ in range(300):
            nx = self.x + self.rng.uniform(-400, 400)
            ny = self.y + self.rng.uniform(-400, 400)
            nz = self.rng.uniform(50, 180)
            if self._point_free(nx, ny, nz, DRONE_RADIUS + 25):
                break
        self.reset(nx, ny, nz, self.rng.uniform(0, 360))

    # ── stepping ────────────────────────────────────────────────────────────────
    def step(self, action):
        """Advance one step. Returns True when the drone is stuck and was respawned."""
        yaw_d, fwd_factor, vdir = ACTIONS[action]
        self.last_action = action
        prev_dist = self.distance
        px, py, pz = self.x, self.y, self.z

        # spin tracking (same-direction turning for too long)
        sign = (yaw_d > 0) - (yaw_d < 0)
        if sign != 0 and sign == self._last_yaw_sign:
            self._spin_steps += 1
        else:
            self._spin_steps = 0
        self._last_yaw_sign = sign

        self.yaw = (self.yaw + yaw_d) % 360.0
        move = self.speed * fwd_factor
        yr = math.radians(self.yaw)
        nx = self.x + move * math.cos(yr)
        ny = self.y + move * math.sin(yr)
        nz = self.z + vdir * CLIMB_RATE * (self.speed / SPEED_FAST)
        self.vz = nz - self.z

        # collision against nearby boxes and the ground plane
        self.is_colliding, self.collision_kind = False, ""
        if nz - DRONE_RADIUS < 0.0:
            self.is_colliding, self.collision_kind = True, "ground"
        else:
            boxes = self.boxes_near(nx, ny, DRONE_RADIUS + 2)
            if len(boxes):
                cx = np.clip(nx, boxes[:, 0], boxes[:, 3])
                cy = np.clip(ny, boxes[:, 1], boxes[:, 4])
                cz = np.clip(nz, boxes[:, 2], boxes[:, 5])
                if np.any((cx - nx) ** 2 + (cy - ny) ** 2 + (cz - nz) ** 2 < DRONE_RADIUS ** 2):
                    self.is_colliding, self.collision_kind = True, "obstacle"
        if self.is_colliding:
            self.x, self.y, self.z = px, py, pz
            self.vz = 0.0
        else:
            self.x, self.y, self.z = nx, ny, nz

        self._update_sensors()
        self._update_nav()
        return self._compute_reward(prev_dist, action)

    # ── perception ──────────────────────────────────────────────────────────────
    def _update_sensors(self):
        """Analytic 3D rangefinders: ray vs. axis-aligned boxes (slab method) + ground plane."""
        yr = math.radians(self.yaw)
        c, s = math.cos(yr), math.sin(yr)
        d = np.empty_like(_LOCAL_DIRS)
        d[:, 0] = _LOCAL_DIRS[:, 0] * c - _LOCAL_DIRS[:, 1] * s
        d[:, 1] = _LOCAL_DIRS[:, 0] * s + _LOCAL_DIRS[:, 1] * c
        d[:, 2] = _LOCAL_DIRS[:, 2]
        origin = np.array([self.x, self.y, self.z])

        best = np.full(N_SENSORS, SENSOR_MAX_DIST)
        # ground
        down = d[:, 2] < -1e-9
        t_ground = np.where(down, -self.z / np.where(down, d[:, 2], -1.0), np.inf)
        best = np.minimum(best, np.where(t_ground >= 0, t_ground, best))

        boxes = self.boxes_near(self.x, self.y, SENSOR_MAX_DIST)
        if len(boxes):
            inv = 1.0 / np.where(np.abs(d) < 1e-9, 1e-9, d)                 # (R, 3)
            b0, b1 = boxes[:, :3], boxes[:, 3:]                              # (M, 3)
            t1 = (b0[None, :, :] - origin) * inv[:, None, :]                 # (R, M, 3)
            t2 = (b1[None, :, :] - origin) * inv[:, None, :]
            tmin = np.minimum(t1, t2).max(axis=2)                            # (R, M)
            tmax = np.maximum(t1, t2).min(axis=2)
            hit = (tmax >= np.maximum(tmin, 0.0))
            t_hit = np.where(hit, np.maximum(tmin, 0.0), np.inf).min(axis=1)  # (R,)
            best = np.minimum(best, t_hit)

        best = np.maximum(best, 1.0)
        self.signals = (1.0 - best / SENSOR_MAX_DIST).tolist()
        self.sensor_end = origin + d * best[:, None]

    def _update_nav(self):
        gx, gy, gz = self.goal
        dx, dy, dz = gx - self.x, gy - self.y, gz - self.z
        dh = math.hypot(dx, dy)
        self.distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        # bearing: signed angle between heading and the goal direction in the horizontal plane
        goal_yaw = math.degrees(math.atan2(dy, dx))
        b = (goal_yaw - self.yaw + 180.0) % 360.0 - 180.0
        self._bearing = b / 180.0
        self._elev = math.degrees(math.atan2(dz, max(dh, 1e-6))) / 90.0

    def get_state(self):
        return [
            *self.signals,                                          # 28 obstacle readings, 0 = far, 1 = touching
            self._bearing, -self._bearing,                          # signed bearing to the goal
            self._elev,                                             # signed elevation to the goal
            min(1.0, self.distance / DIST_NORM),                    # normalised 3D distance
            min(1.0, self.z / 300.0),                               # altitude
            self.vz / CLIMB_RATE,                                   # vertical motion of the last step
        ]                                                           # total: 28 + 6 = 34

    # ── reward ──────────────────────────────────────────────────────────────────
    def _compute_reward(self, prev_dist, action):
        if self.is_colliding:
            self.speed = SPEED_SLOW
            self.reward = COLLISION_PENALTY
        else:
            self.speed = SPEED_FAST
            delta = prev_dist - self.distance
            self.reward = float(np.clip(delta * 0.2, -0.3, 0.5))
            closest = max(self.signals)
            if closest > PROXIMITY_THRESHOLD:
                relief = min(1.0, max(0.0, (self.distance - GOAL_RADIUS) / PROXIMITY_FADE_DIST))
                self.reward -= relief * PROXIMITY_PENALTY_SCALE * (closest - PROXIMITY_THRESHOLD)
            if ACTIONS[action][1] == 0.0:
                self.reward -= HOVER_STEP_COST

        self._steps_to_goal += 1
        if self.distance < GOAL_RADIUS:
            self.goals_reached += 1
            self.reward = 1.0 + max(0.0, 1.0 - self._steps_to_goal / SPEED_REF)
            self.new_goal()
            self._update_nav()

        if self._spin_steps >= SPIN_LIMIT:
            self.reward -= SPIN_PENALTY

        if self.reward <= -1.0:
            self._stuck_steps += 1
            if self._stuck_steps >= STUCK_LIMIT:
                self._stuck_steps = 0
                self.respawn()
                return True
        else:
            self._stuck_steps = 0

        if self.distance < self._best_dist - PROGRESS_MIN:
            self._best_dist = self.distance
            self._no_progress = 0
        else:
            self._no_progress += 1
            if self._no_progress >= CIRCLING_LIMIT:
                self.respawn()
                return True
        return False

# ════════ weather, wind, battery ════════

@dataclass(frozen=True)
class Weather:
    mean: float = 0.0            # mean wind speed, units/step
    gust: float = 0.0            # gust strength (std of the correlated gust process), units/step
    burst: float = 0.0           # extra strength of a short burst (a few per minute), units/step
    ray_noise: float = 0.0       # rangefinder noise, fraction of the max range (std)
    ray_dropout: float = 0.0     # chance that a ray returns nothing


CONDITIONS = {
    'calm':       Weather(),
    'breeze':     Weather(mean=0.8, gust=0.3, burst=0.5),
    'windy':      Weather(mean=1.6, gust=0.6, burst=1.0),
    'gale':       Weather(mean=2.6, gust=1.0, burst=1.5),
    'gale_dusty': Weather(mean=2.6, gust=1.0, burst=1.5, ray_noise=0.04, ray_dropout=0.03),
}
# how often training shows each condition (domain randomisation)
TRAIN_MIX = {'calm': 0.15, 'breeze': 0.25, 'windy': 0.30, 'gale': 0.20, 'gale_dusty': 0.10}

WIND_INPUTS = 3
GUST_TAU = 60.0                  # steps over which a gust decorrelates
BURST_EVERY = 600                # a burst starts about once per this many steps
BURST_LEN = 25
WIND_NORM = 3.0                  # wind estimate is divided by this before it reaches the network
WIND_ESTIMATE_NOISE = 0.15       # std of the wind estimate error, units/step

BATTERY_STEPS = 20000.0          # steps a full battery lasts when flying at full speed in calm air


class EdgeDroneEnv(DroneEnv):

    def __init__(self, seed=1234, rng_seed=None, weather='calm', wind_aware=True):
        self.wind_aware = wind_aware
        self.weather = CONDITIONS[weather] if isinstance(weather, str) else weather
        self.weather_name = weather if isinstance(weather, str) else 'custom'
        self.wind = np.zeros(3)                  # wind right now (world frame), units/step
        self._wind_dir = 0.0
        self._gust = np.zeros(3)
        self._burst_left = 0
        self._burst_vec = np.zeros(3)
        self.energy = 0.0                        # total energy used, in "full-speed calm steps"
        self.ground_dist = 0.0                   # distance actually covered over the ground
        self.wind_drift = 0.0                    # total distance the wind moved the drone
        self.wind_estimate = np.zeros(3)
        super().__init__(seed=seed, rng_seed=rng_seed)
        self._wind_dir = float(self.rng.uniform(0, 2 * math.pi))

    # ── weather ──────────────────────────────────────────────────────────────────
    def set_weather(self, weather):
        self.weather = CONDITIONS[weather] if isinstance(weather, str) else weather
        self.weather_name = weather if isinstance(weather, str) else 'custom'

    def _update_wind(self):
        w = self.weather
        if w.mean <= 0 and w.gust <= 0:
            self.wind = np.zeros(3)
            return
        self._wind_dir += self.rng.normal(0.0, 0.01)
        k = math.exp(-1.0 / GUST_TAU)
        self._gust = self._gust * k + w.gust * math.sqrt(1 - k * k) * self.rng.normal(size=3) * np.array([1.0, 1.0, 0.3])
        if self._burst_left > 0:
            self._burst_left -= 1
        elif w.burst > 0 and self.rng.random() < 1.0 / BURST_EVERY:
            self._burst_left = BURST_LEN
            a = self.rng.uniform(0, 2 * math.pi)
            self._burst_vec = w.burst * np.array([math.cos(a), math.sin(a), 0.0])
        mean = w.mean * np.array([math.cos(self._wind_dir), math.sin(self._wind_dir), 0.0])
        burst = self._burst_vec if self._burst_left > 0 else 0.0
        self.wind = mean + self._gust + burst

    def _measure_wind(self):
        """Wind in the drone's own frame (forward, left, up), the way an IMU / airspeed sensor would report it."""
        yr = math.radians(self.yaw)
        c, s = math.cos(yr), math.sin(yr)
        wf = self.wind[0] * c + self.wind[1] * s
        wl = -self.wind[0] * s + self.wind[1] * c
        est = np.array([wf, wl, self.wind[2]]) + self.rng.normal(0.0, WIND_ESTIMATE_NOISE, 3)
        self.wind_estimate = est
        return est

    # ── state: 34 base inputs, plus the wind estimate when wind_aware ────────────
    def get_state(self):
        s = super().get_state()
        if self.wind_aware:
            est = np.clip(self._measure_wind() / WIND_NORM, -1.5, 1.5)
            s = s + est.tolist()
        return s

    # ── perception with noise ────────────────────────────────────────────────────
    def _update_sensors(self):
        super()._update_sensors()
        w = self.weather
        if w.ray_noise > 0 or w.ray_dropout > 0:
            sig = np.asarray(self.signals)
            dist = (1.0 - sig) * SENSOR_MAX_DIST
            if w.ray_noise > 0:
                dist = np.clip(dist + self.rng.normal(0.0, w.ray_noise * SENSOR_MAX_DIST, N_SENSORS), 1.0, SENSOR_MAX_DIST)
            sig = 1.0 - dist / SENSOR_MAX_DIST
            if w.ray_dropout > 0:
                sig = np.where(self.rng.random(N_SENSORS) < w.ray_dropout, 0.0, sig)
            self.signals = sig.tolist()

    # ── stepping: the base step plus wind and energy ─────────────────────────────
    def step(self, action):
        yaw_d, fwd_factor, vdir = ACTIONS[action]
        self.last_action = action
        prev_dist = self.distance
        px, py, pz = self.x, self.y, self.z
        self._update_wind()

        sign = (yaw_d > 0) - (yaw_d < 0)
        if sign != 0 and sign == self._last_yaw_sign:
            self._spin_steps += 1
        else:
            self._spin_steps = 0
        self._last_yaw_sign = sign

        self.yaw = (self.yaw + yaw_d) % 360.0
        move = self.speed * fwd_factor
        yr = math.radians(self.yaw)
        nx = self.x + move * math.cos(yr) + self.wind[0]
        ny = self.y + move * math.sin(yr) + self.wind[1]
        nz = self.z + vdir * CLIMB_RATE * (self.speed / SPEED_FAST) + self.wind[2]
        self.vz = nz - self.z

        self.is_colliding, self.collision_kind = False, ""
        if nz - DRONE_RADIUS < 0.0:
            self.is_colliding, self.collision_kind = True, "ground"
        else:
            boxes = self.boxes_near(nx, ny, DRONE_RADIUS + 2)
            if len(boxes):
                cx = np.clip(nx, boxes[:, 0], boxes[:, 3])
                cy = np.clip(ny, boxes[:, 1], boxes[:, 4])
                cz = np.clip(nz, boxes[:, 2], boxes[:, 5])
                if np.any((cx - nx) ** 2 + (cy - ny) ** 2 + (cz - nz) ** 2 < DRONE_RADIUS ** 2):
                    self.is_colliding, self.collision_kind = True, "obstacle"
        if self.is_colliding:
            self.x, self.y, self.z = px, py, pz
            self.vz = 0.0
        else:
            self.x, self.y, self.z = nx, ny, nz
            self.ground_dist += math.sqrt((nx - px) ** 2 + (ny - py) ** 2 + (nz - pz) ** 2)
        self.wind_drift += float(np.linalg.norm(self.wind))

        # energy: base + about speed cubed + climbing + working against the wind
        v = (self.speed * fwd_factor) / SPEED_FAST
        self.energy += 0.3 + 0.7 * v ** 3 + (0.15 if vdir else 0.0) + 0.1 * float(np.dot(self.wind, self.wind))

        self._update_sensors()
        self._update_nav()
        return self._compute_reward(prev_dist, action)

    @property
    def battery_percent(self):
        return max(0.0, 100.0 * (1.0 - self.energy / BATTERY_STEPS))


# ════════ 9 actions, wind-scaled caution ════════

# ── v2 begins here (the standalone build starts at this line) ────────────────────────────────────────────
ACTIONS.append((0.0, 0.5, 0))       # 7 slow: straight at half speed
ACTIONS.append((0.0, -0.5, 0))      # 8 reverse: backwards at half speed
ACTION_NAMES.extend(["slow", "reverse"])
N_ACTIONS = len(ACTIONS)

# how often training shows each condition (v2): more gale than v1
TRAIN_MIX_V2 = {'calm': 0.10, 'breeze': 0.20, 'windy': 0.30, 'gale': 0.25, 'gale_dusty': 0.15}
WIND_PROX_GAIN = 1.0                # extra proximity penalty = this * (wind / 2.6) * the normal proximity penalty
GALE_WIND = 2.6


class EdgeDroneEnvV2(EdgeDroneEnv):

    def step(self, action):
        fwd = ACTIONS[action][1]
        v = abs(self.speed * fwd) / SPEED_FAST
        resp = super().step(action)
        if fwd < 0:                                          # the parent added 0.7 * v**3 with a negative v: make it a cost
            self.energy += 2.0 * 0.7 * v ** 3
        if not resp and not self.is_colliding and self.reward < 1.0:
            closest = max(self.signals)
            if closest > PROXIMITY_THRESHOLD:
                relief = min(1.0, max(0.0, (self.distance - GOAL_RADIUS) / PROXIMITY_FADE_DIST))
                wind = float(math.hypot(self.wind[0], self.wind[1]))
                self.reward -= relief * PROXIMITY_PENALTY_SCALE * WIND_PROX_GAIN * (wind / GALE_WIND) * (closest - PROXIMITY_THRESHOLD)
        return resp


# ════════ shared v3 parts: wind compensation, safety filter, replay, evaluation ════════

# ── v3 common begins here (the standalone build starts at this line) ─────────────────────────────────────
WEATHER_NAMES = list(CONDITIONS)
RANDOM_STEPS_AFTER_RESPAWN = 90
SHIELD_MARGIN = 8.0                 # extra clearance the safety filter wants in front of the drone, in world units


class EdgeDroneEnvV3(EdgeDroneEnvV2):
    """EdgeDroneEnvV2 plus an optional wind compensation, like the wind feed-forward of a real flight controller.

    With wind_comp = 0.7 the controller uses its own noisy wind measurement to cancel 70% of the push the wind gives every step
    (the drone still feels the remaining 30% and the measurement error). The brain still sees the ambient wind. Working against the wind costs
    the same energy as before, so energy figures stay comparable. wind_comp = 0 behaves exactly like EdgeDroneEnvV2."""

    def __init__(self, seed=1234, rng_seed=None, weather='calm', wind_aware=True, wind_comp=0.0):
        self.wind_comp = float(wind_comp)
        self.wind_true = np.zeros(3)
        super().__init__(seed=seed, rng_seed=rng_seed, weather=weather, wind_aware=wind_aware)

    def _update_wind(self):
        super()._update_wind()                                   # self.wind = the ambient wind
        self.wind_true = self.wind.copy()
        if self.wind_comp > 0.0:
            measured = self.wind_true + self.rng.normal(0.0, WIND_ESTIMATE_NOISE, 3)      # the controller's own noisy measurement
            self.wind = self.wind_true - self.wind_comp * measured                        # the push that is left over

    def _measure_wind(self):                                     # the brain is told the ambient wind, in the drone's own frame
        yr = math.radians(self.yaw)
        c, s = math.cos(yr), math.sin(yr)
        wt = self.wind_true
        est = np.array([wt[0] * c + wt[1] * s, -wt[0] * s + wt[1] * c, wt[2]]) + self.rng.normal(0.0, WIND_ESTIMATE_NOISE, 3)
        self.wind_estimate = est
        return est

    def step(self, action):
        resp = super().step(action)
        if self.wind_comp > 0.0:                                 # cancelling the wind is not free: same energy as flying in the full wind
            self.energy += 0.1 * (float(np.dot(self.wind_true, self.wind_true)) - float(np.dot(self.wind, self.wind)))
        return resp


def shield_mask(env, margin=SHIELD_MARGIN):
    """The safety filter. For every action: where would the drone be after this step (the action plus the push the wind is expected to give)?
    Look at the rangefinder rays that point that way. The action is allowed if the nearest obstacle is farther than the step length plus the drone
    radius plus a margin, and the drone stays off the ground. If no action passes, the one with the most room is allowed. Returns a boolean array."""
    est = env.wind_estimate * (1.0 - env.wind_comp) if env.wind_aware else np.zeros(3)     # expected push, in the drone's own frame
    dist = (1.0 - np.asarray(env.signals)) * SENSOR_MAX_DIST
    ok = np.zeros(len(ACTIONS), dtype=bool)
    room = np.zeros(len(ACTIONS))
    for a, (yaw_d, fwd, vdir) in enumerate(ACTIONS):
        move = env.speed * fwd
        yr = math.radians(yaw_d)
        d = np.array([move * math.cos(yr) + est[0], move * math.sin(yr) + est[1], vdir * CLIMB_RATE * (env.speed / SPEED_FAST) + est[2]])
        length = float(np.linalg.norm(d))
        if length < 1e-6:
            ok[a], room[a] = True, SENSOR_MAX_DIST
            continue
        dots = _LOCAL_DIRS @ (d / length)
        near = dots > 0.85
        ray = float(dist[near].min()) if near.any() else float(dist[int(np.argmax(dots))])
        clear = ray - length
        if env.z + d[2] - DRONE_RADIUS < 3.0:
            clear = min(clear, -1.0)
        room[a] = clear
        ok[a] = clear > DRONE_RADIUS + margin
    if not ok.any():
        ok[int(np.argmax(room))] = True
    return ok


class StateStack:
    """The last k states glued together, so the brain can see how the wind and the obstacles changed a moment ago."""

    def __init__(self, k):
        self.k = k
        self.buf = deque(maxlen=k)

    def push(self, state):
        s = np.asarray(state, dtype=np.float32)
        if not self.buf:
            for _ in range(self.k):
                self.buf.append(s)
        else:
            self.buf.append(s)
        return np.concatenate(self.buf)

    def reset(self):
        self.buf.clear()


class ReplayWriter:
    """replay.jsonl for the 3D viewer: a meta line, then one line per recorded step (20 numbers, see train_edge_v2.py)."""

    def __init__(self, path, meta, every=1, enabled=True):
        self.fh = open(path, 'w', encoding='utf-8') if enabled else None
        self.buf = [json.dumps(meta)] if self.fh else []
        self.every = max(1, every)

    def add(self, step, env, action, score, gradient_steps):
        if self.fh is None or step % self.every:
            return
        self.buf.append(f'[{step},{env.x:.1f},{env.y:.1f},{env.z:.1f},{env.yaw:.1f},{action},{env.reward:.3f},{int(env.is_colliding)},'
                        f'{env.goal[0]:.0f},{env.goal[1]:.0f},{env.goal[2]:.0f},{env.goals_reached},{score:.3f},{gradient_steps},{env.vz:.1f},'
                        f'{env.wind[0]:.2f},{env.wind[1]:.2f},{env.wind[2]:.2f},{env.battery_percent:.1f},{WEATHER_NAMES.index(env.weather_name)}]')
        if len(self.buf) >= 2000:
            self.flush()

    def flush(self):
        if self.fh and self.buf:
            self.fh.write('\n'.join(self.buf) + '\n')
            self.fh.flush()
            self.buf.clear()

    def close(self):
        self.flush()
        if self.fh:
            self.fh.close()


class TrainLog:
    def __init__(self, path, fields):
        self.fh = open(path, 'w', newline='')
        self.w = csv.writer(self.fh)
        self.w.writerow(fields)

    def row(self, *values):
        self.w.writerow(values)
        self.fh.flush()

    def close(self):
        self.fh.close()


def to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    return obj


def save_checkpoint(path, payload):
    tmp = path.with_suffix('.tmp')
    torch.save(to_cpu(payload), tmp)
    os.replace(tmp, path)


def weather_probs_generic(mix, progress, curriculum, frac):
    """Probabilities of the conditions. With a curriculum the harsh ones (windy, gale, gale_dusty) fade in over the first `frac` of the run."""
    p = np.array(list(mix.values()), dtype=float)
    if curriculum:
        f = min(1.0, progress / max(frac, 1e-6))
        p = p * (1.0 - np.array([0.0, 0.0, 1.0, 1.0, 1.0]) * (1.0 - f))
    return p / p.sum()


def evaluate_generic(make_env, make_policy, conditions, world_seeds, steps, label=''):
    """Every (condition, world seed) is one run of `steps` steps on its own sky. The same skies and wind as evaluate_edge.py (world seeds 1000..1005).
    make_env(world_seed, condition) -> env; make_policy() -> an object with act(env, rng, explore) and on_respawn()."""
    result = {}
    for cond in conditions:
        runs = []
        for ws in world_seeds:
            env, pol, rng = make_env(ws, cond), make_policy(), random.Random(ws)
            goals = coll = resp = ok = allx = random_steps = 0
            for _ in range(steps):
                a = pol.act(env, rng, random_steps > 0)
                random_steps = max(0, random_steps - 1)
                g0 = env.goals_reached
                respawned = env.step(a)
                coll += int(env.is_colliding)
                if env.goals_reached > g0:
                    goals += 1
                    ok += 1
                    allx += 1
                if respawned:
                    resp += 1
                    allx += 1
                    random_steps = RANDOM_STEPS_AFTER_RESPAWN
                    pol.on_respawn()
            runs.append(dict(goals_per_1k=1000.0 * goals / steps, collisions_per_1k=1000.0 * coll / steps, respawns_per_1k=1000.0 * resp / steps,
                             success_rate=ok / allx if allx else 0.0, energy_per_goal=env.energy / goals if goals else float('nan'),
                             ground_speed=env.ground_dist / steps))
        agg = {}
        for k in runs[0]:
            vals = np.array([r[k] for r in runs], dtype=float)
            good = vals[~np.isnan(vals)]
            agg[k] = (float(good.mean()) if len(good) else float('nan'), float(good.std()) if len(good) else float('nan'))
        result[cond] = dict(agg=agg, runs=runs)
        print(f'  {label}{cond:11s} goals/1k {agg["goals_per_1k"][0]:5.2f}  collisions/1k {agg["collisions_per_1k"][0]:6.1f}  success {agg["success_rate"][0]:.2f}  '
              f'energy/goal {agg["energy_per_goal"][0]:7.1f}  ground speed {agg["ground_speed"][0]:.2f}', flush=True)
    return result


# ════════ agent ════════

# ── Prioritized Experience Replay ──────────────────────────────────────────────
#
# Transitions with high |TD-error| are sampled more often (alpha controls how
# much).  Importance-sampling weights (IS-weights) correct the resulting
# gradient bias; beta anneals from beta_start -> 1.

try:
    import fastreplay                      # optional C++ (pybind11) sum-tree: O(log n) sample/update instead of
except ImportError:                        # numpy's O(n) cumsum -- ~60x faster at capacity=500_000/batch=256
    fastreplay = None                      # (see ../web3d_drones_edge/src/fastray_cpp/fastreplay.cpp). Falls back
                                            # to plain numpy when the module isn't built, so this file still runs standalone.


class PrioritizedReplayMemory:
    """Flat numpy priority array + cumsum/searchsorted (or a C++ sum-tree when `fastreplay` is built), transitions
    in preallocated numpy arrays (see ai_v2.py). Every transition also stores its own discount (gamma ** n, shorter
    at a respawn)."""

    def __init__(self, capacity, input_size, alpha=0.6, beta_start=0.4, beta_frames=400_000):
        self.capacity     = capacity
        self.prio         = np.zeros(capacity, dtype=np.float64)
        self.states       = np.zeros((capacity, input_size), dtype=np.float32)
        self.next_states  = np.zeros((capacity, input_size), dtype=np.float32)
        self.actions      = np.zeros(capacity, dtype=np.int64)
        self.rewards      = np.zeros(capacity, dtype=np.float32)
        self.discounts    = np.zeros(capacity, dtype=np.float32)
        self.write        = 0
        self.n_entries    = 0
        self.alpha        = alpha
        self.beta_start   = beta_start
        self.beta_frames  = beta_frames
        self.frame        = 1
        self.epsilon      = 1e-5
        self.max_priority = 1.0
        self._tree        = fastreplay.SumTree(capacity) if fastreplay is not None else None

    def push(self, state, next_state, action, reward, discount):
        w = self.write
        self.prio[w]        = self.max_priority
        self.states[w]      = state
        self.next_states[w] = next_state
        self.actions[w]     = action
        self.rewards[w]     = reward
        self.discounts[w]   = discount
        if self._tree is not None:
            self._tree.set(w, self.max_priority)
        self.write     = (w + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def sample(self, batch_size):
        n    = self.n_entries
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        if self._tree is not None:                                 # C++ sum-tree: O(log n) per sample instead of O(n) cumsum
            idxs, p, total = self._tree.sample_batch(batch_size)
            probs = p / total
        else:
            priorities = self.prio[:n]
            cumsum     = np.cumsum(priorities)
            total      = cumsum[-1]
            segment    = total / batch_size
            targets    = (np.arange(batch_size) + np.random.random(batch_size)) * segment
            idxs       = np.minimum(np.searchsorted(cumsum, targets), n - 1)
            probs      = priorities[idxs] / total
        weights = (n * probs) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)
        batch = (torch.from_numpy(self.states[idxs]), torch.from_numpy(self.next_states[idxs]),
                 torch.from_numpy(self.actions[idxs]), torch.from_numpy(self.rewards[idxs]), torch.from_numpy(self.discounts[idxs]))
        return batch, idxs, torch.from_numpy(weights)

    def update_priorities(self, idxs, td_errors):
        p = (np.abs(td_errors) + self.epsilon) ** self.alpha
        self.prio[idxs] = p
        if self._tree is not None:
            self._tree.set_many(np.asarray(idxs, dtype=np.int64), p.astype(np.float64))
        self.max_priority = max(self.max_priority, float(p.max()))

    def __len__(self):
        return self.n_entries


# ── Dueling Network ────────────────────────────────────────────────────────────
#   Q(s,a) = V(s) + A(s,a) - mean_a' A(s,a')      trunk: input -> hidden -> hidden, heads: hidden -> hidden/2 -> (1 | actions)

class Network(nn.Module):

    def __init__(self, input_size, nb_action, hidden=128):
        super().__init__()
        head = hidden // 2
        self.fc1  = nn.Linear(input_size, hidden)
        self.fc2  = nn.Linear(hidden, hidden)
        self.val1 = nn.Linear(hidden, head)
        self.val2 = nn.Linear(head, 1)
        self.adv1 = nn.Linear(hidden, head)
        self.adv2 = nn.Linear(head, nb_action)

    def forward(self, state):
        x   = F.relu(self.fc1(state))
        x   = F.relu(self.fc2(x))
        val = self.val2(F.relu(self.val1(x)))
        adv = self.adv2(F.relu(self.adv1(x)))
        return val + adv - adv.mean(dim=1, keepdim=True)


# ── Deep Q-Learning agent ──────────────────────────────────────────────────────

class Dqn:

    def __init__(self, input_size, nb_action, gamma, train_every=2, device='cpu', hidden=128, n_step=3, memory=500_000, lr=5e-4, batch=256):
        """
        `train_every`: one gradient step every N calls to update(). Acting uses a numpy copy of the online network (see ai_v2.py).
        `n_step`: length of the return used for learning (1 = the old one-step DQN).
        """
        self.gamma         = gamma
        self.n_step        = max(1, int(n_step))
        self.batch         = batch
        self.device        = torch.device(device)
        self.reward_window = deque(maxlen=1000)
        self._rsum         = 0.0
        self.hidden        = hidden
        self.model         = self._make_net(input_size, nb_action, hidden)
        self.target_model  = self._make_net(input_size, nb_action, hidden)
        self.model.to(self.device)
        self.target_model.to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.memory        = PrioritizedReplayMemory(memory, input_size)
        self.optimizer     = optim.Adam(self.model.parameters(), lr=lr)
        self.last_state    = np.zeros(input_size, dtype=np.float32)
        self.last_action   = 0
        self.steps         = 0                  # gradient steps
        self.nb_action     = nb_action
        self.random_steps  = 0
        self.train_every   = max(1, int(train_every))
        self._calls        = 0                  # sim steps
        self._nbuf         = deque()            # the last n (state, action, reward)
        self._flush        = False
        self._sync_np()

    def _make_net(self, input_size, nb_action, hidden):
        return Network(input_size, nb_action, hidden)

    # -- numpy copy of the online network, used for acting ------------------------
    def _sync_np(self):
        m = self.model
        self._np = [(w.detach().cpu().numpy().T.astype(np.float32).copy(),
                     b.detach().cpu().numpy().astype(np.float32).copy())
                    for w, b in ((m.fc1.weight, m.fc1.bias), (m.fc2.weight, m.fc2.bias),
                                 (m.val1.weight, m.val1.bias), (m.val2.weight, m.val2.bias),
                                 (m.adv1.weight, m.adv1.bias), (m.adv2.weight, m.adv2.bias))]

    def _q_np(self, s):
        (w1, b1), (w2, b2), (wv1, bv1), (wv2, bv2), (wa1, ba1), (wa2, ba2) = self._np
        x = np.maximum(s @ w1 + b1, 0.0)
        x = np.maximum(x @ w2 + b2, 0.0)
        val = np.maximum(x @ wv1 + bv1, 0.0) @ wv2 + bv2
        adv = np.maximum(x @ wa1 + ba1, 0.0) @ wa2 + ba2
        return val + adv - adv.mean()

    def temperature(self):
        return max(15.0, 100.0 - self._calls * 0.00025)      # same schedule per sim step as before: 15 after 340k steps

    def select_action(self, state, mask=None):
        """mask: optional boolean array, False = this action may not be chosen (the safety filter)."""
        if self.random_steps > 0:
            self.random_steps -= 1
            if mask is None:
                return random.randint(0, self.nb_action - 1)
            return random.choice([a for a in range(self.nb_action) if mask[a]])
        z = self._q_np(state) * self.temperature()             # Boltzmann: softmax(Q * T), sampled
        if mask is not None:
            z = np.where(mask, z, -1e9)
        p = np.exp(z - z.max())
        c = np.cumsum(p)
        return min(int(np.searchsorted(c, random.random() * c[-1])), self.nb_action - 1)

    def eval_action(self, state, mask, rng):
        """The action used when a finished brain is tested: Boltzmann with T = 15 (nearly greedy), like evaluate_edge.py. `rng` is a random.Random."""
        z = self._q_np(state) * 15.0
        if mask is not None:
            z = np.where(mask, z, -1e9)
        p = np.exp(z - z.max())
        c = np.cumsum(p)
        return min(int(np.searchsorted(c, rng.random() * c[-1])), self.nb_action - 1)

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def learn(self, batch_state, batch_next_state, batch_action, batch_reward, batch_discount, idxs, weights):
        dev = self.device
        batch_state, batch_next_state = batch_state.to(dev), batch_next_state.to(dev)
        batch_reward, batch_action, weights = batch_reward.to(dev), batch_action.to(dev), weights.to(dev)
        batch_discount = batch_discount.to(dev)

        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_next = self.model(batch_next_state).argmax(1, keepdim=True)                        # Double DQN: online net picks ...
            next_outputs = self.target_model(batch_next_state).gather(1, best_next).squeeze(1)      # ... target net evaluates
        target    = batch_reward + batch_discount * next_outputs                                     # n-step target: R + gamma^n * Q(s_t+n)
        td_errors = (outputs - target).detach().abs()
        loss = (weights * F.smooth_l1_loss(outputs, target, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self._sync_np()
        self.memory.update_priorities(idxs, td_errors.cpu().numpy())
        self.steps += 1
        if self.steps % 500 == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    # -- n-step bookkeeping -----------------------------------------------------------
    def _push_from(self, start, next_state):
        """Push the transition that starts at buffer item `start` and runs to the end of the buffer."""
        ret, g = 0.0, 1.0
        for k in range(start, len(self._nbuf)):
            ret += g * self._nbuf[k][2]
            g *= self.gamma
        s0, a0, _ = self._nbuf[start]
        self.memory.push(s0, next_state, a0, ret, g)

    def flush_nstep(self):
        """Call after the drone was respawned (teleported): the next update() closes all open returns, so none of them runs across the jump."""
        self._flush = True

    def update(self, reward, new_signal, mask=None):
        new_state = np.asarray(new_signal, dtype=np.float32)
        self._nbuf.append((self.last_state, self.last_action, reward))
        if self._flush:
            for start in range(len(self._nbuf)):
                self._push_from(start, new_state)
            self._nbuf.clear()
            self._flush = False
        elif len(self._nbuf) == self.n_step:
            self._push_from(0, new_state)
            self._nbuf.popleft()
        action = self.select_action(new_state, mask)
        self._calls += 1
        if len(self.memory) > 1024 and self._calls % self.train_every == 0:
            (bs, bns, ba, br, bd), idxs, weights = self.memory.sample(self.batch)
            self.learn(bs, bns, ba, br, bd, idxs, weights)
        self.last_action = action
        self.last_state  = new_state
        if len(self.reward_window) == self.reward_window.maxlen:
            self._rsum -= self.reward_window[0]
        self.reward_window.append(reward)
        self._rsum += reward
        return action

    def score(self):
        n = len(self.reward_window)
        return self._rsum / n if n else 0.0

    def get_q_values(self):
        return self._q_np(self.last_state).tolist()


# ── Warm start: an older, smaller brain inside this network ─────────────────────────

def warm_start_from(brain, path, wind_aware):
    """Load an older brain (fewer hidden units, fewer actions, no wind inputs) so that it behaves the same at first.
    New hidden units start with random inputs and zero outputs, new inputs start at 0, and every new action starts as a copy of action 0
    (straight) with a slightly lower bias, so it gets tried but does not take over. Checkpoints of this same size are simply copied."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    old = ck['state_dict']
    new = {k: v.detach().cpu() for k, v in brain.model.state_dict().items()}        # work on the CPU, load_state_dict moves it to the GPU
    H0, H = old['fc1.weight'].shape[0], new['fc1.weight'].shape[0]
    h0, h = old['val1.weight'].shape[0], new['val1.weight'].shape[0]
    n0, n = old['adv2.weight'].shape[0], new['adv2.weight'].shape[0]
    i0, i = old['fc1.weight'].shape[1], new['fc1.weight'].shape[1]
    if H < H0 or h < h0 or n < n0 or i < i0:
        raise SystemExit(f'cannot load {path}: it is larger than this network (hidden {H0} vs {H}, actions {n0} vs {n}, inputs {i0} vs {i})')
    out = {k: v.clone() for k, v in new.items()}
    out['fc1.weight'][:H0, :i0] = old['fc1.weight']; out['fc1.weight'][:H0, i0:] = 0.0; out['fc1.bias'][:H0] = old['fc1.bias']
    out['fc2.weight'][:H0, :H0] = old['fc2.weight']; out['fc2.weight'][:H0, H0:] = 0.0; out['fc2.bias'][:H0] = old['fc2.bias']
    for name, rows0 in (('val1', h0), ('adv1', h0)):
        out[name + '.weight'][:rows0, :H0] = old[name + '.weight']; out[name + '.weight'][:rows0, H0:] = 0.0
        out[name + '.bias'][:rows0] = old[name + '.bias']
    out['val2.weight'][:, :h0] = old['val2.weight']; out['val2.weight'][:, h0:] = 0.0; out['val2.bias'] = old['val2.bias'].clone()
    out['adv2.weight'][:n0, :h0] = old['adv2.weight']; out['adv2.weight'][:n0, h0:] = 0.0; out['adv2.bias'][:n0] = old['adv2.bias']
    for a in range(n0, n):                                   # new actions
        out['adv2.weight'][a, :h0] = old['adv2.weight'][0]; out['adv2.weight'][a, h0:] = 0.0
        out['adv2.bias'][a] = old['adv2.bias'][0] - 0.05
    brain.model.load_state_dict(out)
    brain.target_model.load_state_dict(out)
    brain._sync_np()
    return dict(hidden=(H0, H), actions=(n0, n), inputs=(i0, i))


# ════════ agent ════════

# ── PID baseline for the residual actor ─────────────────────────────────────────────────────────────────
# A proportional rule with a deadband, correcting altitude before heading, standing in for a real PID (this
# environment's actions are discrete, so there is no continuous motor command to PID onto -- this picks which of
# the 9 discrete actions a classical controller would prefer). Reuses get_state()'s bearing (index 28) and
# elevation (index 30) -- no new sensing, just a fixed rule layered under the learned residual.
PID_BEARING_IDX, PID_ELEV_IDX = 28, 30
REPULSE_GAIN = 2.5        # how strongly a close ray pushes the desired direction away from it

# every action's own movement direction, in the same local (yaw-relative) frame as the sensor rays -- used to score
# how well each action agrees with the potential field's combined goal-attraction + obstacle-repulsion vector
_ACTION_DIRS = np.array([
    [math.cos(math.radians(yaw_d)), math.sin(math.radians(yaw_d)), (0.35 * vdir if vdir else (0.15 if fwd > 0 else -0.15 if fwd < 0 else 0.0))]
    for (yaw_d, fwd, vdir) in ACTIONS
], dtype=np.float32)
_ACTION_DIRS /= np.linalg.norm(_ACTION_DIRS, axis=1, keepdims=True) + 1e-9


def _pid_bias_np(s):
    """Continuous per-action alignment with a potential field: goal-attraction (bearing/elevation) plus
    obstacle-repulsion (every rangefinder ray pushes back along its own direction, weighted by closeness squared)."""
    bearing, elev = float(s[PID_BEARING_IDX]), float(s[PID_ELEV_IDX])
    goal_dir = np.array([math.cos(bearing * math.pi), math.sin(bearing * math.pi), elev], dtype=np.float32)
    sig = np.asarray(s[:N_SENSORS], dtype=np.float32)
    repulse = -(sig[:, None] ** 2 * _LOCAL_DIRS).sum(axis=0).astype(np.float32)
    desired = goal_dir + REPULSE_GAIN * repulse
    n = float(np.linalg.norm(desired))
    if n > 1e-6:
        desired = desired / n
    return (_ACTION_DIRS @ desired).astype(np.float32)


def _pid_bias_torch(s):
    """Batched version of _pid_bias_np: s is (B, n_in) -> (B, N_ACTIONS) continuous alignment scores."""
    bearing, elev = s[:, PID_BEARING_IDX], s[:, PID_ELEV_IDX]
    goal_dir = torch.stack([torch.cos(bearing * math.pi), torch.sin(bearing * math.pi), elev], dim=1)    # (B, 3)
    sig = s[:, :N_SENSORS]                                                                                # (B, N_SENSORS)
    local_dirs = torch.as_tensor(_LOCAL_DIRS, dtype=s.dtype, device=s.device)                              # (N_SENSORS, 3)
    repulse = -torch.einsum('bn,nk->bk', sig ** 2, local_dirs)                                             # (B, 3)
    desired = goal_dir + REPULSE_GAIN * repulse
    desired = desired / (desired.norm(dim=1, keepdim=True) + 1e-6)
    action_dirs = torch.as_tensor(_ACTION_DIRS, dtype=s.dtype, device=s.device)                            # (N_ACTIONS, 3)
    return desired @ action_dirs.T                                                                          # (B, N_ACTIONS)


# ── SAC begins here (the standalone build starts at this line) ───────────────────────────────────────────

class SacNets(nn.Module):
    """actor (logits over the actions), critic 1 and critic 2 (a Q value per action), each input -> hidden -> hidden -> actions"""

    def __init__(self, input_size, nb_action, hidden=256):
        super().__init__()

        def mlp():
            return nn.Sequential(nn.Linear(input_size, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, nb_action))
        self.actor, self.q1, self.q2 = mlp(), mlp(), mlp()

    def forward(self, x):
        return self.actor(x), self.q1(x), self.q2(x)


class SacDiscrete(Dqn):
    """RESIDUAL SAC (this file's one change to plain SAC): the actor's logits are pid_weight * _pid_bias(s) + the
    network's own output, everywhere the policy is used (acting, and both the current and next-state logits in
    learn()). The network only ever has to learn the correction on top of the PID baseline, not the whole policy
    from scratch -- the same idea as residual RL on top of an existing PID controller."""

    def __init__(self, input_size, nb_action, gamma, tau=0.005, entropy_frac=0.5, potential_weight=1.5, **kw):
        super().__init__(input_size, nb_action, gamma, **kw)
        self.tau = tau
        self.pid_weight = potential_weight       # kept as self.pid_weight so the rest of this file is unchanged from edge_v3_9
        self.target_entropy = entropy_frac * math.log(nb_action)          # the amount of entropy (in nats) the policy should keep
        self.log_alpha = torch.zeros(1, device=self.device, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

    def _make_net(self, input_size, nb_action, hidden):
        return SacNets(input_size, nb_action, hidden)

    def _sync_np(self):
        self._np = [(l.weight.detach().cpu().numpy().T.astype(np.float32).copy(), l.bias.detach().cpu().numpy().astype(np.float32).copy())
                    for l in (self.model.actor[0], self.model.actor[2], self.model.actor[4])]

    def _q_np(self, s):                                                   # PID bias + the actor's residual logits
        (w1, b1), (w2, b2), (w3, b3) = self._np
        h = np.maximum(s @ w1 + b1, 0.0)
        h = np.maximum(h @ w2 + b2, 0.0)
        return h @ w3 + b3 + self.pid_weight * _pid_bias_np(s)

    def select_action(self, state, mask=None):
        if self.random_steps > 0:
            self.random_steps -= 1
            if mask is None:
                return random.randint(0, self.nb_action - 1)
            return random.choice([a for a in range(self.nb_action) if mask[a]])
        z = self._q_np(state)
        if mask is not None:
            z = np.where(mask, z, -1e9)
        p = np.exp(z - z.max())                                           # sample from the policy
        c = np.cumsum(p)
        return min(int(np.searchsorted(c, random.random() * c[-1])), self.nb_action - 1)

    def eval_action(self, state, mask, rng):
        z = self._q_np(state)
        if mask is not None:
            z = np.where(mask, z, -1e9)
        return int(np.argmax(z))

    def learn(self, batch_state, batch_next_state, batch_action, batch_reward, batch_discount, idxs, weights):
        dev = self.device
        s, ns = batch_state.to(dev), batch_next_state.to(dev)
        a, r, disc, w = batch_action.to(dev), batch_reward.to(dev), batch_discount.to(dev), weights.to(dev)
        rows = torch.arange(len(a), device=dev)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():                                             # soft n-step target: R + gamma^n * sum_a' pi(a') (min Q(s', a') - alpha log pi(a'))
            n_logits, _, _ = self.model(ns)
            n_logits = n_logits + self.pid_weight * _pid_bias_torch(ns)
            n_logp = F.log_softmax(n_logits, dim=-1)
            _, t1, t2 = self.target_model(ns)
            v_next = (n_logp.exp() * (torch.min(t1, t2) - alpha * n_logp)).sum(dim=-1)
            y = r + disc * v_next
        logits, q1, q2 = self.model(s)
        logits = logits + self.pid_weight * _pid_bias_torch(s)
        q1a, q2a = q1[rows, a], q2[rows, a]
        critic_loss = (w * (F.smooth_l1_loss(q1a, y, reduction='none') + F.smooth_l1_loss(q2a, y, reduction='none'))).mean()
        logp = F.log_softmax(logits, dim=-1)
        prob = logp.exp()
        actor_loss = (prob * (alpha * logp - torch.min(q1, q2).detach())).sum(dim=-1).mean()      # the critics are not changed by the actor loss
        self.optimizer.zero_grad()
        (critic_loss + actor_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        entropy = -(prob.detach() * logp.detach()).sum(dim=-1)            # keep the policy's entropy near the target: alpha goes up when it is too low
        alpha_loss = -(self.log_alpha * (self.target_entropy - entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():                                             # the target critics follow the critics slowly
            for p, tp in zip(self.model.parameters(), self.target_model.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * p)
        self._sync_np()
        self.memory.update_priorities(idxs, (0.5 * ((q1a - y).abs() + (q2a - y).abs())).detach().cpu().numpy())
        self.steps += 1
        self.last_entropy = float(entropy.mean())
        self.last_alpha = float(alpha)


# ════════ training loop ════════

# STANDALONE-BODY-START

TRAIN_MIX_V3 = {'calm': 0.10, 'breeze': 0.20, 'windy': 0.30, 'gale': 0.25, 'gale_dusty': 0.15}


class DqnFamilyPolicy:
    """A trained brain of this file, ready to fly (used by the evaluation)."""

    def __init__(self, brain, stack, shield):
        self.brain, self.shield = brain, shield
        self.stack = StateStack(stack) if stack > 1 else None

    def act(self, env, rng, explore):
        state = env.get_state()
        x = self.stack.push(state) if self.stack else np.asarray(state, dtype=np.float32)
        mask = shield_mask(env) if self.shield else None
        if explore:
            return rng.choice([a for a in range(N_ACTIONS) if mask is None or mask[a]])
        return self.brain.eval_action(x, mask, rng)

    def on_respawn(self):
        if self.stack:
            self.stack.reset()


def main(argv=None):
    ap = argparse.ArgumentParser(description='Drone trainer v3 (dqn / qr / sac)')
    ap.add_argument('--agent', default='dqn', choices=['dqn', 'qr', 'sac'])
    ap.add_argument('--minutes', type=float, default=60.0)
    ap.add_argument('--wind-aware', dest='wind_aware', action='store_true')
    ap.add_argument('--no-wind-aware', dest='wind_aware', action='store_false')
    ap.set_defaults(wind_aware=True)
    ap.add_argument('--shield', action='store_true', help='safety filter: actions that would fly into an obstacle are not allowed')
    ap.add_argument('--wind-comp', type=float, default=0.0, help='the flight controller cancels this share of the wind push (0 = none, 0.7 = 70%%)')
    ap.add_argument('--stack', type=int, default=1, help='the brain sees the last N states (1 = only the current one)')
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--n-step', type=int, default=3)
    ap.add_argument('--train-every', type=int, default=2)
    ap.add_argument('--memory', type=int, default=500_000)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--lr-final-frac', type=float, default=0.1)
    ap.add_argument('--n-quant', type=int, default=32, help='qr: quantiles per action')
    ap.add_argument('--cvar', type=float, default=0.5, help='qr: the drone chooses by the mean of the worst this share of outcomes (1 = plain mean)')
    ap.add_argument('--tau', type=float, default=0.005, help='sac: how fast the target critics follow the critics')
    ap.add_argument('--entropy-frac', type=float, default=0.5, help='sac: the policy keeps this share of the maximum entropy')
    ap.add_argument('--potential-weight', type=float, default=1.5, help='sac: weight of the fixed potential-field bias added to the actor logits (0 = plain SAC)')
    ap.add_argument('--curriculum-frac', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--world-seed', type=int, default=4321)
    ap.add_argument('--switch-every', type=int, default=1500)
    ap.add_argument('--max-steps', type=int, default=None)
    ap.add_argument('--no-replay', action='store_true')
    ap.add_argument('--replay-every', type=int, default=1)
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    ap.add_argument('--eval-seeds', type=int, default=6)
    ap.add_argument('--eval-steps', type=int, default=6000)
    ap.add_argument('--no-eval', action='store_true')
    ap.add_argument('--out', default=None)
    ap.set_defaults(**globals().get('RUN_DEFAULTS', {}))
    args = ap.parse_known_args(argv)[0]

    device = 'cuda' if (args.device == 'auto' and torch.cuda.is_available()) else ('cpu' if args.device == 'auto' else args.device)
    out = Path(args.out) if args.out else (Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path('drone_edge_v3_out'))
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == 'cpu':
        torch.set_num_threads(1)

    def make_env(world_seed, weather, rng_seed=None):
        return EdgeDroneEnvV3(seed=world_seed, rng_seed=world_seed if rng_seed is None else rng_seed, weather=weather,
                              wind_aware=args.wind_aware, wind_comp=args.wind_comp)
    env = make_env(args.world_seed, 'calm', args.seed)
    stack = StateStack(args.stack) if args.stack > 1 else None
    n_in = len(env.get_state()) * max(1, args.stack)
    common = dict(train_every=args.train_every, device=device, hidden=args.hidden, n_step=args.n_step, memory=args.memory, lr=args.lr)
    if args.agent == 'qr':
        brain = QrDqn(n_in, N_ACTIONS, 0.99, n_quant=args.n_quant, cvar=args.cvar, **common)
    elif args.agent == 'sac':
        brain = SacDiscrete(n_in, N_ACTIONS, 0.99, tau=args.tau, entropy_frac=args.entropy_frac, potential_weight=args.potential_weight, **common)
    else:
        brain = Dqn(n_in, N_ACTIONS, 0.99, **common)

    def save(path):
        save_checkpoint(path, {'state_dict': brain.model.state_dict(), 'agent': args.agent, 'hidden': args.hidden, 'n_actions': N_ACTIONS,
                               'inputs': n_in, 'stack': args.stack, 'shield': args.shield, 'wind_comp': args.wind_comp, 'steps': brain.steps})

    def run_eval():
        seeds = [1000 + i for i in range(args.eval_seeds)]
        print(f'evaluation: {len(seeds)} world seeds x {args.eval_steps} steps per condition (the same skies as evaluate_edge.py)', flush=True)
        results = {}
        print('the system as configured:', flush=True)
        results['conditions'] = evaluate_generic(lambda ws, c: make_env(ws, c), lambda: DqnFamilyPolicy(brain, args.stack, args.shield),
                                                 list(CONDITIONS), seeds, args.eval_steps)
        if args.shield:
            print('the same brain without the safety filter:', flush=True)
            results['brain_only_conditions'] = evaluate_generic(lambda ws, c: make_env(ws, c), lambda: DqnFamilyPolicy(brain, args.stack, False),
                                                                list(CONDITIONS), seeds, args.eval_steps)
        if args.wind_comp > 0:
            print('the same brain with the raw wind (no wind compensation):', flush=True)
            results['raw_wind_conditions'] = evaluate_generic(
                lambda ws, c: EdgeDroneEnvV3(seed=ws, rng_seed=ws, weather=c, wind_aware=args.wind_aware, wind_comp=0.0),
                lambda: DqnFamilyPolicy(brain, args.stack, args.shield), list(CONDITIONS), seeds, args.eval_steps)
        (out / 'eval_v3.json').write_text(json.dumps(dict(name=f'{args.agent}', agent=args.agent, inputs=n_in, actions=N_ACTIONS, seeds=seeds,
                                                          steps=args.eval_steps, shield=args.shield, wind_comp=args.wind_comp, **results), indent=2))

    replay = ReplayWriter(out / 'replay.jsonl', {'t': 'meta', 'kind': 'drone_edge', 'version': 2, 'world_seed': args.world_seed, 'wind_aware': args.wind_aware,
                                                 'weathers': WEATHER_NAMES, 'actions': N_ACTIONS}, args.replay_every, not args.no_replay)
    log = TrainLog(out / 'train_log.csv', ['step', 'time_s', 'condition', 'goals', 'collisions', 'respawns', 'score', 'gradient_steps', 'lr', 'steps_per_s'])
    stop = {'now': False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    started, total_s = time.time(), args.minutes * 60
    win = dict(goals=0, coll=0, resp=0)
    totals = {c: dict(steps=0, goals=0, coll=0) for c in CONDITIONS}
    conds = list(TRAIN_MIX_V3)
    step, last_reward, last_print, last_block, last_save = 0, 0.0, started, started, started
    print(f'device={device} agent={args.agent} shield={args.shield} wind_comp={args.wind_comp} stack={args.stack} inputs={n_in} actions={N_ACTIONS} '
          f'hidden={args.hidden} n_step={args.n_step} train_every={args.train_every} wind_aware={args.wind_aware} minutes={args.minutes} out={out}', flush=True)
    try:
        while not stop['now']:
            if args.max_steps and step >= args.max_steps:
                break
            if step % 512 == 0 and time.time() - started >= total_s:
                break
            progress = (time.time() - started) / total_s
            if step % args.switch_every == 0:
                env.set_weather(str(np.random.choice(conds, p=weather_probs_generic(TRAIN_MIX_V3, progress, True, args.curriculum_frac))))
            if step % 1000 == 0:
                brain.set_lr(args.lr * (1.0 - (1.0 - args.lr_final_frac) * min(1.0, progress)))
            g0 = env.goals_reached
            state = env.get_state()
            x = stack.push(state) if stack else np.asarray(state, dtype=np.float32)
            mask = shield_mask(env) if args.shield else None
            action = brain.update(last_reward, x, mask)
            respawned = env.step(action)
            step += 1
            last_reward = env.reward
            t = totals[env.weather_name]
            t['steps'] += 1
            t['coll'] += int(env.is_colliding)
            win['coll'] += int(env.is_colliding)
            if env.goals_reached > g0:
                win['goals'] += 1
                t['goals'] += 1
            if respawned:
                win['resp'] += 1
                brain.random_steps = RANDOM_STEPS_AFTER_RESPAWN
                brain.flush_nstep()
                if stack:
                    stack.reset()
            replay.add(step, env, action, brain.score(), brain.steps)
            if step % 1000 == 0:
                now = time.time()
                log.row(step, round(now - started, 1), env.weather_name, win['goals'], win['coll'], win['resp'], round(brain.score(), 4), brain.steps,
                        f"{brain.optimizer.param_groups[0]['lr']:.2e}", round(1000 / max(now - last_block, 1e-9)))
                win.update(goals=0, coll=0, resp=0)
                last_block = now
            now = time.time()
            if now - last_save >= 180:
                save(out / 'brain_autosave.pth')
                replay.flush()
                last_save = now
            if now - last_print >= 60:
                last_print = now
                el = now - started
                print(f'[{int(el // 60):3d} min] step {step:>9,} {step / el:6.0f}/s  score {brain.score():+.3f}  cond {env.weather_name}  gradient steps {brain.steps:,}', flush=True)
    finally:
        replay.close()
        log.close()
        save(out / 'brain_final.pth')
        summary = dict(agent=args.agent, shield=args.shield, wind_comp=args.wind_comp, stack=args.stack, wind_aware=args.wind_aware, steps=step,
                       minutes=round((time.time() - started) / 60, 1), device=device, gradient_steps=brain.steps,
                       per_condition={c: dict(steps=v['steps'], goals_per_1k=round(1000 * v['goals'] / max(v['steps'], 1), 2),
                                              collisions_per_1k=round(1000 * v['coll'] / max(v['steps'], 1), 1)) for c, v in totals.items()})
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    if not args.no_eval:
        run_eval()
    print('files in', out, ':', ', '.join(sorted(p.name for p in out.iterdir())))




if __name__ == "__main__":
    main()
