"""
Headless test bench for the trained residual-SAC brain: no Unreal, no browser, no GPU.

It flies the brain through the SAME finite 4 km x 4 km obstacle world that was built in Unreal (chunks -8..7 of
DroneEnv(seed=1234), see unreal_env_mapping.md), using the trainer's own environment code so the physics, sensors
and wind are exactly what the brain was trained on.

    python headless_sim.py                                   # 60 000 steps per weather, prints a table
    python headless_sim.py --weather gale --steps 20000
    python headless_sim.py --brain logs/brain_final.pth --out flight.jsonl    # also record the flight

Plug in another model: pass any callable  policy(state: np.ndarray) -> int  to run_episode() / evaluate().
The trajectory file has one JSON line per step: [step, x, y, z, yaw, action, hit, goal_x, goal_y, goal_z, goals],
in sim units (see unreal_env_mapping.md for the Unreal mapping) -- ready for a replay actor in Unreal.
"""

import argparse
import contextlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('edge_trainer', HERE / 'edge_v3_9_residual_pid_sac.py')
E = importlib.util.module_from_spec(_spec)
sys.modules['edge_trainer'] = E
_spec.loader.exec_module(E)                     # only defines things; main() runs under __main__

WORLD_SEED = 1234
ARENA_CHUNKS = 8                                # chunks -8..7 on both axes = exactly what was built in Unreal
ARENA = ARENA_CHUNKS * E.CHUNK                  # 2000 m half-extent
DEFAULT_BRAIN = HERE / 'results' / 'models' / 'sac_residual' / 'brain_final.pth'


class ArenaEnv(E.EdgeDroneEnvV2):
    """The trainer's environment, but the obstacle field ends where the Unreal level ends, and goals stay inside it."""

    def _chunk(self, cx, cy):
        if not (-ARENA_CHUNKS <= cx < ARENA_CHUNKS and -ARENA_CHUNKS <= cy < ARENA_CHUNKS):
            return np.zeros((0, 6))
        return super()._chunk(cx, cy)

    def new_goal(self, min_dist=E.GOAL_DIST[0], max_dist=E.GOAL_DIST[1]):
        for _ in range(50):
            super().new_goal(min_dist, max_dist)
            if abs(self.goal[0]) < ARENA - 100 and abs(self.goal[1]) < ARENA - 100:
                return
        self.goal = np.array([0.0, 0.0, 80.0])          # fall back to the centre (goal-free of obstacles by construction)

    def respawn(self):
        super().respawn()
        if abs(self.x) > ARENA - 100 or abs(self.y) > ARENA - 100:
            self.reset(0.0, 0.0, 80.0, self.rng.uniform(0, 360))


class ResidualSacPolicy:
    """Greedy actor of a residual-SAC checkpoint: PID bias + the network's own logits (numpy only at inference)."""

    def __init__(self, path):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        sd = ck['state_dict'] if 'state_dict' in ck else ck
        self.w = [(sd[f'actor.{i}.weight'].numpy().T.astype(np.float32).copy(), sd[f'actor.{i}.bias'].numpy().astype(np.float32).copy())
                  for i in (0, 2, 4)]
        self.n_inputs = self.w[0][0].shape[0]
        self.pid_weight = float(ck.get('pid_weight', E.RUN_DEFAULTS['pid_weight']))

    def __call__(self, s):
        s = np.asarray(s, dtype=np.float32)
        (w1, b1), (w2, b2), (w3, b3) = self.w
        h = np.maximum(s @ w1 + b1, 0.0)
        h = np.maximum(h @ w2 + b2, 0.0)
        return int(np.argmax(h @ w3 + b3 + self.pid_weight * E._pid_bias_np(s)))


def run_episode(policy, weather='windy', steps=20000, seed=WORLD_SEED, rng_seed=5, record=None):
    env = ArenaEnv(seed=seed, rng_seed=rng_seed, weather=weather, wind_aware=True)
    rng = random.Random(rng_seed)
    goals0, coll, resp, random_steps = env.goals_reached, 0, 0, 0
    t0 = time.time()
    for t in range(steps):
        a = rng.randrange(E.N_ACTIONS) if random_steps > 0 else policy(env.get_state())
        random_steps = max(0, random_steps - 1)
        if env.step(a):
            resp += 1
            random_steps = E.RANDOM_STEPS_AFTER_RESPAWN
        coll += int(env.is_colliding)
        if record is not None:
            record.write(json.dumps([t, round(env.x, 2), round(env.y, 2), round(env.z, 2), round(env.yaw, 1), a, int(env.is_colliding),
                                     round(float(env.goal[0]), 1), round(float(env.goal[1]), 1), round(float(env.goal[2]), 1), env.goals_reached]) + '\n')
    goals = env.goals_reached - goals0
    return dict(weather=weather, steps=steps, goals=goals, collisions=coll, respawns=resp,
                goals_per_1k=1000.0 * goals / steps, collisions_per_1k=1000.0 * coll / steps, steps_per_s=steps / (time.time() - t0))


def evaluate(policy, weathers=None, steps=20000, seed=WORLD_SEED, record=None):
    rows = []
    for w in weathers or list(E.CONDITIONS):
        r = run_episode(policy, w, steps, seed, record=record)
        rows.append(r)
        print(f"  {w:11s} goals/1k {r['goals_per_1k']:6.2f}   collisions/1k {r['collisions_per_1k']:7.1f}   respawns {r['respawns']:3d}   "
              f"{r['steps_per_s']:6.0f} steps/s", flush=True)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--brain', default=str(DEFAULT_BRAIN))
    ap.add_argument('--weather', choices=list(E.CONDITIONS) + ['all'], default='all')
    ap.add_argument('--steps', type=int, default=20000, help='steps per weather condition')
    ap.add_argument('--seed', type=int, default=WORLD_SEED, help='world seed (1234 = the world built in Unreal)')
    ap.add_argument('--out', help='write the flight to this .jsonl file (one weather at a time is easiest to replay)')
    args = ap.parse_args()
    pol = ResidualSacPolicy(args.brain)
    print(f'brain {args.brain}  ({pol.n_inputs} inputs)   world seed {args.seed}   arena +-{ARENA:.0f} m')
    with (open(args.out, 'w') if args.out else contextlib.nullcontext()) as rec:
        evaluate(pol, None if args.weather == 'all' else [args.weather], args.steps, args.seed, record=rec)
