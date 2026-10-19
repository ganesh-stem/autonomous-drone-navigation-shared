"""
Does the trained brain cope with obstacles that MOVE?  Flies the same brain (the numpy actor of weights/residual_adaptive_actor.npz, no torch, no ROS) in the same
skies twice, once with static floating blocks and once with the moving ones of flyers.py, and prints goals and collisions per 1000 steps.

    python tools/eval_moving_obstacles.py                      # 5 seeds x 3 weathers x 4000 steps, takes a few minutes
    python tools/eval_moving_obstacles.py --steps 2000 --seeds 3

The brain was trained on a static world, so some loss is expected; the numbers say how much. Results go to results/moving_obstacles_eval.json.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / 'edge_residual_adaptive'
sys.path.insert(0, str(PKG))
from edge_residual_adaptive import flyers                                    # noqa: E402
from edge_residual_adaptive.core import EdgeDroneEnvV3                       # noqa: E402
from edge_residual_adaptive.residual_policy import ResidualAdaptiveActor     # noqa: E402

RANDOM_STEPS_AFTER_RESPAWN = 90                                              # the same recovery as brain_node.py and the viewer server


def fly(actor, seed, weather, steps, moving):
    flyers.install(EdgeDroneEnvV3)
    env = EdgeDroneEnvV3(seed=seed, rng_seed=5, weather=weather, wind_aware=True, wind_comp=0.0)
    env.flyers_on = moving
    env._update_sensors(); env._update_nav()
    rng, np_rng = random.Random(7), np.random.default_rng(7)
    goals0, hits, respawns, random_left = env.goals_reached, 0, 0, 0
    for _ in range(steps):
        a = rng.randrange(actor.n_actions) if random_left > 0 else actor.act(np.asarray(env.get_state(), dtype=np.float32), np_rng, True)
        random_left = max(0, random_left - 1)
        if env.step(a):
            respawns += 1
            random_left = RANDOM_STEPS_AFTER_RESPAWN
        hits += int(env.is_colliding)
    return dict(goals=env.goals_reached - goals0, collisions=hits, respawns=respawns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--weathers', default='calm,windy,gale')
    ap.add_argument('--weights', default=str(PKG / 'weights' / 'residual_adaptive_actor.npz'))
    ap.add_argument('--out', default=str(ROOT / 'results' / 'moving_obstacles_eval.json'))
    args = ap.parse_args()
    actor = ResidualAdaptiveActor(args.weights)
    weathers = args.weathers.split(',')
    rows, t0 = [], time.time()
    for moving in (False, True):
        for w in weathers:
            for s in range(args.seeds):
                r = fly(actor, 1234 + s, w, args.steps, moving)
                r.update(moving=moving, weather=w, seed=1234 + s, steps=args.steps)
                rows.append(r)
                print(f"{'moving' if moving else 'static'} {w:7s} seed {1234 + s}: goals {r['goals']:3d}  collisions {r['collisions']:4d}  respawns {r['respawns']}", flush=True)
    print()
    summary = {}
    for moving in (False, True):
        sel = [r for r in rows if r['moving'] == moving]
        tot = sum(r['steps'] for r in sel)
        summary['moving' if moving else 'static'] = dict(
            goals_per_1k=round(1000 * sum(r['goals'] for r in sel) / tot, 2),
            collisions_per_1k=round(1000 * sum(r['collisions'] for r in sel) / tot, 2),
            respawns_per_1k=round(1000 * sum(r['respawns'] for r in sel) / tot, 2), runs=len(sel), steps_each=args.steps)
        print(f"{'moving' if moving else 'static':7s}: {summary['moving' if moving else 'static']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))
    print(f'wrote {args.out} ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
