"""
Tune the gains of the real PID controller (pid_controller.py, 12 numbers) with the same separable CMA-ES as evolve_prior.py, flying it alone among the roaming flyers.

    python evolve_pid.py                       # 25 generations x population 12, about 15 min on 4 cores
    python evolve_pid.py --compare             # no search: fly the default gains, the tuned gains (results/tuned_pid.json) and the prior of prior.py on the same new skies

Score of a flight = goals per 1000 steps - 2 * collisions per 1000 steps, as in evolve_prior.py. Writes results/tuned_pid.json.
"""
import argparse
import json
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evolve_prior as EV                                                           # noqa: E402  (also puts the torch-free world on the path and installs flyers + cues)
import pid_controller as PC                                                         # noqa: E402
import prior as P                                                                   # noqa: E402
from edge_residual_adaptive.core import EdgeDroneEnvV3                              # noqa: E402

RANDOM_STEPS_AFTER_RESPAWN = EV.RANDOM_STEPS_AFTER_RESPAWN


def fly_pid(params, world_seed, weather, steps, moving=True):
    env = EdgeDroneEnvV3(seed=world_seed, rng_seed=world_seed, weather=weather, wind_aware=True, wind_comp=0.0)
    env.flyers_on, env.motion_cues = moving, False
    env._update_sensors(); env._update_nav()
    ctl, rng = PC.PidController(params), random.Random(world_seed)
    goals0, hits, resp, rand_left = env.goals_reached, 0, 0, 0
    for _ in range(steps):
        if rand_left > 0:
            a = rng.randrange(9); rand_left -= 1
        else:
            a = ctl.act(env.get_state())
        before = env.goals_reached
        if env.step(a):
            resp += 1; rand_left = RANDOM_STEPS_AFTER_RESPAWN; ctl.reset()
        elif env.goals_reached > before:
            ctl.reset()                                                              # a new goal: the old error history means nothing
        hits += int(env.is_colliding)
    return env.goals_reached - goals0, hits, resp


def _job(a):
    kind, th, ws, w, steps, moving = a
    return fly_pid(th, ws, w, steps, moving) if kind == 'pid' else EV.fly(th, ws, w, steps, moving, True)


def to_params(u):
    return PC.LOW + np.clip(u, 0, 1) * (PC.HIGH - PC.LOW)


def to_unit(p):
    return np.clip((np.asarray(p) - PC.LOW) / (PC.HIGH - PC.LOW), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--generations', type=int, default=25)
    ap.add_argument('--pop', type=int, default=12)
    ap.add_argument('--episodes', type=int, default=4)
    ap.add_argument('--steps', type=int, default=1200)
    ap.add_argument('--sigma', type=float, default=0.2)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--holdout-seeds', type=int, default=6)
    ap.add_argument('--holdout-steps', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--compare', action='store_true')
    ap.add_argument('--out', default=str(HERE / 'results'))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    hold = [(7000 + i, EV.WEATHERS[i % 5]) for i in range(args.holdout_seeds)]
    tuned_path = out / 'tuned_pid.json'
    with Pool(args.workers) as pool:
        best = PC.DEFAULT.copy()
        if not args.compare:
            es = EV.SepCMAES(to_unit(PC.DEFAULT), args.sigma, args.pop, args.seed)
            rng, t0 = random.Random(args.seed), time.time()
            for g in range(args.generations):
                eps = [(rng.randrange(1, 10 ** 6), EV.WEATHERS[rng.randrange(5)]) for _ in range(args.episodes)]
                xs = es.ask()
                xs[0] = to_unit(PC.DEFAULT) if g == 0 else xs[0]
                res = pool.map(_job, [('pid', to_params(x), ws, w, args.steps, True) for x in xs for ws, w in eps], chunksize=1)
                fit = [EV.score_of(res[i * len(eps):(i + 1) * len(eps)], args.steps)[0] for i in range(len(xs))]
                es.tell(xs, fit)
                b = int(np.argmax(fit)); best = to_params(xs[b])
                print(f'gen {g:2d}  best {fit[b]:7.2f}  mean {np.mean(fit):8.2f}  sigma {es.sigma:.3f}  {time.time() - t0:4.0f} s', flush=True)
            cands = {'default': PC.DEFAULT, 'last_best': best, 'final_mean': to_params(es.mean)}
        else:
            tuned = json.loads(tuned_path.read_text())['params']
            cands = {'default': PC.DEFAULT, 'tuned': np.array(tuned)}
        table = {}
        for name, th in cands.items():
            for moving in (True, False):
                r = pool.map(_job, [('pid', th, ws, w, args.holdout_steps, moving) for ws, w in hold], chunksize=1)
                s, gk, ck = EV.score_of(r, args.holdout_steps)
                table[f'pid {name}|{"moving" if moving else "static"}'] = dict(score=round(s, 2), goals_per_1k=round(gk, 2), collisions_per_1k=round(ck, 2))
        ev = json.loads((out / 'evolved_prior.json').read_text())
        for name, th in (('classic prior', P.CLASSIC), ('evolved prior', np.array(ev['theta']))):
            r = pool.map(_job, [('prior', th, ws, w, args.holdout_steps, True) for ws, w in hold], chunksize=1)
            s, gk, ck = EV.score_of(r, args.holdout_steps)
            table[f'{name}|moving'] = dict(score=round(s, 2), goals_per_1k=round(gk, 2), collisions_per_1k=round(ck, 2))
    for k, v in table.items():
        print(f'held out  {k:26s} {v}')
    if args.compare:
        (out / 'pid_compare.json').write_text(json.dumps(table, indent=2))
        return
    pick = max(['default', 'last_best', 'final_mean'], key=lambda n: table[f'pid {n}|moving']['score'])
    (out / 'tuned_pid.json').write_text(json.dumps(dict(params=[round(float(v), 5) for v in cands[pick]], names=PC.NAMES, picked=pick, held_out=table), indent=2))
    print('picked', pick)


if __name__ == '__main__':
    main()
