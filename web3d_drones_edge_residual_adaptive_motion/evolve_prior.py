"""
Evolve the hand-made prior (prior.py, 12 numbers) with a separable CMA-ES, flying a prior-only policy (no network) among the roaming birds and patrol drones.

    python evolve_prior.py                          # about 25 generations, population 16, takes ~20 min on 4 cores
    python evolve_prior.py --generations 3 --pop 8 --steps 600 --workers 2 --out /tmp/quick     # a quick try

Why this is cheap: the prior is a plain function of the state, so a candidate is scored by FLYING it in the numpy world of the ROS package (no torch, no GPU). Each candidate flies the
same few (world seed, weather) episodes in one generation (common random numbers: a fair comparison inside a generation) and the episodes change from generation to generation
(so the winner cannot memorise one sky). The score of a flight is   goals per 1000 steps - COLLISION_WEIGHT * collisions per 1000 steps.

Writes results/evolved_prior.json (the best numbers, the classic numbers, a held-out comparison on new skies) and results/evolve_log.csv.
build_trainer.py then embeds the numbers as prior.EVOLVED in the Kaggle trainer, and the server loads the json.
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / 'ros2_edge_residual_adaptive' / 'edge_residual_adaptive'          # the torch-free copy of the world (generated from the base trainer)
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(HERE))
import prior as P                                                                  # noqa: E402
import motion_cues as MC                                                           # noqa: E402
from edge_residual_adaptive import flyers                                          # noqa: E402
from edge_residual_adaptive.core import EdgeDroneEnvV3                             # noqa: E402

COLLISION_WEIGHT = 2.0                  # one collision per 1000 steps costs as much as two goals per 1000 steps
RANDOM_STEPS_AFTER_RESPAWN = 90         # the same recovery as the trainer and the server
WEATHERS = ['calm', 'breeze', 'windy', 'gale', 'gale_dusty']

flyers.install(EdgeDroneEnvV3)
MC.install_motion_cues(EdgeDroneEnvV3)


def fly(theta, world_seed, weather, steps, moving=True, motion=True):
    """one flight of the prior alone (greedy: the highest logit). Returns (goals, collisions, respawns)."""
    env = EdgeDroneEnvV3(seed=world_seed, rng_seed=world_seed, weather=weather, wind_aware=True, wind_comp=0.0)
    env.flyers_on, env.motion_cues = moving, motion
    env._update_sensors(); env._update_nav()
    rng = random.Random(world_seed)
    goals0, hits, resp, rand_left = env.goals_reached, 0, 0, 0
    for _ in range(steps):
        if rand_left > 0:
            a = rng.randrange(P.N_ACTIONS)
            rand_left -= 1
        else:
            a = int(np.argmax(P.prior_logits_one(env.get_state(), theta)))
        if env.step(a):
            resp += 1
            rand_left = RANDOM_STEPS_AFTER_RESPAWN
        hits += int(env.is_colliding)
    return env.goals_reached - goals0, hits, resp


def _job(args):
    theta, ws, weather, steps, moving, motion = args
    return fly(theta, ws, weather, steps, moving, motion)


def score_of(rows, steps_each):
    n = len(rows) * steps_each
    goals, hits = sum(r[0] for r in rows), sum(r[1] for r in rows)
    return 1000.0 * goals / n - COLLISION_WEIGHT * 1000.0 * hits / n, 1000.0 * goals / n, 1000.0 * hits / n


class SepCMAES:
    """separable CMA-ES (Ros & Hansen 2008): CMA-ES with a diagonal covariance, linear in the number of parameters. Works in the unit cube; the caller maps to THETA."""

    def __init__(self, x0, sigma, pop, seed):
        n = self.n = len(x0)
        self.rng = np.random.default_rng(seed)
        self.mean, self.sigma, self.lam = np.array(x0, dtype=float), sigma, pop
        mu = self.mu = pop // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        c1 = 2.0 / ((n + 1.3) ** 2 + self.mueff)
        cmu = min(1 - c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.c1, self.cmu = c1 * (n + 2) / 3, cmu * (n + 2) / 3                                 # the "sep" speed-up of the learning rates
        self.damps = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.pc, self.ps, self.C = np.zeros(n), np.zeros(n), np.ones(n)
        self.chi = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.gen = 0

    def ask(self):
        self.z = self.rng.standard_normal((self.lam, self.n))
        self.y = self.z * np.sqrt(self.C)
        return np.clip(self.mean + self.sigma * self.y, 0.0, 1.0)

    def tell(self, xs, fitness):
        order = np.argsort(-np.asarray(fitness))[:self.mu]                                       # the best mu (we maximise)
        xs = np.asarray(xs)
        y = (xs[order] - self.mean) / self.sigma                                                 # steps AFTER the clipping to the cube, so the search never believes in a point outside it
        yw = self.w @ y
        self.mean = self.mean + self.sigma * yw
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * yw / np.sqrt(self.C)
        self.gen += 1
        hs = float(np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * self.gen)) / self.chi < 1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + hs * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (self.pc ** 2 + (1 - hs) * self.cc * (2 - self.cc) * self.C) + self.cmu * (self.w @ (y ** 2))
        self.C = np.maximum(self.C, 1e-8)
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chi - 1))
        self.sigma = float(np.clip(self.sigma, 1e-3, 0.6))


def to_theta(u):
    return P.LOW + np.clip(u, 0.0, 1.0) * (P.HIGH - P.LOW)


def to_unit(theta):
    return np.clip((np.asarray(theta) - P.LOW) / (P.HIGH - P.LOW), 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--generations', type=int, default=25)
    ap.add_argument('--pop', type=int, default=16)
    ap.add_argument('--episodes', type=int, default=4, help='episodes per candidate per generation')
    ap.add_argument('--steps', type=int, default=1500, help='steps per episode')
    ap.add_argument('--sigma', type=float, default=0.25)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--holdout-seeds', type=int, default=6)
    ap.add_argument('--holdout-steps', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--out', default=str(HERE / 'results'))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    es = SepCMAES(to_unit(P.CLASSIC), args.sigma, args.pop, args.seed)
    rng = random.Random(args.seed)
    log = open(out / 'evolve_log.csv', 'w', newline='')
    wr = csv.writer(log)
    wr.writerow(['generation', 'best', 'mean', 'sigma', 'best_goals_per_1k', 'best_collisions_per_1k', 'time_s'] + P.NAMES)
    best_theta, best_score = P.CLASSIC.copy(), -1e9
    t0 = time.time()
    with Pool(args.workers) as pool:
        for g in range(args.generations):
            eps = [(rng.randrange(1, 10 ** 6), WEATHERS[rng.randrange(len(WEATHERS))]) for _ in range(args.episodes)]
            xs = es.ask()
            if g == 0:
                xs[0] = to_unit(P.CLASSIC)                                                    # the classic prior is in the first population: evolution has to beat it
            jobs = [(to_theta(x), ws, w, args.steps, True, True) for x in xs for ws, w in eps]
            res = pool.map(_job, jobs, chunksize=1)
            fit, gp, cp = [], [], []
            for i in range(len(xs)):
                s, gk, ck = score_of(res[i * len(eps):(i + 1) * len(eps)], args.steps)
                fit.append(s); gp.append(gk); cp.append(ck)
            es.tell(xs, fit)
            b = int(np.argmax(fit))
            best_theta, best_score = to_theta(xs[b]), fit[b]                                      # the best of THIS generation (the skies change, so scores are not comparable across generations)
            wr.writerow([g, round(fit[b], 2), round(float(np.mean(fit)), 2), round(es.sigma, 3), round(gp[b], 2), round(cp[b], 2), round(time.time() - t0)] + [round(float(v), 4) for v in best_theta])
            log.flush()
            print(f'gen {g:2d}  best {fit[b]:7.2f} (goals {gp[b]:5.2f}/1k, hits {cp[b]:5.1f}/1k)  mean {np.mean(fit):7.2f}  sigma {es.sigma:.3f}  {time.time() - t0:5.0f} s', flush=True)
        final = to_theta(es.mean)                                                             # the mean of the search is smoother than any one lucky sample
        # held-out comparison on skies the evolution never saw: the classic prior, the best sample of the last generation, and the final mean
        hold = [(5000 + i, WEATHERS[i % len(WEATHERS)]) for i in range(args.holdout_seeds)]
        cands = {'classic': P.CLASSIC, 'last_best': best_theta, 'final_mean': final}
        table = {}
        for name, th in cands.items():
            for moving in (True, False):
                r = pool.map(_job, [(th, ws, w, args.holdout_steps, moving, True) for ws, w in hold], chunksize=1)
                s, gk, ck = score_of(r, args.holdout_steps)
                table[f'{name}|{"moving" if moving else "static"}'] = dict(score=round(s, 2), goals_per_1k=round(gk, 2), collisions_per_1k=round(ck, 2))
                print(f'held out  {name:10s} {"moving" if moving else "static":6s}  score {s:7.2f}  goals {gk:5.2f}/1k  hits {ck:6.1f}/1k', flush=True)
    pick = max(['classic', 'last_best', 'final_mean'], key=lambda n: table[f'{n}|moving']['score'])      # the held-out winner (the classic prior wins if evolution did not help)
    theta = cands[pick]
    (out / 'evolved_prior.json').write_text(json.dumps(dict(
        theta=[round(float(v), 5) for v in theta], names=P.NAMES, picked=pick, classic=[float(v) for v in P.CLASSIC], held_out=table,
        settings=dict(generations=args.generations, pop=args.pop, episodes=args.episodes, steps=args.steps, sigma=args.sigma, holdout_seeds=args.holdout_seeds,
                      holdout_steps=args.holdout_steps, collision_weight=COLLISION_WEIGHT, seed=args.seed), minutes=round((time.time() - t0) / 60, 1)), indent=2))
    print('picked', pick, [round(float(v), 4) for v in theta])


if __name__ == '__main__':
    main()
