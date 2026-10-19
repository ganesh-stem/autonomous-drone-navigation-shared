"""
Generates the standalone Kaggle trainer  edge_v3_16_residual_adaptive_motion_sac.py  from the base trainer of ../web3d_drones_edge_residual_adaptive.

    python build_trainer.py

The generated file is ONE file (numpy + torch only). It is the base trainer with three additions, each pasted from a module of this folder between marker comments so that
tests/selftest_motion.py can check the copies are identical:
  1. flyers.py (already in the base trainer; here with the flyer_speed factor that the curriculum ramps);
  2. motion_cues.py : 28 ray-change inputs for the brain + the approach penalty (idea 1, and the penalty half of idea 2);
  3. prior.py       : the 12-number prior (idea 3) whose EVOLVED numbers come from results/evolved_prior.json, added to the actor's logits in place of the old fixed PID bias.
and the training loop ramps the flyers' speed (and, as in the base, the gale share of the weather) from easy to full over the run (the curriculum half of idea 2).
Re-run it after the evolution (evolve_prior.py) to embed the new numbers. Fails loudly if a piece of the base trainer it patches is not found exactly once.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'web3d_drones_edge_residual_adaptive' / 'edge_v3_12_residual_adaptive_sac.py'
OUT = HERE / 'edge_v3_16_residual_adaptive_motion_sac.py'
EVOLVED_JSON = HERE / 'results' / 'evolved_prior.json'

FLYERS_BEGIN = '# ── flyers.py begins here'
FLYERS_END = '# ── flyers.py ends here'


def module_body(path):
    """the code of a module without its docstring and its 'import numpy as np' (the trainer has numpy already)"""
    text = Path(path).read_text()
    text = re.sub(r'\A"""(.*?)"""\n', '', text, count=1, flags=re.S)
    text = text.replace('import numpy as np\n', '', 1)
    return text.strip('\n') + '\n'


def evolved_numbers():
    if EVOLVED_JSON.exists():
        return json.loads(EVOLVED_JSON.read_text())['theta']
    return None


def motion_block():
    return ('# ── motion_cues.py begins here ─────────────────────────────────────────────────────────────────────────\n'
            + module_body(HERE / 'motion_cues.py')
            + '# ── motion_cues.py ends here ───────────────────────────────────────────────────────────────────────────\n')


def prior_block(theta=None):
    body = module_body(HERE / 'prior.py')
    body = body.replace('N_ACTIONS = 9                                  # 0 straight, 1 left, 2 right, 3 climb, 4 descend, 5 hover-left, 6 hover-right, 7 slow, 8 reverse\n',
                        '# (N_ACTIONS = 9 is defined above: 0 straight, 1 left, 2 right, 3 climb, 4 descend, 5 hover-left, 6 hover-right, 7 slow, 8 reverse)\n', 1)
    if theta is not None:
        line = 'EVOLVED = CLASSIC.copy()'
        assert body.count(line) == 1
        body = body.replace(line, 'EVOLVED = np.array(%s, dtype=np.float64)' % json.dumps([round(float(v), 5) for v in theta]), 1)
    return ('# ── prior.py begins here ───────────────────────────────────────────────────────────────────────────────\n'
            + body
            + '# ── prior.py ends here ─────────────────────────────────────────────────────────────────────────────────\n')


def build():
    src = BASE.read_text()

    def rep(old, new, count=1):
        nonlocal src
        assert src.count(old) == count, f'expected {count} x, found {src.count(old)} x: {old[:70]!r}'
        src = src.replace(old, new)

    # 1. the docstring
    end = src.index('"""', 3) + 3
    doc = '''"""
RESIDUAL SAC WITH MOTION CUES, A MOVING-OBSTACLE CURRICULUM AND AN EVOLVED PRIOR      (edge_v3_16_residual_adaptive_motion_sac.py)

Built from edge_v3_12_residual_adaptive_sac.py (residual SAC on a heuristic baseline, adaptive blend weight, roaming birds and patrol drones as obstacles) by build_trainer.py in
web3d_drones_edge_residual_adaptive_motion/. Three ideas from study_guide/part5_appendix/A5_research_ideas.md are added; each can be switched off, so the run can be compared piece by piece:

  1. MOTION CUES (motion_cues=True). The brain of v3_12 sees only the CURRENT 28 ray readings, so it cannot tell a wall from a bird flying at it. The state gets 28 more numbers: how much each
     ray reading changed since the last step (the looming cue). 65 inputs instead of 37. The first 37 stay where they were, so the old indices (bearing 28, elevation 30) are unchanged.
  2. REWARD AND CURRICULUM FOR MOVING OBSTACLES.
       approach_penalty (default 0.5): a small penalty when an obstacle that is already near closes in faster than the drone's own flight explains (see motion_cues.py);
       flyer ramp (default on): the roaming flyers' speed starts at flyer_speed_start (0.4 of full) and grows to 1.0 by flyer_ramp_frac (0.5) of the run, while the gale share of the weather
       ramps as in v3_12 (curriculum_frac). Evaluation is always at full speed, with the penalty off.
  3. EVOLVED PRIOR (prior='evolved'). v3_12 added a fixed one-hot goal rule (scaled by how free the front is) to the actor's logits. Here that prior is a 12-number function (prior.py)
     with avoidance terms (turn away from the nearer side, slow down, climb to the freer side, brake, react to a closing obstacle), and the 12 numbers were found by a separable CMA-ES that flew
     the prior alone among the roaming flyers (evolve_prior.py; its log and a held-out comparison are in results/). prior='classic' uses the 12 numbers that reproduce v3_12's prior EXACTLY.
     The network still learns the residual on top, and the critics are unchanged.

Everything else is v3_12: the world (250-unit chunks, 28 rays, 9 actions, wind and weather), the roaming birds and patrol drones, the discrete SAC, the evaluation tables.
Brains of this file have 65 inputs: the 3D viewer of this folder (server_edge_residual_adaptive_motion.py) reads them. Its eval_v3.json has the extra table 'static_world_conditions' as in v3_12.

Standalone: one file, no project modules, needs only numpy and torch.

HOW TO RUN: paste the whole file into one Kaggle cell (GPU on) and run it, or  !python edge_v3_16_residual_adaptive_motion_sac.py
            No command-line options are needed: this run's settings are in the SETTINGS block right below.
            To compare: run it three times with  motion_cues=False, approach_penalty=0, flyer_ramp=False, prior='classic'  (= v3_12 behaviour),  and with one of the ideas switched on each.

Outputs (in /kaggle/working): replay.jsonl, train_log.csv, brain_final.pth, brain_autosave.pth, summary.json and eval_v3.json.

GENERATED by build_trainer.py -- do not edit by hand; edit the modules (motion_cues.py, prior.py, flyers.py) or the builder and run it again.
"""'''
    src = doc + src[end:]

    # 2. the settings
    rep("    pid_weight_max=1.5,\n    pid_weight_min=0.0,\n",
        "    motion_cues=True,            # idea 1: 28 ray-change inputs (65 inputs instead of 37)\n"
        "    approach_penalty=0.5,        # idea 2: penalty for obstacles that close in fast (0 = off)\n"
        "    flyer_ramp=True,             # idea 2: the roaming flyers start slow and reach full speed\n"
        "    flyer_speed_start=0.4,       #   their speed at the start of the run, as a share of full speed\n"
        "    flyer_ramp_frac=0.5,         #   the share of the run over which they speed up\n"
        "    prior='evolved',             # idea 3: 'evolved' (the 12 numbers found by evolution) or 'classic' (exactly the v3_12 prior)\n")
    rep("    # more options you can add: static_obstacles=True",
        "    # more options you can add: warm_start='/kaggle/input/.../brain_final.pth' (continue from an older, smaller brain; the new inputs start at 0), static_obstacles=True")

    # 3. the pasted modules, right before the (first) agent section
    marker = '# ════════ agent ════════\n'
    first = src.index(marker)
    theta = evolved_numbers()
    inserted = ('# ════════ motion cues and the evolved prior (idea 1, 2 and 3) ════════\n#\n'
                '# motion_cues.py and prior.py of web3d_drones_edge_residual_adaptive_motion, pasted (tests/selftest_motion.py checks that the copies are identical).\n\n'
                + motion_block() + '\ninstall_motion_cues(EdgeDroneEnvV3)                          # off until main() sets env.motion_cues / env.approach_penalty\n\n'
                + prior_block(theta) + '\nTHETA = EVOLVED.copy()                                      # the numbers the actor uses; main() sets CLASSIC if --prior classic\n\n\n')
    src = src[:first] + inserted + src[first:]

    # 4. the prior in place of the fixed one-hot rule
    a = src.index('# ── PID baseline for the residual actor')
    b = src.index('# ── SAC begins here')
    src = src[:a] + '''# ── the prior of the residual actor ─────────────────────────────────────────────────────────────────────
# v3_12 added pid_scale(s) * pid_bias(s) (a fixed goal rule, faded out near obstacles) to the actor's logits. Here prior_logits(s, THETA) (pasted above) does that, and with THETA = CLASSIC
# it is exactly the same numbers. THETA is NOT learned by gradient descent: it is either the classic values or the result of evolve_prior.py. With stack > 1 it reads the OLDEST frame of the
# stack, so keep stack=1 (the motion cues are the memory of this file).

def _prior_np(s):
    return prior_logits_one(s, THETA)                                      # (N_ACTIONS,) float32


def _prior_torch(s):
    return prior_logits(s, THETA).to(s.dtype)                              # (B, N_ACTIONS)


''' + src[b:]

    # 5. SacDiscrete uses it
    rep('''    """RESIDUAL SAC (this file's one change to plain SAC): the actor's logits are pid_weight * _pid_bias(s) + the
    network's own output, everywhere the policy is used (acting, and both the current and next-state logits in
    learn()). The network only ever has to learn the correction on top of the PID baseline, not the whole policy
    from scratch -- the same idea as residual RL on top of an existing PID controller."""''',
        '''    """RESIDUAL SAC: the actor's logits are prior_logits(s, THETA) + the network's own output, everywhere the policy is used (acting, and both the
    current and next-state logits in learn()). The network only ever has to learn the correction on top of the prior, not the whole policy from scratch."""''')
    rep('entropy_frac=0.5, pid_weight_max=1.5, pid_weight_min=0.0, **kw):', 'entropy_frac=0.5, **kw):')
    rep('        self.pid_weight_max = pid_weight_max\n        self.pid_weight_min = pid_weight_min\n', '')
    rep('    def _q_np(self, s):                                                   # adaptive-weight PID bias + the actor\'s residual logits',
        '    def _q_np(self, s):                                                   # the prior + the actor\'s residual logits')
    rep('return h @ w3 + b3 + _pid_scale_np(s, self.pid_weight_max, self.pid_weight_min) * _pid_bias_np(s)', 'return h @ w3 + b3 + _prior_np(s)')
    rep('n_logits = n_logits + _pid_scale_torch(ns, self.pid_weight_max, self.pid_weight_min) * _pid_bias_torch(ns)', 'n_logits = n_logits + _prior_torch(ns)')
    rep('logits = logits + _pid_scale_torch(s, self.pid_weight_max, self.pid_weight_min) * _pid_bias_torch(s)', 'logits = logits + _prior_torch(s)')

    # 6. main(): options, the env, the curriculum, the checkpoint
    rep("    ap.add_argument('--pid-weight-max', type=float, default=1.5, help='sac: PID bias weight far from any obstacle')\n"
        "    ap.add_argument('--pid-weight-min', type=float, default=0.0, help='sac: PID bias weight right next to an obstacle (fades toward this)')\n",
        "    ap.add_argument('--prior', default='evolved', choices=['evolved', 'classic'], help='sac: the 12-number prior added to the logits (classic = exactly the v3_12 prior)')\n"
        "    ap.add_argument('--motion-cues', dest='motion_cues', action='store_true', help='28 ray-change inputs')\n"
        "    ap.add_argument('--no-motion-cues', dest='motion_cues', action='store_false')\n"
        "    ap.set_defaults(motion_cues=True)\n"
        "    ap.add_argument('--approach-penalty', type=float, default=0.5, help='penalty for obstacles closing in fast (0 = off)')\n"
        "    ap.add_argument('--flyer-ramp', dest='flyer_ramp', action='store_true', help='the roaming flyers start slow and reach full speed')\n"
        "    ap.add_argument('--no-flyer-ramp', dest='flyer_ramp', action='store_false')\n"
        "    ap.set_defaults(flyer_ramp=True)\n"
        "    ap.add_argument('--flyer-speed-start', type=float, default=0.4)\n"
        "    ap.add_argument('--flyer-ramp-frac', type=float, default=0.5)\n"
        "    ap.add_argument('--warm-start', default=None, help='continue from an older brain (checkpoint .pth); inputs it does not have start at 0')\n")
    rep("    global COUNT_PROBS, TOWER_ROAD_INSET\n", "    global COUNT_PROBS, TOWER_ROAD_INSET, THETA\n")
    rep("    moving_default = not args.static_obstacles\n",
        "    moving_default = not args.static_obstacles\n"
        "    THETA = (CLASSIC if args.prior == 'classic' else EVOLVED).copy()\n"
        "    if args.stack > 1:\n"
        "        raise SystemExit('the prior reads the first frame of the state: keep stack=1 (the motion cues give the brain its memory)')\n")
    rep("            e._update_sensors(); e._update_nav()                # the first observation already sees them where they are\n        return e\n",
        "            e._update_sensors(); e._update_nav()                # the first observation already sees them where they are\n"
        "        if args.motion_cues:\n"
        "            e.motion_cues = True; e.ray_delta = np.zeros(N_SENSORS); e._teleport = True\n"
        "        return e\n")
    rep("    env = make_env(args.world_seed, 'calm', args.seed)\n",
        "    env = make_env(args.world_seed, 'calm', args.seed)\n"
        "    env.approach_penalty = args.approach_penalty if args.motion_cues else 0.0            # only the training env is rewarded for it; the evaluation envs keep 0\n"
        "    env.flyer_speed = args.flyer_speed_start if (args.flyer_ramp and moving_default) else 1.0\n")
    rep("                               'moving_obstacles': moving_default, 'viewer_world': not args.classic_world})",
        "                               'moving_obstacles': moving_default, 'viewer_world': not args.classic_world,\n"
        "                               'motion_cues': args.motion_cues, 'prior': args.prior, 'prior_theta': [float(v) for v in THETA], 'prior_names': NAMES,\n"
        "                               'approach_penalty': args.approach_penalty, 'flyer_ramp': args.flyer_ramp})")
    rep("            progress = (time.time() - started) / total_s\n",
        "            progress = (time.time() - started) / total_s\n"
        "            if args.flyer_ramp and moving_default and step % 200 == 0:                  # idea 2: slow flyers first, full speed from flyer_ramp_frac of the run on\n"
        "                env.flyer_speed = args.flyer_speed_start + (1.0 - args.flyer_speed_start) * min(1.0, progress / max(1e-6, args.flyer_ramp_frac))\n")
    rep("    print(f'device={device} agent={args.agent}", "    print(f'motion_cues={args.motion_cues} approach_penalty={args.approach_penalty} flyer_ramp={args.flyer_ramp} prior={args.prior} theta={[round(float(v), 3) for v in THETA]}', flush=True)\n"
        "    print(f'device={device} agent={args.agent}")
    rep("                           pid_weight_max=args.pid_weight_max, pid_weight_min=args.pid_weight_min, **common)", "                           **common)")
    rep("    def save(path):\n", "    if args.warm_start:\n        print('warm start:', warm_start_from(brain, args.warm_start, args.wind_aware), flush=True)\n\n    def save(path):\n")
    rep("summary = dict(agent=args.agent, shield=args.shield,", "summary = dict(agent=args.agent, motion_cues=args.motion_cues, prior=args.prior, approach_penalty=args.approach_penalty, flyer_ramp=args.flyer_ramp, shield=args.shield,")
    OUT.write_text(src)
    return OUT


if __name__ == '__main__':
    p = build()
    print('wrote', p, len(p.read_text().splitlines()), 'lines')
