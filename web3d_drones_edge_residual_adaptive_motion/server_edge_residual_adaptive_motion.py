"""
Live 3D demo of the motion-cue / evolved-prior drone: the viewer of ../web3d_drones_edge_residual_adaptive with three more brains in its list.

    python server_edge_residual_adaptive_motion.py          then open http://localhost:8023

The server code, the page and the world are the base folder's (this file imports server_edge_residual_adaptive.py and changes only what differs), so a fix there reaches here too.
What this file adds:
  - 'motion brain (Kaggle)'     results/models/sac_motion/brain_final.pth : the checkpoint trained by edge_v3_16_residual_adaptive_motion_sac.py (65 inputs, its own 12 prior numbers
                                  are stored in the checkpoint). Listed only when the file exists.
  - 'evolved prior only'        no neural network at all: the 12 evolved numbers of results/evolved_prior.json choose the action (with the motion cues as input). It works the moment you start the
                                  server, and shows what evolution alone found.
  - 'classic prior only'        the same function with the numbers that reproduce the old fixed rule: the baseline the evolution had to beat.
  - the brain of the base folder (37 inputs, trained before moving obstacles), still there for comparison.
The 28 motion-cue inputs are switched on for a brain exactly when it has 65 inputs.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'web3d_drones_edge_residual_adaptive'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BASE))

import server_edge_residual_adaptive as S                                              # noqa: E402  (the base server: app, Sim, messages; also builds its own `sim`)
import motion_cues as MC                                                               # noqa: E402
import prior as P                                                                      # noqa: E402
import pid_controller as PC                                                            # noqa: E402
import uvicorn                                                                         # noqa: E402

PORT = 8023
EVOLVED_JSON = HERE / 'results' / 'evolved_prior.json'
MOTION_BRAIN = HERE / 'results' / 'models' / 'sac_motion' / 'brain_final.pth'


def evolved_theta():
    import json
    return np.array(json.loads(EVOLVED_JSON.read_text())['theta'], dtype=np.float64) if EVOLVED_JSON.exists() else P.CLASSIC.copy()


class PriorOnlyBrain:
    """no network: the 12-number prior alone (greedy)"""
    n_inputs, n_actions = 65, 9

    def __init__(self, theta):
        self.theta = np.asarray(theta, dtype=np.float64)

    def act(self, s, rng=None):
        return int(np.argmax(P.prior_logits_one(s, self.theta)))


class PidBrain:
    """the real PID controller of pid_controller.py (three loops with P, I and D terms), gains from results/tuned_pid.json if it exists. It has memory, so the server resets it on a respawn and a new goal."""
    n_inputs, n_actions = 34, 9

    def __init__(self, params):
        self.ctl = PC.PidController(params)

    def act(self, s, rng=None):
        return self.ctl.act(s)

    def reset(self):
        self.ctl.reset()


def tuned_pid_params():
    import json
    p = HERE / 'results' / 'tuned_pid.json'
    return np.array(json.loads(p.read_text())['params'], dtype=np.float64) if p.exists() else PC.DEFAULT


class MotionSacBrain:
    """the actor of a v3_16 checkpoint: network logits + prior_logits(s, the checkpoint's own 12 numbers)"""

    def __init__(self, path):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        sd = ck['state_dict']
        self.w = [(sd[f'actor.{i}.weight'].numpy().T.astype(np.float32).copy(), sd[f'actor.{i}.bias'].numpy().astype(np.float32).copy()) for i in (0, 2, 4)]
        self.n_inputs, self.n_actions = self.w[0][0].shape[0], self.w[2][0].shape[1]
        self.theta = np.array(ck.get('prior_theta', evolved_theta()), dtype=np.float64)

    def logits(self, s):
        (w1, b1), (w2, b2), (w3, b3) = self.w
        h = np.maximum(s @ w1 + b1, 0.0)
        h = np.maximum(h @ w2 + b2, 0.0)
        return h @ w3 + b3 + P.prior_logits_one(s, self.theta)

    def act(self, s, rng=None):
        return int(np.argmax(self.logits(np.asarray(s, dtype=np.float32))))


MC.install_motion_cues(S.EdgeDroneEnvV2)                                               # flags off until a 65-input brain is flying
_base_brain = S.SacNumpyBrain


def make_brains():
    brains = {}
    if MOTION_BRAIN.is_file():
        brains['motion brain (Kaggle)'] = MotionSacBrain(MOTION_BRAIN)
    brains['real PID (tuned gains)'] = PidBrain(tuned_pid_params())
    brains['evolved prior only'] = PriorOnlyBrain(evolved_theta())
    brains['classic prior only'] = PriorOnlyBrain(P.CLASSIC)
    for name, path in S.BRAINS.items():
        if path.is_file():
            brains[name + ' (base folder)'] = _base_brain(path)
    return brains


_base_step = S.Sim.step


def step(self):
    if self.brain_name is not None:
        brain = self.brains[self.brain_name]
        self.env.motion_cues = brain.n_inputs >= 65                                   # the 28 ray-change inputs, only for the brains that were built for them
        before = (self.respawns, self.env.goals_reached)
        result = _base_step(self)
        if hasattr(brain, 'reset') and (self.respawns, self.env.goals_reached) != before:
            brain.reset()                                                             # a respawn or a new goal: a PID forgets its integral and its last error
        return result
    return _base_step(self)


S.Sim.step = step
S.sim.brains = make_brains()
S.sim.brain_name = next(iter(S.sim.brains))
S.sim.new_env()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=PORT)
    ap.add_argument('--auto-weather', type=float, default=30.0, metavar='SECONDS')
    ap.add_argument('--auto-weather-mode', choices=['cycle', 'random'], default='cycle')
    ap.add_argument('--auto-light', type=float, default=20.0, metavar='SECONDS')
    ap.add_argument('--static-obstacles', action='store_true', help='no roaming birds and patrol drones')
    args = ap.parse_args()
    S.MOVING_OBSTACLES = not args.static_obstacles
    sim = S.sim
    sim.env.flyers_on = S.MOVING_OBSTACLES
    sim.env._update_sensors(); sim.env._update_nav()
    sim.auto_weather.update(on=args.auto_weather > 0, every=max(5.0, args.auto_weather or 30.0), mode=args.auto_weather_mode)
    sim.auto_light.update(on=args.auto_light > 0, every=max(5.0, args.auto_light or 20.0))
    S.LOG_DIR.mkdir(exist_ok=True)
    print(f'brains: {list(sim.brains)}   http://localhost:{args.port}')
    uvicorn.run(S.app, host='0.0.0.0', port=args.port, log_level='warning')
