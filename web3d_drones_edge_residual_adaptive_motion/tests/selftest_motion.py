"""
Offline checks for the motion folder (no torch, no GPU, no ROS):  python tests/selftest_motion.py

It uses the torch-free copy of the world in ../ros2_edge_residual_adaptive (generated from the base trainer) and checks:
  - the CLASSIC prior reproduces the base folder's fixed rule (pid_bias * pid_scale) on random and on real states, with 37 and with 65 inputs;
  - the motion cues: 65 inputs, the first 37 unchanged, zero after a teleport, equal to the change of the readings otherwise; switching them off changes nothing about the flight;
  - the approach penalty: never a bonus, zero when off, fires for something closing in, and leaves collisions / goals alone;
  - flyer_speed scales the flyers' clock;
  - prior_logits: shapes, finite, batch == one by one, a closing obstacle raises 'reverse' / 'slow' only when the cues are there;
  - the sep-CMA-ES finds the optimum of a simple bowl;
  - the generated Kaggle trainer is up to date and contains the identical pasted modules; it compiles; the server compiles;
  - results/evolved_prior.json (if there) is within bounds.
"""
import json
import py_compile
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
PKG = HERE.parent / 'ros2_edge_residual_adaptive' / 'edge_residual_adaptive'
sys.path.insert(0, str(PKG))
import prior as P                                                                   # noqa: E402
import motion_cues as MC                                                            # noqa: E402
import build_trainer as BT                                                          # noqa: E402
import evolve_prior as EV                                                           # noqa: E402
from edge_residual_adaptive import flyers                                           # noqa: E402
from edge_residual_adaptive.core import EdgeDroneEnvV3                              # noqa: E402

results = []


def check(name, ok, detail=''):
    results.append(ok)
    print(('PASS  ' if ok else 'FAIL  ') + name + (f'   [{detail}]' if detail else ''), flush=True)


def new_env(seed=7, weather='windy', moving=True, motion=True, penalty=0.0):
    e = EdgeDroneEnvV3(seed=seed, rng_seed=seed, weather=weather, wind_aware=True, wind_comp=0.0)
    e.flyers_on, e.motion_cues, e.approach_penalty = moving, motion, penalty
    e._update_sensors(); e._update_nav()
    return e


def old_rule(s):
    """the base folder's pid_scale * pid_bias, written out again from its definition (YAW_DEADBAND 0.08, ELEV_DEADBAND 0.15, scale 1.5 -> 0 with the closest ray)"""
    bearing, elev = s[28], s[30]
    idx = 3 if elev > 0.15 else 4 if elev < -0.15 else 1 if bearing > 0.08 else 2 if bearing < -0.08 else 0
    out = np.zeros(9)
    out[idx] = 1.5 * (1.0 - min(1.0, max(0.0, float(np.max(s[:28])))))
    return out


def real_states(n, motion, seed=3):
    e = new_env(seed=seed, motion=motion)
    rng, out = random.Random(seed), []
    for _ in range(n):
        out.append(np.asarray(e.get_state(), dtype=np.float64))
        if e.step(rng.randrange(9)):
            pass
    return np.array(out)


# 1. the installs exist and are idempotent
EV.flyers.install(EdgeDroneEnvV3); MC.install_motion_cues(EdgeDroneEnvV3); MC.install_motion_cues(EdgeDroneEnvV3)
n_wraps = 0
fn = EdgeDroneEnvV3.get_state
while fn is not None:
    n_wraps += getattr(fn, '_motion', False)
    fn = fn.__closure__[0].cell_contents if getattr(fn, '__closure__', None) and callable(fn.__closure__[0].cell_contents) else None
check('install_motion_cues three times wraps get_state once', n_wraps == 1, f'{n_wraps}')

# 2. the classic prior == the old fixed rule
rng = np.random.default_rng(0)
S37 = np.concatenate([rng.uniform(0, 1, (3000, 28)), rng.uniform(-1, 1, (3000, 9))], axis=1)
S37[:, 28], S37[:, 29] = rng.uniform(-0.5, 0.5, 3000), 0.0
S37[:, 30] = rng.uniform(-0.5, 0.5, 3000)
ref = np.array([old_rule(s) for s in S37])
got = P.prior_logits(S37, P.CLASSIC)
check('CLASSIC prior == old rule on 3000 random 37-input states', np.abs(ref - got).max() < 1e-9, f'max diff {np.abs(ref - got).max():.1e}')
S65 = np.concatenate([S37, rng.uniform(-1, 1, (3000, 28))], axis=1)
got65 = P.prior_logits(S65, P.CLASSIC)
check('CLASSIC prior ignores the 28 cue inputs (65 inputs, same numbers)', np.abs(ref - got65).max() < 1e-9)
R = real_states(600, True)
ref_r = np.array([old_rule(s) for s in R])
check('CLASSIC prior == old rule on 600 real states of a flight', np.abs(ref_r - P.prior_logits(R, P.CLASSIC)).max() < 1e-9)

# 3. the motion cues
e = new_env(motion=True)
s0 = e.get_state()
check('65 inputs with the cues, 37 without', len(s0) == 65 and len(new_env(motion=False).get_state()) == 37, f'{len(s0)}')
check('zero cues right after a start', np.abs(np.array(s0[37:])).max() == 0.0)
e.step(0)
sig_before = np.asarray(e.signals)
e.step(0)
d = np.clip((np.asarray(e.signals) - sig_before) * MC.DELTA_GAIN, -1, 1)
check('cues == (new reading - old reading) * 10, clipped', np.abs(np.array(e.get_state()[37:]) - d).max() < 1e-9)
a, b = new_env(motion=True), new_env(motion=False)
same, steps_hit = True, 0
for i in range(800):
    act = (i * 7 + i // 5) % 9
    ra, rb = a.step(act), b.step(act)
    same &= (ra == rb) and (a.x, a.y, a.z) == (b.x, b.y, b.z) and a.reward == b.reward
    steps_hit += int(a.is_colliding)
check('cues on or off: the same flight and the same reward (penalty 0)', same, f'{steps_hit} collisions in the flight')
check('the first 37 inputs match between a cue env and a plain env, all along', np.abs(np.array(a.get_state()[:37]) - np.array(b.get_state()[:37])).max() == 0)

# 4. teleport (respawn) gives zero cues
e = new_env(motion=True, weather='gale')
zero_after_respawn, respawns = True, 0
for i in range(3000):
    if e.step(0):
        respawns += 1
        zero_after_respawn &= float(np.abs(e.ray_delta).max()) == 0.0
check('zero cues on the step after a respawn', zero_after_respawn and respawns > 0, f'{respawns} respawns')

# 5. the approach penalty
e0, e1 = new_env(motion=True, penalty=0.0), new_env(motion=True, penalty=0.5)
worst, fired, ok_no_bonus, ok_events = 0.0, 0, True, True
rr = random.Random(5)
for i in range(4000):
    act = rr.randrange(9)
    g0, g1 = e0.goals_reached, e1.goals_reached
    r0, r1 = e0.step(act), e1.step(act)
    ok_no_bonus &= e1.reward <= e0.reward + 1e-12
    ok_events &= (r0 == r1) and (e0.goals_reached - g0 == e1.goals_reached - g1) and e0.is_colliding == e1.is_colliding
    if e1.reward < e0.reward - 1e-12:
        fired += 1
        worst = max(worst, e0.reward - e1.reward)
    if r0 or r1:
        e0, e1 = new_env(seed=9 + i, motion=True, penalty=0.0), new_env(seed=9 + i, motion=True, penalty=0.5)
check('approach penalty is never a bonus', ok_no_bonus)
check('approach penalty leaves the flight, collisions and goals alone', ok_events)
check('approach penalty fires sometimes and stays small', 0 < fired < 1500 and worst <= 0.5 + 1e-9, f'{fired} of 4000 steps, largest {worst:.3f}')
# a hand-made case: the front ray reads 0.8 and closed by 0.1 in one step, far from the goal -> about the full penalty
def hand_made(penalty):
    f = new_env(motion=True, penalty=penalty)
    f.step(0)
    f.signals = [0.0] * 28
    f.signals[0] = 0.8
    f.ray_delta = np.zeros(28); f.ray_delta[0] = 0.1
    f.is_colliding, f._steps_to_goal, f.distance = False, 5, 400.0
    f._compute_reward(400.0, 0)
    return f.reward


cost = hand_made(0.0) - hand_made(0.5)
check('hand-made case: a ray at 0.8 closing 0.1 per step costs the full 0.5', abs(cost - 0.5) < 1e-9, f'{cost:.4f}')

# 6. the flyers clock
g = new_env(motion=True)
g.flyer_speed = 0.5
t0 = g.flyer_t
for _ in range(10):
    g.step(0)
check('flyer_speed 0.5 advances the flyers clock by half a step per step', abs((g.flyer_t - t0) - 5.0) < 1e-9, f'{g.flyer_t - t0}')
h = new_env(motion=True)
t0 = h.flyer_t
for _ in range(10):
    h.step(0)
check('flyer_speed 1.0 (the default) advances it by one', abs((h.flyer_t - t0) - 10.0) < 1e-9)

# 7. prior_logits behaviour
TH = np.array(BT.evolved_numbers() if BT.evolved_numbers() else P.CLASSIC)
B = P.prior_logits(S65[:50], TH)
one = np.array([P.prior_logits_one(s, TH) for s in S65[:50]])
check('prior_logits: shape (B, 9), finite, batch == one by one', B.shape == (50, 9) and np.isfinite(B).all() and np.abs(B - one).max() < 1e-5)
th = P.CLASSIC.copy(); th[P.NAMES.index('loom')], th[P.NAMES.index('brake')], th[P.NAMES.index('slow')] = 2.0, 1.0, 1.0
s = np.zeros(65); s[0], s[1], s[2] = 0.9, 0.9, 0.9
base_l = P.prior_logits_one(s, th)
s2 = s.copy(); s2[37:65] = 0.0; s2[37] = 1.0
closing_l = P.prior_logits_one(s2, th)
check("a closing obstacle ahead raises 'reverse' and 'slow'", closing_l[8] > base_l[8] and closing_l[7] > base_l[7], f'reverse {base_l[8]:.2f} -> {closing_l[8]:.2f}, slow {base_l[7]:.2f} -> {closing_l[7]:.2f}')
check('with only 37 inputs the same state gets no loom term', np.abs(P.prior_logits_one(s[:37], th) - base_l).max() < 1e-6)
th2 = P.CLASSIC.copy(); th2[P.NAMES.index('avoid')] = 2.0
sl = np.zeros(65); sl[0] = 0.8; sl[2] = 0.8; sl[4] = 0.8                    # blocked ahead and on the RIGHT side only
check('avoid turns away from the nearer side (right blocked -> left)', P.prior_logits_one(sl, th2)[1] > P.prior_logits_one(sl, th2)[2])

# 8. sep-CMA-ES on a bowl
es = EV.SepCMAES(np.full(12, 0.8), 0.3, 12, 1)
target = np.linspace(0.1, 0.9, 12)
best = 1e9
for _ in range(80):
    xs = es.ask()
    fit = -np.sum((xs - target) ** 2, axis=1)
    es.tell(xs, fit)
    best = min(best, -fit.max())
check('sep-CMA-ES drives a 12-D bowl close to its minimum', best < 1e-3, f'best squared error {best:.1e}')

# 9. the generated trainer
trainer = BT.OUT
txt = trainer.read_text() if trainer.exists() else ''
theta = BT.evolved_numbers()
check('trainer exists and compiles', bool(txt) and py_compile.compile(str(trainer), doraise=True) is not None)
check('trainer is up to date with build_trainer.py (rebuild gives the same text)', (lambda: (BT.build(), trainer.read_text() == txt)[1])())
check('trainer holds motion_cues.py exactly as pasted', BT.motion_block() in txt)
check('trainer holds prior.py exactly as pasted (with the evolved numbers)', BT.prior_block(theta) in txt)
a, b = txt.index(BT.FLYERS_BEGIN), txt.index(BT.FLYERS_END)
paste = txt[txt.index('\n', a) + 1:b].strip()
mod = BT.module_body(HERE / 'flyers.py').replace('import math\n', '', 1)
mod = '\n'.join(l for l in mod.splitlines() if not l.startswith('DRONE_RADIUS = ')).strip()        # the trainer already has math and DRONE_RADIUS
check('the flyers.py block in the trainer is the flyers.py of this folder', paste == mod)
check('no leftover of the old fixed rule in the trainer', '_pid_scale' not in txt and '_pid_bias' not in txt and 'pid_weight' not in txt)
check('the prior is used in act and in both places of learn', txt.count('_prior_np(s)') == 2 and txt.count('_prior_torch(') == 3)
check('the server compiles', py_compile.compile(str(HERE / 'server_edge_residual_adaptive_motion.py'), doraise=True) is not None)

# 10. the evolution result
ej = HERE / 'results' / 'evolved_prior.json'
if ej.exists():
    j = json.loads(ej.read_text())
    th = np.array(j['theta'])
    check('evolved_prior.json: 12 numbers inside the search bounds', len(th) == 12 and (th >= P.LOW - 1e-9).all() and (th <= P.HIGH + 1e-9).all())
    ho = j['held_out']
    print('   held-out (new skies), score = goals/1k - 2 * collisions/1k:')
    for k, v in ho.items():
        print(f'     {k:22s} {v}')
    check('the picked prior is at least as good as the classic one on the held-out moving skies', ho[f"{j['picked']}|moving"]['score'] >= ho['classic|moving']['score'])

# 11. the real PID controller
import pid_controller as PC                                                         # noqa: E402
c = PC.PID(1.0, 0.0, 0.0, 100.0)
check('PID: P term only is kp * error', abs(c(3.0) - 3.0) < 1e-12)
c = PC.PID(0.0, 0.1, 0.0, 100.0)
outs = [c(2.0) for _ in range(5)]
check('PID: I term accumulates the error (ki * sum)', abs(outs[-1] - 0.1 * 10.0) < 1e-9, f'{outs}')
c = PC.PID(0.0, 0.0, 1.0, 100.0, alpha=1.0)
c(1.0)
check('PID: D term is kd * (change of the error)', abs(c(4.0) - 3.0) < 1e-12)
c = PC.PID(0.0, 1.0, 0.0, 1.0, i_limit=1.0)
for _ in range(200):
    c(5.0)
check('PID: the integral does not wind up past its limit', c.integral <= 1.0 + 1e-9, f'integral {c.integral}')
c.integral, c.prev = 0.0, None
y, err = 0.0, []
loop = PC.PID(0.5, 0.05, 0.2, 100.0)
for _ in range(120):                                                                   # a first-order plant: y moves by the command, the setpoint is 10
    y += 0.3 * loop(10.0 - y)
    err.append(abs(10.0 - y))
check('PID: closes a simple loop (error below 0.05 after 120 steps)', err[-1] < 0.05, f'{err[-1]:.4f}')
check('arbitration: stop -> hover turn, reverse, climb, turn, slow, straight',
      [PC.PidController.choose(10, 0, 0.5), PC.PidController.choose(0, 0, -2), PC.PidController.choose(0, 3, 5), PC.PidController.choose(-9, 0, 5), PC.PidController.choose(0, 0, 3), PC.PidController.choose(0, 0, 5)] == [5, 8, 3, 2, 7, 0])
ctl = PC.PidController()
acts = [ctl.act(np.concatenate([rng.uniform(0, 1, 28), rng.uniform(-1, 1, 9)])) for _ in range(500)]
check('PidController.act: always an action 0..8', set(acts) <= set(range(9)), f'{sorted(set(acts))}')
goal_left = np.zeros(37); goal_left[28], goal_left[29], goal_left[31] = 0.3, -0.3, 0.8
ctl = PC.PidController()
check('PidController: goal on the left, clear sky -> turn left', ctl.act(goal_left) == 1)
wall = np.zeros(37); wall[0] = wall[1] = wall[2] = 1.0; wall[31] = 0.8
ctl = PC.PidController()
ctl.act(wall)
check('PidController: a wall dead ahead -> not straight (stop-turn or reverse)', ctl.act(wall) in (5, 6, 8))
import evolve_pid as EP                                                              # noqa: E402
rows = [EP.fly_pid(PC.DEFAULT, 5000 + i, EV.WEATHERS[i % 5], 800) for i in range(3)]
check('PID flies 3 x 800 steps among roaming flyers without error', all(len(r) == 3 for r in rows), f'{rows}')
tp = HERE / 'results' / 'tuned_pid.json'
if tp.exists():
    tj = json.loads(tp.read_text())
    tpar = np.array(tj['params'])
    check('tuned_pid.json: 12 gains inside the bounds', len(tpar) == 12 and (tpar >= PC.LOW - 1e-9).all() and (tpar <= PC.HIGH + 1e-9).all())
    for k, v in tj['held_out'].items():
        print(f'     {k:26s} {v}')

# 12. the video tool (no torch: a stand-in for torch.load is used to read a checkpoint)
import types                                                                         # noqa: E402
sys.path.insert(0, str(HERE / 'kaggle'))
import render_flight_video as RV                                                     # noqa: E402


class _T:
    def __init__(self, a):
        self.a = np.asarray(a, dtype=np.float32)

    def numpy(self):
        return self.a


wr = np.random.default_rng(4)
sd = {}
for i, (n_in, n_out) in zip((0, 2, 4), ((65, 16), (16, 16), (16, 9))):
    sd[f'actor.{i}.weight'], sd[f'actor.{i}.bias'] = _T(wr.normal(0, 0.3, (n_out, n_in))), _T(wr.normal(0, 0.1, n_out))
fake = types.ModuleType('torch')
fake.load = lambda path, **k: {'state_dict': sd, 'steps': 123, 'prior_theta': [float(v) for v in P.CLASSIC]}
sys.modules['torch'] = fake
nb_ = RV.NetBrain('x.pth')
xs = wr.uniform(-1, 1, (20, 65))
manual = []
for x in xs:
    h = np.maximum(x.astype(np.float32) @ sd['actor.0.weight'].a.T + sd['actor.0.bias'].a, 0)
    h = np.maximum(h @ sd['actor.2.weight'].a.T + sd['actor.2.bias'].a, 0)
    manual.append(int(np.argmax(h @ sd['actor.4.weight'].a.T + sd['actor.4.bias'].a + P.prior_logits_one(x, P.CLASSIC))))
check('video tool: the numpy actor of a checkpoint == the formula (network + prior)', [nb_.act(x) for x in xs] == manual and nb_.cues)
del sys.modules['torch']
import tempfile                                                                      # noqa: E402
import subprocess                                                                    # noqa: E402
with tempfile.TemporaryDirectory() as td:
    r = subprocess.run([sys.executable, str(HERE / 'kaggle' / 'render_flight_video.py'), '--brain', 'evolved', '--minutes', '0.03', '--size', '320x180', '--out', f'{td}/t.mp4'], capture_output=True, text=True)
    ok = r.returncode == 0 and (Path(td) / 't.mp4').exists() and (Path(td) / 't.mp4').stat().st_size > 1000
check('video tool: renders a 2-second 320x180 clip to an MP4', ok, (r.stderr or r.stdout)[-200:] if not ok else '')
nbk = json.loads((HERE / 'kaggle' / 'drone_motion_on_kaggle.ipynb').read_text())
srcs = ''.join(''.join(c['source']) for c in nbk['cells'])
check('notebook: trains 15 minutes and renders 5 minutes', '--minutes 15' in srcs and '--minutes 5 ' in srcs and 'record_viewer_video.py' in srcs and 'playwright install' in srcs)
check('notebook is up to date with make_notebook.py', (lambda: (subprocess.run([sys.executable, str(HERE / 'kaggle' / 'make_notebook.py')], capture_output=True), json.loads((HERE / 'kaggle' / 'drone_motion_on_kaggle.ipynb').read_text()) == nbk)[1])())

# 13. the messages for the real viewer
import record_viewer_video as RW                                                     # noqa: E402
we = new_env(seed=3, motion=False)
for _ in range(5):
    we.step(0)
wm = RW.world_messages(we)
sm = RW.state_message(we, 'x', 'day', dict(goals=1, hits=0, resp=0), 5)
fm = RW.flyers_message(we)
need = {'drone', 'sensors', 'goal', 'goalRadius', 'wind', 'weather', 'brain', 'light', 'goals', 'collPer1k', 'auto', 'brains', 'weathers', 'hasDetector'}
check('viewer messages: state has every field the viewer reads, 28 sensors, JSON-serialisable', need <= set(sm) and len(sm['sensors']) == 28 and bool(json.dumps(sm)))
check('viewer messages: obstacles + targets + flyers (15 numbers each, velocity 0)', wm[0]['type'] == 'obstacles' and wm[1]['type'] == 'targets' and fm['type'] == 'flyers' and (not fm['f'] or (len(fm['f'][0]) == 15 and fm['f'][0][12:] == [0.0, 0.0, 0.0])))
check('viewer: the page is found next to this folder', (RW.STATIC / 'index.html').is_file() and (RW.STATIC / 'js' / 'net.js').is_file())

print(f'\n{sum(results)}/{len(results)} checks passed')
sys.exit(0 if all(results) else 1)
