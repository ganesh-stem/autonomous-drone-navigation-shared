"""
Self-test WITHOUT ROS 2: it runs the real node classes (sim_node, brain_node, kpi_node, telemetry_node, gcs_node, web_bridge) of this package against a tiny
in-process stand-in for rclpy (publishers, subscriptions and timers on a message queue), so the logic can be checked on any machine with numpy.

What it checks (it exits with a non-zero status when something is wrong):
  1. the numpy world in core.py is the same world as the server's (web3d_drones_edge/src EdgeDroneEnvV2 + the viewer's obstacle density), step by step
  2. the optimised policy gives the same action as the straightforward reference on thousands of real states, and the cost of both
  3. sim_node + brain_node, in lockstep through messages, fly EXACTLY the trajectory of the plain loop the server runs (same positions, goals, collisions)
  4. the same in the timer mode (fast:=false): no late answers, and the counters of the KPI node match
  5. every number the viewer's HUD reads (static/js/ui.js hud()) is in the message of web_bridge.state_message()
  6. telemetry_node + gcs_node: MAVLink frames arrive with no CRC errors and no losses

What it does NOT check: the real ROS 2 transport (DDS), the executors, the launch file, the camera / detector nodes (they need torch) and the browser.

    python tools/offline_selftest.py [--steps 6000]
"""
import argparse
import array
import importlib
import re
import sys
import tempfile
import time
import types
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = ROOT / 'edge_residual_adaptive'
sys.path.insert(0, str(PKG_DIR))
sys.path.insert(0, str(ROOT.parent / 'web3d_drones_edge' / 'src'))
WEIGHTS = PKG_DIR / 'weights' / 'residual_adaptive_actor.npz'


# ───────────────────────── a tiny stand-in for rclpy ─────────────────────────
class Bus:
    def __init__(self):
        self.subs, self.queue = defaultdict(list), deque()

    def pump(self, stop=None):
        while self.queue:
            topic, msg = self.queue.popleft()
            for cb in list(self.subs[topic]):
                cb(msg)
            if stop is not None and stop():
                self.queue.clear()
                return


BUS = Bus()
PARAMS = {}                                       # node name -> {parameter: value} (what the launch file would pass)


class _Auto:                                      # PoseStamped.pose.position.x = ... : nested attributes appear when assigned
    def __getattr__(self, k):
        v = _Auto()
        object.__setattr__(self, k, v)
        return v


class Float32MultiArray:
    def __init__(self): self._d = array.array('f')
    data = property(lambda s: s._d, lambda s, v: setattr(s, '_d', array.array('f', v)))      # like rclpy: a float32 array, values are rounded to float32


class Int32MultiArray:
    def __init__(self): self._d = array.array('i')
    data = property(lambda s: s._d, lambda s, v: setattr(s, '_d', array.array('i', v)))


class UInt8MultiArray:
    def __init__(self): self.data = []


class String:
    def __init__(self): self.data = ''


class Publisher:
    def __init__(self, topic): self.topic = topic
    def publish(self, m): BUS.queue.append((self.topic, m))
    def get_subscription_count(self): return len(BUS.subs[self.topic])


class Timer:
    def __init__(self, cb): self.cb, self.cancelled = cb, False
    def cancel(self): self.cancelled = True


class Clock:
    def now(self): return types.SimpleNamespace(to_msg=lambda: _Auto())


class Node:
    def __init__(self, name, **kw):
        self.name, self._p, self.timers = name, {}, []
    def declare_parameter(self, name, default):
        self._p[name] = types.SimpleNamespace(value=PARAMS.get(self.name, {}).get(name, default))
    def get_parameter(self, name): return self._p[name]
    def create_publisher(self, typ, topic, qos): return Publisher(topic)
    def create_subscription(self, typ, topic, cb, qos): BUS.subs[topic].append(cb)
    def create_timer(self, period, cb):
        t = Timer(cb)
        self.timers.append(t)
        return t
    def get_logger(self): return types.SimpleNamespace(info=lambda *a: None, warn=lambda *a: None, error=lambda *a: None)
    def get_clock(self): return Clock()
    def destroy_node(self): pass


def install_fake_ros():
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m
    mod('rclpy', init=lambda *a, **k: None, ok=lambda: True, shutdown=lambda: None, try_shutdown=lambda: None, spin=lambda n: None)
    mod('rclpy.node', Node=Node)
    mod('rclpy.executors', ExternalShutdownException=RuntimeError)
    mod('std_msgs'); mod('std_msgs.msg', Float32MultiArray=Float32MultiArray, Int32MultiArray=Int32MultiArray, UInt8MultiArray=UInt8MultiArray, String=String)
    mod('geometry_msgs'); mod('geometry_msgs.msg', PointStamped=_Auto, PoseStamped=_Auto, Vector3Stamped=_Auto)
    mod('sensor_msgs'); mod('sensor_msgs.msg', BatteryState=_Auto, Image=_Auto)
    mod('ament_index_python'); mod('ament_index_python.packages', get_package_share_directory=lambda name: str(PKG_DIR))


def load(name):
    return importlib.import_module(f'edge_residual_adaptive.{name}')


def check(ok, text):
    print(('  PASS  ' if ok else '  FAIL  ') + text)
    if not ok:
        check.failed += 1


check.failed = 0


# ───────────────────────── the checks ─────────────────────────
def check_world_parity(steps):
    print('1. world: core.py (ROS) against EdgeDroneEnvV2 (server)')
    import drone_env as D
    D.COUNT_PROBS = [0.19, 0.30, 0.34, 0.17]                    # what server_edge_residual_adaptive.py sets before it builds its world
    server = (ROOT.parent / 'web3d_drones_edge_residual_adaptive' / 'server_edge_residual_adaptive.py').read_text(encoding='utf-8')
    exec(server[server.index('ROAD_INSET = 18.0'):server.index('D.DroneEnv._chunk = _chunk_towers_off_roads') + len('D.DroneEnv._chunk = _chunk_towers_off_roads')], {'D': D})   # the road-inset shift of the towers
    from env_edge_v2 import EdgeDroneEnvV2
    core = load('core')
    for weather in ('calm', 'windy', 'gale_dusty'):
        a = EdgeDroneEnvV2(seed=1234, rng_seed=5, weather=weather, wind_aware=True)
        b = core.EdgeDroneEnvV3(seed=1234, rng_seed=5, weather=weather, wind_aware=True, wind_comp=0.0)
        rng, same = np.random.default_rng(0), True
        for _ in range(steps):
            act = int(rng.integers(0, 9))
            ra, rb = a.step(act), b.step(act)
            if ra != rb or not np.array_equal(a.get_state(), b.get_state()) or (a.x, a.y, a.z, a.yaw) != (b.x, b.y, b.z, b.yaw):
                same = False
                break
        check(same, f'{weather}: {steps} random steps identical (state, position, respawns)')


def check_towers_off_roads():
    print('1b. towers stay off the roads and sidewalks (they run 17 units either side of every chunk border)')
    core = load('core')
    w = core.DroneEnv(seed=1234)
    worst, n = 1e9, 0
    for cx in range(-12, 13):
        for cy in range(-12, 13):
            for b in w._chunk(cx, cy):
                if b[2] <= 0.5:
                    n += 1
                    worst = min(worst, b[0] - cx * core.CHUNK, (cx + 1) * core.CHUNK - b[3], b[1] - cy * core.CHUNK, (cy + 1) * core.CHUNK - b[4])
    check(n > 100 and worst >= 17.0, f'{n} towers in 25 x 25 chunks, the closest is {worst:.1f} units from a chunk border (the sidewalk ends at 17)')


def check_policy(steps):
    print('2. policy: optimised against the straightforward reference')
    from edge_residual_adaptive.residual_policy import ResidualAdaptiveActor, reference_logits
    core = load('core')
    actor = ResidualAdaptiveActor(str(WEIGHTS))
    env = core.EdgeDroneEnvV3(seed=1234, rng_seed=5, weather='windy', wind_aware=True)
    rng, states = np.random.default_rng(1), []
    for k in range(min(steps, 4000)):
        s = np.asarray(env.get_state(), dtype=np.float32)
        states.append(s)
        env.step(actor.act(s) if k % 3 else int(rng.integers(0, 9)))
    z0 = np.load(str(WEIGHTS))
    diff = max(float(np.abs(actor.logits(s) - reference_logits(str(WEIGHTS), s)).max()) for s in states[:600])
    other = sum(int(np.argmax(actor.logits(s)) != np.argmax(reference_logits(str(WEIGHTS), s))) for s in states[:600])
    check(other == 0 and diff < 1e-4, f'same action on {min(len(states), 600)} real states, largest logit difference {diff:.1e}')
    w1t, w2t, w3t, b1, b2, b3 = z0['w1'].T, z0['w2'].T, z0['w3'].T, z0['b1'], z0['b2'], z0['b3']

    def naive(s):                                                  # what the server does on every step
        h = np.maximum(s @ w1t + b1, 0.0)
        h = np.maximum(h @ w2t + b2, 0.0)
        closest = min(1.0, max(0.0, float(np.max(s[:28]))))
        bias = np.zeros(9, dtype=np.float32)
        bias[3 if s[30] > 0.15 else 4 if s[30] < -0.15 else 1 if s[28] > 0.08 else 2 if s[28] < -0.08 else 0] = 1.0
        return int(np.argmax(h @ w3t + b3 + (0.0 + 1.5 * (1.0 - closest)) * bias))

    def best(f):
        t = []
        for _ in range(5):
            t0 = time.perf_counter()
            for s in states[:2000]:
                f(s)
            t.append((time.perf_counter() - t0) / 2000)
        return min(t) * 1e6
    print(f'        cost per decision: straightforward {best(naive):.1f} us, optimised {best(actor.act):.1f} us')


def build_system(fast, hz=60.0, weather='windy', seed=1234, moving=False):
    """sim_node + brain_node + kpi_node + telemetry_node + gcs_node + web_bridge on the fake bus, like the launch file starts them."""
    BUS.subs.clear()
    BUS.queue.clear()
    log_dir = tempfile.mkdtemp(prefix='edge_residual_selftest_')
    PARAMS.clear()
    PARAMS['sim_node'] = {'world_seed': seed, 'weather': weather, 'fast': fast, 'sim_hz': hz, 'aux_every': 2, 'auto_weather_s': 0.0, 'auto_light_s': 0.0,
                          'moving_obstacles': moving}                 # the parity checks compare with the static world of the server, so they run static; check 8 runs the moving one
    PARAMS['brain_node'] = {'weights': str(WEIGHTS)}
    PARAMS['kpi_node'] = {'log_dir': log_dir}
    PARAMS['web_bridge'] = {'world_seed': seed, 'log_dir': log_dir}
    PARAMS['camera_node'] = {'world_seed': seed, 'light': 'day', 'hz': 2.0}
    load('web_bridge').WebBridge.run_server = lambda self: None          # no HTTP / WebSocket server here
    nodes = types.SimpleNamespace(
        sim=load('sim_node').SimNode(), brain=load('brain_node').BrainNode(), kpi=load('kpi_node').KpiNode(),
        tele=load('telemetry_node').TelemetryNode(), gcs=load('gcs_node').GcsNode(), bridge=load('web_bridge').WebBridge(), camera=load('camera_node').CameraNode())
    return nodes


def run_nodes(n, steps):
    sim = n.sim
    sim.first.cancel()
    sim.first_state()                                              # the first observation (the launch does this a second after start)
    if sim.fast:                                                   # fast: every answer triggers the next step by itself
        BUS.pump(stop=lambda: sim.step_no >= steps)
    else:                                                          # timer: one tick per period; the brain answers in between
        BUS.pump()
        while sim.step_no < steps:
            sim.tick()
            BUS.pump()
    sim.paused = True
    BUS.queue.clear()


def run_plain(steps, weather='windy', seed=1234):
    """What server_edge_residual_adaptive.py does in its loop: act on the state, step the world, 90 random actions after a respawn."""
    import random
    from edge_residual_adaptive.residual_policy import ResidualAdaptiveActor
    core = load('core')
    actor, env, rng = ResidualAdaptiveActor(str(WEIGHTS)), core.EdgeDroneEnvV3(seed=seed, rng_seed=5, weather=weather, wind_aware=True), random.Random(7)
    coll = resp = random_left = 0
    for _ in range(steps):
        a = rng.randrange(9) if random_left > 0 else actor.act(np.asarray(np.asarray(env.get_state(), dtype=np.float32), dtype=np.float32))
        random_left = max(0, random_left - 1)
        r = env.step(a)
        coll += int(env.is_colliding)
        if r:
            resp += 1
            random_left = 90
    return env, coll, resp


def check_lockstep(steps, fast):
    mode = 'fast (answer-driven)' if fast else 'timer'
    print(f'{3 if fast else 4}. {mode}: nodes through messages against the plain loop, {steps} steps in windy air')
    n = build_system(fast)
    t0 = time.perf_counter()
    run_nodes(n, steps)
    dt = time.perf_counter() - t0
    env, coll, resp = run_plain(steps)
    e = n.sim.env
    same = (e.x, e.y, e.z, e.yaw) == (env.x, env.y, env.z, env.yaw) and e.goals_reached == env.goals_reached
    check(n.sim.step_no == steps, f'ran {n.sim.step_no} steps ({n.sim.step_no / dt:.0f} steps/s through the in-process bus)')
    check(n.sim.stale == 0, f'late brain answers: {n.sim.stale}')
    check(same and n.sim.collisions == coll and n.sim.respawns == resp,
          f'same flight as the plain loop: position {e.x:.2f}, {e.y:.2f}, {e.z:.2f}, goals {e.goals_reached} (plain {env.goals_reached}), collisions {n.sim.collisions} (plain {coll}), respawns {n.sim.respawns} (plain {resp})')
    n.kpi.publish()
    BUS.pump()
    k = n.kpi.kpi
    check(k.get('steps', 0) > 0 and k.get('goals') == e.goals_reached, f'KPI node: {k.get("goals")} goals, {k.get("collisions")} collisions, {k.get("goals_per_1k")} goals / 1k, {k.get("collisions_per_1k")} collisions / 1k')
    return n


def check_viewer_message(n):
    print('5. web_bridge: everything the HUD reads is in the state message')
    ui = (PKG_DIR / 'web' / 'js' / 'ui.js').read_text(encoding='utf-8')
    hud = ui[ui.index('function hud('):]
    hud = hud[:hud.index('\n}\n')]
    wanted = set(re.findall(r'\bs\.(\w+)', hud)) - {'length', 'map'}
    check(len(n.bridge.rays) == 28 * 4 and bool(n.bridge.status), 'the bridge received the rays and the status over the bus (the simulator builds /rays only because the bridge subscribes)')
    msg = n.bridge.state_message()
    missing = sorted(w for w in wanted if w not in msg)
    check(not missing, f'{len(wanted)} fields read by hud(): missing {missing or "none"}')
    check(msg['brain'] == 'adaptive-weight residual SAC on PID' and len(msg['sensors']) == 28 and msg['weather'] == 'windy', 'brain name, 28 sensor rays, weather')
    wm = n.bridge.world_messages(msg['drone']['x'], msg['drone']['y'])
    check(wm[0]['type'] == 'obstacles' and wm[1]['type'] == 'targets' and len(wm[0]['boxes']) > 0, f'world messages: {len(wm[0]["boxes"])} obstacle boxes, {len(wm[1]["trees"])} trees, {len(wm[1]["vehicles"])} vehicles')
    import json
    try:
        import orjson
        same = orjson.loads(orjson.dumps(msg)) == json.loads(json.dumps(msg))
    except ImportError:
        same = True
    check(same, 'the message encodes to JSON (orjson and json agree when orjson is installed)')


def check_telemetry(n):
    print('6. telemetry_node -> gcs_node (MAVLink)')
    st = n.gcs.gs.stats()
    check(st['frames'] > 0 and st['crc_errors'] == 0 and st['lost_packets'] == 0, f'{st["frames"]} frames in {n.sim.step_no} steps, {st["crc_errors"]} CRC errors, {st["lost_packets"]} lost')


def check_cars(n):
    print('7. the cars of the city: traffic, parked cars, and what the camera sees')
    import json
    sim, CW = n.sim, load('city_world')
    rows = sim.traffic.snapshot()
    msg = n.bridge.cars_message()
    check(n.bridge.traffic is not None and len(msg['cars']) > 0 and all(len(r) == 7 for r in msg['cars']) and abs(msg['t'] - sim.traffic.clock) < 0.2,
          f'sim_node publishes the traffic, web_bridge turns it into a "cars" message: {len(msg["cars"])} cars, clock {msg["t"]} s')
    check(len(msg['cars']) <= len(rows) + 5 and sum(1 for r in msg['cars'] if r[6] > 1) > 0, f'{sum(1 for r in msg["cars"] if r[6] > 1)} of them are moving')
    wm = n.bridge.world_messages(sim.env.x, sim.env.y)
    check('parked' in wm[1] and wm[1]['vehicles'] == [], f'the targets message carries the parked cars ({len(wm[1]["parked"])} here) and no random sim cars')
    server = (ROOT.parent / 'web3d_drones_edge_residual_adaptive' / 'city_world.py').read_bytes()
    check(server == (PKG_DIR / 'edge_residual_adaptive' / 'city_world.py').read_bytes(), 'the package copy of city_world.py is identical to the server\'s (tools/build_core.py copies it)')

    seen = []                                                       # the camera's "really there" counts
    BUS.subs['/edge_residual/camera/truth'].append(lambda m: seen.append(json.loads(m.data)))
    status = json.loads(n.camera.raw)

    def look(x, y, z=60.0):
        n.camera.raw = json.dumps(dict(status, x=x, y=y, z=z))
        n.camera.tick()
        BUS.pump()                                                  # deliver the camera's /truth message
        return seen[-1]['truth'][0]
    moving = [r for r in sim.traffic.snapshot() if r[6] > 1]
    n.camera.traffic_rows = [r for r in sim.traffic.snapshot()]
    best = max((look(r[0], r[1]) for r in moving[:12]), default=0)
    check(best >= 1, f'driving cars: the camera above a road counts up to {best} vehicles really there (the old random cars are gone)')
    lot = next(((bx, by) for bx in range(-30, 31) for by in range(-30, 31) if CW.parked_cars(sim.seed, bx, by, n.camera.world._chunk(bx, by))), None)
    cx, cy = ((lot[0] + 0.5) * 250, (lot[1] + 0.5) * 250) if lot else (0, 0)
    parked = CW.parked_cars(sim.seed, lot[0], lot[1], n.camera.world._chunk(*lot)) if lot else []
    n.camera.traffic_rows = []
    counts = [look(x, y, 45.0) for x, y in [(cx - 60, cy), (cx, cy), (cx + 60, cy)]]
    check(lot is not None and max(counts) >= 2, f'parked cars: above the car park at block {lot} ({len(parked)} cars in total) the camera counts {counts} vehicles')


def check_flyers():
    print('8. roaming birds and patrol drones as obstacles: the simulator, the camera and the viewer all put them in the same place')
    import json
    flyers, core = load('flyers'), load('core')
    # the path function itself, on the floating blocks of a real sky
    w0 = core.DroneEnv(seed=1234)
    base = w0.chunk_boxes_around(0, 0, 6)
    base = base[base[:, 2] > 0.5]
    k = flyers._consts(base)
    n_bird, n_drone = int((k['kind'] < 3).sum()), int((k['kind'] == 3).sum())
    spots = len(base)
    check(spots >= 100 and spots <= n_bird <= 3 * spots and spots <= n_drone <= 2 * spots, f'{spots} floating blocks send out {n_bird} birds and {n_drone} patrol drones (1-3 birds and 1-2 drones each)')
    pb = pd = dev = reach = 0.0
    zlo, zhi = 1e9, -1e9
    bx, by = k['cx'], k['cy']
    for t in range(0, 6000, 17):
        pos, vel = flyers._at(k, t)
        pos1, _ = flyers._at(k, t + 1)
        sp = np.hypot(vel[:, 0], vel[:, 1])                                    # the limits are on the horizontal speed
        pb, pd = max(pb, float(sp[k['kind'] < 3].max())), max(pd, float(sp[k['kind'] == 3].max()))
        dev = max(dev, float(np.abs((pos1 - pos) - vel).max()))
        reach = max(reach, float(np.hypot(pos[:, 0] - bx, pos[:, 1] - by).max()))
        zlo, zhi = min(zlo, float(pos[:, 2].min())), max(zhi, float(pos[:, 2].max()))
    check(pb <= flyers.SPEED_BIRD[1] + 1e-9 and pd <= flyers.SPEED_DRONE[1] + 1e-9, f'peak horizontal speed: birds {pb:.2f} units/step (limit {flyers.SPEED_BIRD[1]}), patrol drones {pd:.2f} (limit {flyers.SPEED_DRONE[1]}); the drone flies 5.0')
    check(reach <= flyers.MARGIN - 15.0 + 1e-6 and flyers.Z_RANGE[0] <= zlo and zhi <= flyers.Z_RANGE[1] and reach > 250, f'they roam up to {reach:.0f} units from their spot (window margin {flyers.MARGIN:.0f}), between {zlo:.0f} and {zhi:.0f} high')
    check(dev < 0.1, f'the velocity sent to the viewer matches the real step-to-step motion (largest difference {dev:.3f} units per step)')

    # the moving system on the fake bus
    n = build_system(fast=True, moving=True)
    run_nodes(n, 900)
    sim = n.sim
    check(sim.env.flyers_on and sim.env.flyer_t == sim.step_no, f'sim_node: moving_obstacles is on and the obstacle clock ({sim.env.flyer_t}) equals the step number ({sim.step_no})')
    check(n.camera.world.flyers_on and 0 <= sim.env.flyer_t - n.camera.world.flyer_t <= 6 and n.bridge.world.flyers_on and 0 <= sim.env.flyer_t - n.bridge.world.flyer_t <= 6,
          f'camera_node ({n.camera.world.flyer_t}) and web_bridge ({n.bridge.world.flyer_t}) learned the clock from /edge_residual/flyers (within a few steps of {sim.env.flyer_t})')
    x, y = sim.env.x, sim.env.y
    n.camera.world.flyer_t = n.bridge.world.flyer_t = sim.env.flyer_t         # the same moment
    mine = sim.env.boxes_near(x, y, 300)
    cam = n.camera.world.boxes_near(x, y, 300)
    static = core.EdgeDroneEnvV3(seed=1234, rng_seed=5, weather='windy', wind_aware=True).boxes_near(x, y, 300)
    have = {tuple(np.round(r, 3)) for r in static}
    fb = np.array([r for r in mine if tuple(np.round(r, 3)) not in have])
    check(len(mine) == len(cam) and np.array_equal(mine, cam) and len(fb) > 0, f'the camera node sees exactly the boxes the simulator collides with: {len(static)} static ones plus {len(fb)} roaming birds and drones near the drone')
    wm = n.bridge.world_messages(x, y)
    fm = n.bridge.flyers_message(x, y)
    check(any(b[2] > 0.5 for b in wm[0]['boxes']) and len(fm['f']) > 0 and all(len(r) == 15 for r in fm['f']),
          f'live viewer messages: the "obstacles" message still has the floating blocks for the gates ({sum(1 for b in wm[0]["boxes"] if b[2] > 0.5)} here), "flyers" has {len(fm["f"])} birds and drones')
    centres = [((r[9]), (r[10]), (r[11])) for r in fm['f']]
    found = sum(1 for b in fb if any(abs((b[0] + b[3]) / 2 - c[0]) < 0.06 and abs((b[1] + b[4]) / 2 - c[1]) < 0.06 and abs((b[2] + b[5]) / 2 - c[2]) < 0.06 for c in centres))
    check(len(fb) > 0 and found == len(fb), f'all {len(fb)} roaming collision boxes near the drone are exactly where the "flyers" message puts the birds and drones')
    json.dumps(fm)
    # the moving world is real: a short flight with the brain, and how fast it runs
    import random
    import time
    from edge_residual_adaptive.residual_policy import ResidualAdaptiveActor
    actor, env, rg = ResidualAdaptiveActor(str(WEIGHTS)), core.EdgeDroneEnvV3(seed=1234, rng_seed=5, weather='calm', wind_aware=True), random.Random(7)
    env.flyers_on = True
    env._update_sensors(); env._update_nav()
    random_left, t0 = 0, time.perf_counter()
    for _ in range(1500):
        a = rg.randrange(9) if random_left > 0 else actor.act(np.asarray(env.get_state(), dtype=np.float32))
        random_left = max(0, random_left - 1)
        if env.step(a):
            random_left = 90
    rate = 1500 / (time.perf_counter() - t0)
    check(env.flyer_t == 1500 and env.goals_reached >= 2, f'a 1500-step flight among roaming birds and drones: {env.goals_reached} goals, {env.respawns} respawns ({rate:.0f} steps/s with the brain)')
    srv = (ROOT.parent / 'web3d_drones_edge_residual_adaptive' / 'flyers.py').read_bytes()
    check(srv == (PKG_DIR / 'edge_residual_adaptive' / 'flyers.py').read_bytes(), 'the package copy of flyers.py is identical to the server\'s (tools/build_core.py copies it)')
    trainer = (ROOT.parent / 'web3d_drones_edge_residual_adaptive' / 'edge_v3_12_residual_adaptive_sac.py').read_text(encoding='utf-8')
    a = trainer.index('# ── flyers.py begins here'); a = trainer.index('\n', a) + 1
    pasted = trainer[a:trainer.index('# ── flyers.py ends here')].strip()
    original = srv.decode('utf-8'); original = original[original.index('AMP_XY = '):].replace('DRONE_RADIUS = 3.0             # the same number as drone_env.DRONE_RADIUS\n', '').strip()
    check(pasted == original, 'the copy of the flyers code pasted into the standalone Kaggle trainer is identical to flyers.py')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=6000)
    args = ap.parse_args()
    install_fake_ros()
    check_world_parity(min(args.steps, 6000))
    check_towers_off_roads()
    check_policy(args.steps)
    n = check_lockstep(args.steps, fast=True)
    check_lockstep(min(args.steps, 1200), fast=False)
    check_viewer_message(n)
    check_telemetry(n)
    check_cars(n)
    check_flyers()
    print('\nALL CHECKS PASSED' if not check.failed else f'\n{check.failed} CHECK(S) FAILED')
    sys.exit(1 if check.failed else 0)


if __name__ == '__main__':
    main()
