"""
Live 3D view of the brain in the packaged Unreal game (no editor needed).

The physics, sensors and wind run here in Python (headless_sim.ArenaEnv -- exactly what the brain trained on) and every
step the drone / goal / chase-camera poses are pushed to the running game over Unreal's Remote Control HTTP API
(port 30010, started by the game itself). Unreal only draws.

    python ue_bridge.py                     with the game (or the editor's PIE, Remote Control plugin on) running;
                                            a standalone game needs the -RCWebControlEnable argument

NOT TESTED YET against a running game: the packaged build is blocked on this machine (see unreal_env_mapping.md).

    python ue_bridge.py --weather gale --hz 30 --brain logs/brain_final.pth

Unreal coordinates: x*100, -y*100, z*100 (cm), yaw negated -- see unreal_env_mapping.md.
"""

import argparse
import http.client
import json
import math
import random
import time
from pathlib import Path

import headless_sim as H
from headless_sim import E

HERE = Path(__file__).resolve().parent
ACTORS = json.loads((HERE / 'ue_actors.json').read_text())
CAM_BACK, CAM_UP, CAM_PITCH = 4000.0, 1500.0, -12.0          # chase camera: 40 m behind, 15 m above (Unreal cm)


class UnrealLink:
    def __init__(self, host='127.0.0.1', port=30010):
        self.host, self.port = host, port
        self.conn = None
        self.rid = 0

    def wait(self, timeout=120.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                c = http.client.HTTPConnection(self.host, self.port, timeout=2)
                c.request('GET', '/remote/info')
                if c.getresponse().status == 200:
                    return True
            except OSError:
                pass
            time.sleep(1.0)
        return False

    def _post(self, verb, url, body):
        if self.conn is None:
            self.conn = http.client.HTTPConnection(self.host, self.port, timeout=5)
        try:
            self.conn.request(verb, url, json.dumps(body), {'Content-Type': 'application/json'})
            r = self.conn.getresponse()
            data = r.read()
        except (OSError, http.client.HTTPException):
            self.conn = None
            raise
        return r.status, data

    @staticmethod
    def _pose(path, x, y, z, yaw, pitch=0.0, roll=0.0):
        return {'URL': '/remote/object/call', 'Verb': 'PUT', 'Body': {
            'objectPath': path, 'functionName': 'K2_SetActorLocationAndRotation', 'generateTransaction': False,
            'parameters': {'NewLocation': {'X': x, 'Y': y, 'Z': z}, 'NewRotation': {'Pitch': pitch, 'Yaw': yaw, 'Roll': roll},
                           'bSweep': False, 'bTeleport': True}}}

    def poses(self, items):
        """items: [(objectPath, x, y, z, yaw, pitch, roll)] in Unreal cm / degrees. One HTTP round trip."""
        reqs = []
        for i, it in enumerate(items):
            r = self._pose(*it)
            r['RequestId'] = i
            reqs.append(r)
        return self._post('PUT', '/remote/batch', {'Requests': reqs})


def to_ue(x, y, z):
    return x * 100.0, -y * 100.0, z * 100.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--brain', default=str(H.DEFAULT_BRAIN))
    ap.add_argument('--weather', choices=list(E.CONDITIONS), default='windy')
    ap.add_argument('--hz', type=float, default=60.0, help='simulation steps per second shown in Unreal (60 = real time)')
    ap.add_argument('--seed', type=int, default=H.WORLD_SEED)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=30010)
    ap.add_argument('--stride', type=int, default=1, help='send a pose every N steps (raise it if the link is slow)')
    args = ap.parse_args()

    policy = H.ResidualSacPolicy(args.brain)
    env = H.ArenaEnv(seed=args.seed, rng_seed=5, weather=args.weather, wind_aware=True)
    link = UnrealLink(args.host, args.port)
    print('waiting for Unreal Remote Control on port', args.port, '...')
    if not link.wait():
        raise SystemExit('Unreal did not answer. Start the game with -RCWebControlEnable (or use run_unreal_sim.bat).')
    print('connected -- flying', args.weather)

    rng = random.Random(5)
    random_steps = goals = coll = 0
    cam = None
    t_next = time.perf_counter()
    for t in range(1 << 62):
        a = rng.randrange(E.N_ACTIONS) if random_steps > 0 else policy(env.get_state())
        random_steps = max(0, random_steps - 1)
        if env.step(a):
            random_steps = E.RANDOM_STEPS_AFTER_RESPAWN
            cam = None
        coll += int(env.is_colliding)
        if t % args.stride == 0:
            ux, uy, uz = to_ue(env.x, env.y, env.z)
            yaw = -env.yaw
            pitch = -env.vz / E.CLIMB_RATE * 12.0
            fx, fy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            target = (ux - fx * CAM_BACK, uy - fy * CAM_BACK, uz + CAM_UP)
            cam = target if cam is None else tuple(c + 0.12 * (g - c) for c, g in zip(cam, target))
            gx, gy, gz = to_ue(*env.goal)
            try:
                link.poses([(ACTORS['drone'], ux, uy, uz, yaw, pitch, 0.0),
                            (ACTORS['goal'], gx, gy, gz, 0.0, 0.0, 0.0),
                            (ACTORS['camera'], cam[0], cam[1], cam[2], yaw, CAM_PITCH, 0.0)])
            except OSError:
                raise SystemExit('lost the connection to Unreal (game closed?)')
        if t % 600 == 0:
            print(f'step {t:7d}  goals {env.goals_reached:4d}  collisions {coll:4d}  wind {math.hypot(env.wind[0], env.wind[1]):.1f}', flush=True)
        t_next += 1.0 / args.hz
        delay = t_next - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            t_next = time.perf_counter()


if __name__ == '__main__':
    main()
