"""
The adaptive-weight residual SAC policy in plain numpy (no torch). It is exactly what web3d_drones_edge_residual_adaptive/server_edge_residual_adaptive.py flies:

    logits(s) = actor(s) + pid_scale(s) * pid_bias(s)

  pid_bias(s)   one-hot over the 9 actions: what a simple proportional controller (with a dead band) would do from the bearing / elevation to the goal
                (state entries 28 and 30): climb, descend, turn left, turn right, or go straight
  pid_scale(s)  how much to trust that controller: pid_max (1.5) when nothing is near, fading to pid_min (0.0) right next to an obstacle,
                from the closest of the 28 rangefinder signals (state entries 0..27). Near an obstacle the learned network decides alone.

The network is the actor of a finished SAC brain, so the action is the most probable one (argmax) unless greedy=False, which samples from softmax(logits).

Speed (this is called once per simulation step, so it is on the hot path): the weights are stored already transposed and contiguous, the two hidden layers
write into preallocated buffers (no allocation per call), the bias is added to ONE logit instead of building a one-hot vector and a scaled copy, and the
rangefinder maximum is a plain float. The results agree with the straightforward reference to float rounding (about 5e-6 on the logits, the same action on every one of
4,000 real states, see tools/offline_selftest.py).
"""

import numpy as np

N_SENSORS = 28
PID_BEARING_IDX, PID_ELEV_IDX = 28, 30
YAW_DEADBAND = 0.08            # bearing units (roughly 14 degrees): inside this, go straight instead of turning
ELEV_DEADBAND = 0.15


def pid_action(bearing, elev):
    """Index of the action a classical controller would prefer: 3 climb, 4 descend, 1 left, 2 right, 0 straight."""
    return 3 if elev > ELEV_DEADBAND else 4 if elev < -ELEV_DEADBAND else 1 if bearing > YAW_DEADBAND else 2 if bearing < -YAW_DEADBAND else 0


class ResidualAdaptiveActor:

    def __init__(self, path, pid_max=None, pid_min=None):
        z = np.load(path)
        self.n_obs = int(z['n_obs'])
        self.n_actions = int(z['actions'])
        self.pid_max = float(z['pid_max'] if pid_max is None else pid_max)
        self.pid_min = float(z['pid_min'] if pid_min is None else pid_min)
        t = lambda a: np.ascontiguousarray(np.asarray(a, dtype=np.float32).T)                    # (out, in) -> (in, out), contiguous
        self.w1, self.w2, self.w3 = t(z['w1']), t(z['w2']), t(z['w3'])
        self.b1, self.b2, self.b3 = (np.asarray(z[k], dtype=np.float32) for k in ('b1', 'b2', 'b3'))
        self._h1 = np.empty(self.w1.shape[1], dtype=np.float32)
        self._h2 = np.empty(self.w2.shape[1], dtype=np.float32)
        self._z = np.empty(self.n_actions, dtype=np.float32)
        self._span = self.pid_max - self.pid_min

    def pid_scale(self, closest):
        """pid_max when nothing is near (closest signal 0), pid_min when something is touching (closest signal 1)."""
        c = 1.0 if closest > 1.0 else 0.0 if closest < 0.0 else closest
        return self.pid_min + self._span * (1.0 - c)

    def logits(self, s):
        """s: the 37-value state (float32 array). Returns a view of an internal buffer: copy it if you keep it."""
        h1, h2, z = self._h1, self._h2, self._z
        np.matmul(s, self.w1, out=h1); h1 += self.b1; np.maximum(h1, 0.0, out=h1)
        np.matmul(h1, self.w2, out=h2); h2 += self.b2; np.maximum(h2, 0.0, out=h2)
        np.matmul(h2, self.w3, out=z); z += self.b3
        z[pid_action(float(s[PID_BEARING_IDX]), float(s[PID_ELEV_IDX]))] += self.pid_scale(float(s[:N_SENSORS].max()))
        return z

    def act(self, s, rng=None, greedy=True):
        z = self.logits(s)
        if greedy or rng is None:
            return int(np.argmax(z))
        p = np.exp(z - z.max())
        return int(np.searchsorted(np.cumsum(p), rng.random() * p.sum()))


def reference_logits(path, s):
    """The straightforward version (what the server does: new arrays on every call, a one-hot bias times a scale). Used by the self-test and the benchmark."""
    z = np.load(path)
    s = np.asarray(s, dtype=np.float32)
    h = np.maximum(s @ z['w1'].T + z['b1'], 0.0)
    h = np.maximum(h @ z['w2'].T + z['b2'], 0.0)
    closest = min(1.0, max(0.0, float(np.max(s[:N_SENSORS]))))
    bias = np.zeros(z['w3'].shape[0], dtype=np.float32)
    bias[pid_action(float(s[PID_BEARING_IDX]), float(s[PID_ELEV_IDX]))] = 1.0
    return h @ z['w3'].T + z['b3'] + (float(z['pid_min']) + (float(z['pid_max']) - float(z['pid_min'])) * (1.0 - closest)) * bias
