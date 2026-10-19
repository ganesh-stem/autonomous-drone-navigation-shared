"""
A real PID controller for the drone: three feedback loops (yaw, altitude, speed), each with a proportional, an integral and a derivative term, with anti-windup, whose continuous commands are
rounded to the nearest of the environment's 9 discrete actions.

Why this file exists: the 'PID' of the residual trainers (pid_bias / pid_scale, and the prior of prior.py) is NOT a PID: it is a dead-band rule that picks an action from the sign of the goal error.
It has no integral term, no derivative term, no gains and no memory. This controller has all of those. It is a standalone policy (like the 'prior only' brains of the server): it needs memory
(the integral and the previous error), which the replay buffer of the SAC trainer does not store, so it is NOT wired into the learner. It flies on its own, and evolve_pid.py tunes its gains.

The three loops (all in physical units, one time step = 1):
  yaw       error  = bearing to the goal (degrees; state[28] * 180), plus an avoidance offset (turn away from the nearer side when the front is blocked)
            output = a turn-rate demand in degrees per step (+ = left), at most +-25 (the hover turn rate)
  altitude  error  = height of the goal above the drone (units; distance * sin(elevation) from the state), plus an offset toward the freer side (above or below) when the front is blocked
            output = a climb-rate demand in units per step (+ = up), at most +-4 (the climb rate)
  speed     setpoint = 5 * (1 - how blocked the front is) (units per step); measured = the speed the controller commanded last step (the environment has no speed sensor in its state)
            output = a speed demand in units per step, from -2.5 (reverse) to 5 (full)
Each loop:   u = kp * e + ki * integral(e) + kd * d(e)/dt      the integral is clamped (anti-windup) and frozen while the output is saturated; the derivative is of the filtered error
and the three demands are turned into one action by a fixed arbitration (`choose`): stop-and-turn or reverse if the speed demand is <= 0, else climb / descend if the altitude demand is large,
else turn if the yaw demand is large, else slow if the speed demand is small, else straight.

PARAMS (name: default): kp_yaw 0.9, ki_yaw 0.01, kd_yaw 4.0, kp_alt 0.08, ki_alt 0.001, kd_alt 0.6, kp_spd 1.0, ki_spd 0.05, kd_spd 0.0,
                         avoid_yaw 60 (degrees of heading offset at a fully blocked front), avoid_alt 120 (units of altitude offset), block_from 0.3 (front reading where avoidance and braking start).
"""
import math

import numpy as np

NAMES = ['kp_yaw', 'ki_yaw', 'kd_yaw', 'kp_alt', 'ki_alt', 'kd_alt', 'kp_spd', 'ki_spd', 'kd_spd', 'avoid_yaw', 'avoid_alt', 'block_from']
DEFAULT = np.array([0.9, 0.01, 4.0, 0.08, 0.001, 0.6, 1.0, 0.05, 0.0, 60.0, 120.0, 0.3])
LOW = np.array([0.1, 0.0, 0.0, 0.01, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.1])
HIGH = np.array([3.0, 0.2, 12.0, 0.4, 0.02, 3.0, 3.0, 0.5, 2.0, 120.0, 300.0, 0.7])

YAW_MAX, CLIMB_MAX = 25.0, 4.0            # the environment's fastest turn (hover) and climb rate
SPEED_FULL, SPEED_REVERSE = 5.0, -2.5
# the forward speed each action gives (units per step): straight, left, right, climb, descend, hover-left, hover-right, slow, reverse
ACTION_SPEED = [5.0, 5.0, 5.0, 4.0, 4.0, 0.0, 0.0, 2.5, -2.5]
FRONT, LEFT, RIGHT = (0, 1, 2), (1, 3, 5), (2, 4, 6)
UP_RAYS, DOWN_RAYS = (14, 15, 16, 26), (20, 21, 22, 27)


class PID:
    """u = kp e + ki sum(e) + kd (e - e_prev); output clamped to +-limit; the integral is clamped and does not grow while the output is saturated in the same direction (anti-windup)."""

    def __init__(self, kp, ki, kd, limit, i_limit=None, alpha=0.5):
        self.kp, self.ki, self.kd, self.limit, self.alpha = kp, ki, kd, limit, alpha
        self.i_limit = limit if i_limit is None else i_limit
        self.reset()

    def reset(self):
        self.integral, self.prev, self.d_filt = 0.0, None, 0.0

    def __call__(self, error):
        d = 0.0 if self.prev is None else error - self.prev
        self.prev = error
        self.d_filt = self.alpha * d + (1.0 - self.alpha) * self.d_filt                    # a little smoothing: the readings are noisy in dust and wind
        lim = self.i_limit / max(self.ki, 1e-9)                                            # the integral state that gives an I-term of i_limit
        trial = max(-lim, min(lim, self.integral + error))                                 # this step's error is included in this step's output
        u_unsat = self.kp * error + self.ki * trial + self.kd * self.d_filt
        u = max(-self.limit, min(self.limit, u_unsat))
        if u == u_unsat or (u_unsat * error) < 0:                                          # anti-windup: keep the new integral only if the output is not saturated (or the error pulls it back)
            self.integral = trial
        return u


class PidController:
    """act(state) -> an action 0..8; call reset() after a respawn (the loops forget). The state is the environment's get_state() list (37 or 65 numbers; only the first 34 are used)."""

    def __init__(self, params=None):
        p = DEFAULT if params is None else np.asarray(params, dtype=float)
        (self.kp_yaw, self.ki_yaw, self.kd_yaw, self.kp_alt, self.ki_alt, self.kd_alt, self.kp_spd, self.ki_spd, self.kd_spd,
         self.avoid_yaw, self.avoid_alt, self.block_from) = [float(v) for v in p]
        self.yaw = PID(self.kp_yaw, self.ki_yaw, self.kd_yaw, YAW_MAX, i_limit=10.0)
        self.alt = PID(self.kp_alt, self.ki_alt, self.kd_alt, CLIMB_MAX, i_limit=2.0)
        self.spd = PID(self.kp_spd, self.ki_spd, self.kd_spd, SPEED_FULL - SPEED_REVERSE, i_limit=3.0)
        self.v_meas = SPEED_FULL
        self.last = dict(yaw=0.0, alt=0.0, spd=0.0)
        self.n_inputs = 65

    def reset(self):
        self.yaw.reset(); self.alt.reset(); self.spd.reset()
        self.v_meas = SPEED_FULL

    def demands(self, s):
        rays = s[:28]
        front = max(rays[i] for i in FRONT)
        block = min(1.0, max(0.0, (front - self.block_from) / (1.0 - self.block_from)))      # 0 = clear ahead, 1 = touching
        # yaw: the goal bearing, plus a push away from the nearer side while something is ahead (a reference offset, so the loop still does the turning)
        lft, rgt = max(rays[i] for i in LEFT), max(rays[i] for i in RIGHT)
        bearing = s[28] * 180.0
        side = (1.0 if rgt > lft else -1.0) if abs(rgt - lft) > 0.02 else (1.0 if bearing > 0 else -1.0)       # +1: turn left (the right side is nearer; on a tie, toward the goal)
        avoid = self.avoid_yaw * block * side
        e_yaw = bearing + avoid
        # altitude: the goal's height above the drone, plus a push toward the freer of up and down while something is ahead
        dist = s[31] * 600.0
        dz = dist * math.sin(math.radians(s[30] * 90.0))
        free_up, free_dn = 1.0 - max(rays[i] for i in UP_RAYS), 1.0 - max(rays[i] for i in DOWN_RAYS)
        e_alt = dz + self.avoid_alt * block * (1.0 if free_up >= free_dn else -1.0) * min(1.0, abs(free_up - free_dn) * 3.0)
        # speed: slow down for what is ahead; the error is against what we commanded last
        v_ref = SPEED_FULL * (1.0 - block)
        if front > 0.9:
            v_ref = SPEED_REVERSE * min(1.0, (front - 0.9) / 0.1)                               # nearly touching: back away
        u_yaw = self.yaw(e_yaw)
        u_alt = self.alt(e_alt)
        u_spd = self.spd(v_ref - self.v_meas)
        v_cmd = max(SPEED_REVERSE, min(SPEED_FULL, self.v_meas + u_spd))
        self.last = dict(yaw=u_yaw, alt=u_alt, spd=v_cmd)
        return u_yaw, u_alt, v_cmd

    @staticmethod
    def choose(u_yaw, u_alt, v_cmd):
        """round the three demands to one of the 9 actions"""
        if v_cmd < -1.25:
            return 8                                                  # reverse
        if v_cmd < 1.25:
            return 5 if u_yaw >= 0 else 6                             # (almost) stopped: turn on the spot
        if abs(u_alt) > 1.5:
            return 3 if u_alt > 0 else 4
        if abs(u_yaw) > 5.0:
            return 1 if u_yaw > 0 else 2
        if v_cmd < 3.75:
            return 7                                                  # slow
        return 0

    def act(self, state, rng=None):
        s = [float(v) for v in state[:34]]
        a = self.choose(*self.demands(s))
        self.v_meas = ACTION_SPEED[a]                                 # what the environment will actually fly this step
        return a
