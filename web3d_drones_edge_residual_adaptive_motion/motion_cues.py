"""
Motion cues for the brain, and a penalty for things that close in fast.

The brain of web3d_drones_edge_residual_adaptive sees 28 range readings (0 = far, 1 = touching) but only the CURRENT ones: it cannot tell whether an obstacle is standing still or
coming toward it. This module adds what insects use for that, the "looming" cue (the rate at which things grow in the field of view):

  1. get_state() gets 28 more numbers: how much each ray reading changed since the last step (positive = something got closer), times DELTA_GAIN and clipped to [-1, 1].
     They are zero right after a start or a respawn (a teleport is not motion). They sit at the END of the state, so the earlier indices (28 = bearing, 30 = elevation, ...) do not move.
  2. A small reward penalty (approach_penalty) for rays that are already fairly near (reading above APPROACH_FROM) and are closing FASTER than the drone's own forward flight can explain
     (its top speed closes a ray by about 5 / 150 = 0.033 per step: APPROACH_FREE). So flying straight at a wall is not punished by this term, but something that comes toward the drone, or a
     turn that swings a ray onto a near obstacle, is. It fades out near the goal like the proximity penalty does. It is a hand-set heuristic: switch it off with approach_penalty = 0.

install_motion_cues(cls) adds this to an environment class (flags on the instance: motion_cues, approach_penalty; both off by default, so an untouched env behaves exactly as before).
This file is shared: web3d_drones_edge_residual_adaptive_motion/server_edge_residual_adaptive_motion.py imports it and build_trainer.py pastes the same code into the standalone Kaggle trainer
(tests/selftest_motion.py checks that the copies are identical).
"""
import numpy as np

N_RAYS = 28                    # the number of rangefinder rays of the drone
DELTA_GAIN = 10.0              # the change of a reading per step is about 0.03-0.05 for a closing obstacle; times 10 that is about 0.3-0.5, inside [-1, 1]
APPROACH_FROM = 0.4            # only rays that already read above this (an obstacle within 90 units) count for the approach penalty
APPROACH_FREE = 0.035          # closing speed (reading change per step) that the drone's own top speed explains: 5.0 / 150
APPROACH_SCALE = 0.06          # closing faster than APPROACH_FREE by this much gives the full penalty
_GOAL_RADIUS, _FADE_DIST = 30.0, 120.0     # the same numbers as GOAL_RADIUS and PROXIMITY_FADE_DIST of drone_env: the penalty fades out within this distance of the goal


def install_motion_cues(cls):
    """Add the motion cues and the approach penalty to environment class `cls` (call once per class; wrapping is idempotent along the inheritance chain)."""
    cls.motion_cues, cls.approach_penalty, cls.ray_delta = False, 0.0, np.zeros(N_RAYS)
    if not getattr(cls._update_sensors, '_motion', False):
        orig = cls._update_sensors

        def _update_sensors(self):
            old = np.asarray(self.signals, dtype=float) if getattr(self, 'signals', None) is not None else None
            orig(self)
            new = np.asarray(self.signals, dtype=float)
            if old is None or getattr(self, '_teleport', True) or len(old) != len(new):
                self.ray_delta = np.zeros(len(new))                              # a start or a respawn: no motion to report
            else:
                self.ray_delta = np.clip(new - old, -1.0, 1.0)                  # positive: that ray's obstacle got closer
            self._teleport = False

        _update_sensors._motion = True
        cls._update_sensors = _update_sensors
    if not getattr(cls.reset, '_motion', False):
        orig_reset = cls.reset

        def reset(self, *a, **k):
            self._teleport = True                                                # the next sensor reading follows a teleport
            return orig_reset(self, *a, **k)

        reset._motion = True
        cls.reset = reset
    if not getattr(cls.get_state, '_motion', False):
        orig_state = cls.get_state

        def get_state(self):
            s = orig_state(self)
            if not self.motion_cues:
                return s
            return list(s) + np.clip(self.ray_delta * DELTA_GAIN, -1.0, 1.0).tolist()

        get_state._motion = True
        cls.get_state = get_state
    if not getattr(cls._compute_reward, '_motion', False):
        orig_reward = cls._compute_reward

        def _compute_reward(self, prev_dist, action):
            result = orig_reward(self, prev_dist, action)
            if self.approach_penalty > 0 and not self.is_colliding and self._steps_to_goal > 0:     # (_steps_to_goal is 0 right after a goal: that reward is left alone)
                sig = np.asarray(self.signals)
                near = sig > APPROACH_FROM
                if near.any():
                    closing = float(self.ray_delta[near].max()) - APPROACH_FREE
                    if closing > 0:
                        relief = min(1.0, max(0.0, (self.distance - _GOAL_RADIUS) / _FADE_DIST))
                        self.reward -= relief * self.approach_penalty * min(1.0, closing / APPROACH_SCALE)
            return result

        _compute_reward._motion = True
        cls._compute_reward = _compute_reward
