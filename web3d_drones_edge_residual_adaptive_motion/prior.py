"""
The hand-made prior of the residual brain, written as a function with 12 numbers that an evolution strategy can tune.

The base folder (web3d_drones_edge_residual_adaptive) adds to the network's logits   pid_scale(s) * pid_bias(s):
  - pid_bias  : one-hot "goal rule" (climb / descend if the goal is off in height, else turn toward it, else straight);
  - pid_scale : 1.5 far from obstacles, fading to 0 when the closest ray reads 1.
That prior only knows the GOAL. It says nothing about which way to dodge. Here the same prior gets 12 numbers (THETA) and four extra terms, and with the
CLASSIC numbers it is EXACTLY the old prior (tests/selftest_motion.py checks that against the base trainer's functions):

   logits = scale * goal_onehot                                   the old rule; scale = w_min + (w_max - w_min) * (1 - closest)**fade
          + avoid  * (right_side - left_side)  on 'left'/'right'   turn away from the side that reads nearer, when something is ahead
          + slow   * ahead                     on 'slow'/'hover'   slow down / stop forward flight when the front is close
          + climb  * (free_above - free_below) * ahead  on 'climb'/'descend'
          + loom   * closing_ahead             on 'reverse', 'slow' and the turn away (needs the motion cues; 0 when the state has only 37 inputs)

THETA, in order (name: classic value):
  0 yaw_db     0.08   dead band of the bearing (turn only if |bearing| is above it)
  1 elev_db    0.15   dead band of the elevation
  2 w_max      1.5    scale far from obstacles
  3 w_min      0.0    scale right next to an obstacle
  4 fade       1.0    exponent of the fade: 1 = the old straight line, >1 keeps the goal rule alive longer, <1 drops it earlier
  5 ahead_from 0.35   the front reading above which the avoidance terms start
  6 avoid      0.0    how hard to turn away from the nearer side
  7 slow       0.0    how much to prefer slow / hover-turn when the front is close
  8 climb      0.0    how much to prefer the freer of 'above' and 'below'
  9 loom       0.0    how much a closing obstacle ahead pushes toward reverse / slow / turning away
 10 brake      0.0    how much to prefer 'reverse' when the front reads very close (above 0.85)
 11 side_w     1.0    weight of the 70-degree rays against the 20/40-degree rays when comparing the left and the right

All the maths is done on arrays of shape (B, n_inputs) with operators that numpy and torch share, so the SAME function runs in the server (numpy), in the evolution (numpy)
and inside the learning step of the trainer (torch tensors, so the actor loss sees the prior): prior_logits(s, theta) works on either and returns (B, 9).
This file does not import torch; build_trainer.py pastes it into the standalone trainer.
"""
import numpy as np

# indices into the state / action list
N_ACTIONS = 9                                  # 0 straight, 1 left, 2 right, 3 climb, 4 descend, 5 hover-left, 6 hover-right, 7 slow, 8 reverse
BEARING_IDX, ELEV_IDX = 28, 30                 # in get_state(): 28 rays, then bearing, -bearing, elevation, ...
DELTA_FROM = 37                                # 28 rays + 6 extras + 3 wind; the 28 motion cues (if on) follow at 37..64
# ray groups (the order of SENSOR_DIRS: 14 horizontal at yaw 0, 20, -20, 40, -40, 70, -70, 100, -100, 130, -130, 160, -160, 180; then 6 up-tilted, 6 down-tilted, straight up, straight down)
FRONT = (0, 1, 2)                              # yaw 0, +20, -20
LEFT_NEAR, RIGHT_NEAR = (1, 3), (2, 4)         # +20, +40  /  -20, -40
LEFT_SIDE, RIGHT_SIDE = (5,), (6,)             # +70 / -70
UP_RAYS, DOWN_RAYS = (14, 15, 16, 26), (20, 21, 22, 27)     # tilted-up front rays (yaw 0, 60, -60) + straight up; tilted-down + straight down

NAMES = ['yaw_db', 'elev_db', 'w_max', 'w_min', 'fade', 'ahead_from', 'avoid', 'slow', 'climb', 'loom', 'brake', 'side_w']
CLASSIC = np.array([0.08, 0.15, 1.5, 0.0, 1.0, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
LOW = np.array([0.02, 0.05, 0.5, 0.0, 0.5, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2], dtype=np.float64)
HIGH = np.array([0.30, 0.40, 3.0, 1.0, 3.0, 0.70, 4.0, 3.0, 3.0, 4.0, 3.0, 2.0], dtype=np.float64)

EVOLVED = CLASSIC.copy()                       # replaced by the result of evolve_prior.py (build_trainer.py embeds results/evolved_prior.json here)


def _mx(a, b):
    """element-wise maximum that works for numpy arrays and torch tensors alike"""
    return (a + b + abs(a - b)) * 0.5


def _group_max(s, idx):
    m = s[:, idx[0]]
    for i in idx[1:]:
        m = _mx(m, s[:, i])
    return m


def prior_logits(s, theta):
    """s: (B, n_inputs) numpy array or torch tensor; theta: the 12 numbers (list / array). Returns (B, 9) logits to ADD to the network's output."""
    yaw_db, elev_db, w_max, w_min, fade, ahead_from, avoid, slow, climb, loom, brake, side_w = [float(t) for t in theta]
    closest = _group_max(s, list(range(28))).clip(0.0, 1.0)
    scale = w_min + (w_max - w_min) * (1.0 - closest).clip(0.0, 1.0) ** fade           # (B,)
    bearing, elev = s[:, BEARING_IDX], s[:, ELEV_IDX]
    up, down = elev > elev_db, elev < -elev_db
    left, right = bearing > yaw_db, bearing < -yaw_db
    ones = s[:, 0] * 0.0 + 1.0
    # the old goal rule as five 0/1 columns (same priority: altitude first, then heading, else straight); & and ~ on booleans, then * 1.0 to get numbers
    rest = (~up) & (~down)
    g_climb = up * 1.0
    g_desc = ((~up) & down) * 1.0
    g_left = (rest & left) * 1.0
    g_right = (rest & (~left) & right) * 1.0
    g_str = (rest & (~left) & (~right)) * 1.0
    # avoidance terms, all multiplied by how much the front is blocked
    front = _group_max(s, FRONT)
    ahead = ((front - ahead_from) / max(1e-6, 1.0 - ahead_from)).clip(0.0, 1.0)
    lft = _mx(_group_max(s, LEFT_NEAR), side_w * _group_max(s, LEFT_SIDE)).clip(0.0, 1.0)
    rgt = _mx(_group_max(s, RIGHT_NEAR), side_w * _group_max(s, RIGHT_SIDE)).clip(0.0, 1.0)
    turn = (rgt - lft) * ahead                                                          # >0: the right is nearer, so turn left
    free_up = 1.0 - _group_max(s, UP_RAYS)
    free_dn = 1.0 - _group_max(s, DOWN_RAYS)
    vert = (free_up - free_dn) * ahead                                                  # >0: more room above
    if s.shape[1] >= DELTA_FROM + 28:
        closing = _group_max(s[:, DELTA_FROM:DELTA_FROM + 28], FRONT).clip(0.0, 1.0)    # the front rays that are getting nearer (cues are scaled by 10 and clipped to 1)
    else:
        closing = ones * 0.0
    very_near = ((front - 0.85) / 0.15).clip(0.0, 1.0)
    tl, tr = turn.clip(0.0, 1.0), (-turn).clip(0.0, 1.0)                                # how much to turn left / right to get away
    cols = [None] * N_ACTIONS
    cols[0] = scale * g_str
    cols[1] = scale * g_left + (avoid + loom * closing) * tl
    cols[2] = scale * g_right + (avoid + loom * closing) * tr
    cols[3] = scale * g_climb + climb * vert.clip(0.0, 1.0)
    cols[4] = scale * g_desc + climb * (-vert).clip(0.0, 1.0)
    cols[5] = slow * ahead * 0.5 + avoid * 0.5 * tl * very_near
    cols[6] = slow * ahead * 0.5 + avoid * 0.5 * tr * very_near
    cols[7] = slow * ahead + loom * closing * ahead
    cols[8] = brake * very_near + loom * closing * very_near
    if hasattr(s, 'new_zeros'):                                                          # a torch tensor
        import torch
        return torch.stack(cols, dim=1)
    return np.stack(cols, axis=1)


def prior_logits_one(s, theta):
    """one state (1-D list/array) -> a (9,) float32 vector, for acting one step at a time"""
    return prior_logits(np.asarray(s, dtype=np.float64).reshape(1, -1), theta)[0].astype(np.float32)
