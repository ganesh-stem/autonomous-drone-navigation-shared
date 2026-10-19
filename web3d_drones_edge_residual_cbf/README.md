# web3d_drones_edge_residual_cbf

Standalone Kaggle trainer only (no viewer yet) combining **residual SAC on a PID baseline** with a **real
control-barrier-function shield stacked on top**, built from
[`edge_v3_9_residual_pid_sac.py`](../web3d_drones_edge_residual/edge_v3_9_residual_pid_sac.py) and
[`edge_v3_10_cbf_shield.py`](../web3d_drones_edge_cbf/edge_v3_10_cbf_shield.py).

## What changed from edge_v3_9

Two independent mechanisms, combined rather than run separately:

1. The actor's logits still get `PID_WEIGHT * pid_bias(s)` added on top of the network's own output, exactly as in
   `edge_v3_9` (a *soft* bias toward the goal).
2. `edge_v3_10`'s `shield_mask()` -- the real discrete-time CBF condition `h_next >= (1 - alpha) * h_now`, not just
   a fixed-margin check -- filters which actions are allowed at all, both during training exploration and at
   evaluation. **On by default here** (`edge_v3_9` defaults to no shield).

Targets `edge_v3_9`'s calm-air weak spot (13.3 collisions vs plain SAC's 0.0 per 1000 steps) a third way: instead
of changing what the soft PID bias does, this adds a hard filter after it -- a stray PID-induced action simply
cannot pass if it would fly the drone into something. The shield does not know the actor has a PID bias at all; it
only ever sees the final action distribution, so this is a direct combination of two already-built variants, not a
new algorithm.

## Run it

    python edge_v3_14_residual_cbf_sac.py

Paste into a Kaggle cell (GPU on) for a real 60-minute run, or run locally for a quick CPU smoke test
(`--max-steps 1600 --device cpu`). No viewer/server built for this one yet -- ask if you want one.
