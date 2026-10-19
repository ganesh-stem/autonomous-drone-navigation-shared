# web3d_drones_edge_residual_potential

Standalone Kaggle trainer only (no viewer yet) for **residual SAC on an obstacle-aware potential-field baseline**,
built directly on [`edge_v3_9_residual_pid_sac.py`](../web3d_drones_edge_residual/edge_v3_9_residual_pid_sac.py).

## What changed from edge_v3_9

`edge_v3_9`'s baseline only looks at bearing/elevation to the goal -- it is blind to obstacles, a one-hot vote for
a single "best" action with no idea a wall is there. Here the baseline is a simple potential field instead: a
goal-attraction vector (same bearing/elevation as before) combined with an obstacle-repulsion vector built from the
same 28 rangefinder signals already in the state (each ray pushes back along its own direction, weighted by
closeness squared). Every action's bias becomes a continuous score -- how well that action's own movement direction
aligns with the combined vector -- rather than a single preferred action getting the whole weight.

Targets the same weakness as `web3d_drones_edge_residual_adaptive` (edge_v3_9's calm-air collisions, 13.3 vs plain
SAC's 0.0 per 1000 steps) from a different angle: instead of trusting the baseline less near obstacles, this makes
the baseline itself aware obstacles exist, at a fixed blend weight.

General idea from: *Residual Reinforcement Learning for Robot Control*, Johannink et al. (arXiv:1812.03201).

## Run it

    python edge_v3_13_residual_potential_sac.py

Paste into a Kaggle cell (GPU on) for a real 60-minute run, or run locally for a quick CPU smoke test
(`--max-steps 1600 --device cpu`). No viewer/server built for this one yet -- ask if you want one.
