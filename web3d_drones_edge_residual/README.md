# web3d_drones_edge_residual

Live 3D viewer + standalone trainer for a **residual SAC on a heuristic PID baseline**, built directly on plain
[`edge_v3_4_sac.py`](../web3d_drones_edge/kaggle/v3/edge_v3_4_sac.py) -- no charging stations, no battery-aware
state or reward (see [`web3d_drones_edge_sac/README.md`](../web3d_drones_edge_sac/README.md) for why).

## What changed from plain SAC

A fixed, hand-written "PID" gives every action a bias before the learned logits are added: it looks at the same
bearing-to-goal and elevation-to-goal signals already in the state (a proportional rule with a deadband -- correct
altitude first if it's off by more than a threshold, else correct heading, else fly straight) and adds
`PID_WEIGHT * pid_bias(s)` to the actor's own logits. The network only ever has to learn a *correction* on top of
that baseline, not the whole policy from scratch -- the same idea as residual RL on top of an existing PID
controller, adapted to this project's discrete action space with a simple heuristic stand-in for a real PID (this
project's own embedded C++ policy already ships a real one, in
`../web3d_drones_edge/embedded/flight_ctrl.hpp` -- the point being tested here is the residual-learning
architecture, not the baseline's quality).

Because the actor was trained to only ever be evaluated with that bias added, the viewer's brain loader
(`server_edge_residual.py`) reproduces the exact same `pid_bias()` and adds it back in at inference -- using the
checkpoint without it would fly a policy it was never actually trained to be.

## Run it

    python server_edge_residual.py                 then open http://localhost:8016

`results/models/sac_residual/brain_final.pth` is a **5-minute local CPU smoke-training** checkpoint (not a full
60-minute Kaggle GPU run) -- it proves the wiring works end-to-end, not that the brain flies well.

## edge_v3_9_residual_pid_sac.py

The standalone (one file, no project modules, needs only numpy and torch) Kaggle-style trainer, built by hand from
`edge_v3_4_sac.py`. Run with `!python edge_v3_9_residual_pid_sac.py` or paste into a Kaggle cell.
