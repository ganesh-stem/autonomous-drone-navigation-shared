# web3d_drones_edge_residual_adaptive

Standalone Kaggle trainer, a live 3D viewer (`server_edge_residual_adaptive.py`, port 8019) and a ROS 2 port (`../ros2_edge_residual_adaptive/`) for **residual SAC on a PID baseline, with an adaptive blend
weight**, built directly on [`edge_v3_9_residual_pid_sac.py`](../web3d_drones_edge_residual/edge_v3_9_residual_pid_sac.py).

## What changed from edge_v3_9

`edge_v3_9`'s `PID_WEIGHT` is a fixed constant (1.5) always added to the actor's logits. That variant's one
measured weakness was calm air: 13.3 collisions per 1000 steps vs plain SAC's 0.0 -- with nothing to avoid, the
fixed bias still nudges the drone around and can override a network correction that would have avoided a wall.

Here the weight is a per-state scalar instead of a constant: `_pid_scale_np(s, wmax, wmin)` looks at the closest of
the 28 rangefinder signals and returns `wmax` (default 1.5) when nothing is nearby, fading toward `wmin` (default
0.0) right next to an obstacle -- trusting the residual network fully for avoidance exactly when avoidance matters
most, and trusting the (obstacle-blind) PID baseline only when there is nothing to avoid.

Follows: *Adaptive Outer-Loop Control of Quadrotors via Reinforcement Learning* (arXiv:2605.16015), which modulates
a learned correction's trust by an estimated disturbance magnitude rather than a fixed blend weight -- here that
magnitude is approximated cheaply by obstacle proximity instead of a learned GRU predictor.

## Run it

    python edge_v3_12_residual_adaptive_sac.py

Paste into a Kaggle cell (GPU on) for a real 60-minute run, or run locally for a quick CPU smoke test
(`--max-steps 1600 --device cpu`). The live viewer is `python server_edge_residual_adaptive.py` (then http://localhost:8019).

## The world: roaming birds and patrol drones are obstacles (new)

The floating blocks of the sky stay where they are, marked by the colour-changing gates of the viewer. In addition, every floating block sends out its birds and patrol drones
(1-3 birds and 1-2 drones), and each of them is now a real obstacle of the simulation (`flyers.py`, shared by the server, the ROS 2 package and, pasted in, this trainer):
a small box (bird 9 x 9 x 4, patrol drone 10 x 10 x 3 units) that roams along its own long, smooth path, up to about 560 units from its block, at up to 1.0-2.2 units per step
(a bird) or 0.8-1.6 (a drone); the drone flies 5.0. A flyer's position is a pure function of its block, its number and the step number, so the simulator, the camera and the viewer all
compute the same positions. Rays, collisions, the safety filter and the aerial camera see them. A flyer that moves onto the drone shoves it aside and that counts as a collision.
Towers and the static floating blocks do not move.

- **Viewer server:** the flyers are obstacles by default, `--static-obstacles` removes them (the old world). Replays always show the static world (their files have no obstacle clock).
- **Trainer (`edge_v3_12_residual_adaptive_sac.py`):** it trains and evaluates in this world by default: the roaming flyers, ~20% fewer obstacles per chunk and towers kept
  18 units inside their chunk (the viewer's world). `--static-obstacles` removes the flyers, `--classic-world` restores the old density and tower placement; in a Kaggle cell put
  `static_obstacles=True` / `classic_world=True` into the SETTINGS block. `eval_v3.json` also holds `static_world_conditions` (the same brain in the static world) to compare with
  the tables of the earlier runs. The environment runs at about 2,000 steps per second with the flyers (5,800 without), so training is slower in steps per second.
  The shipped brain (`results/models/sac_residual_adaptive/`) was trained BEFORE this and has only ever seen a static world.
- **What the shipped brain does among the roaming flyers** (`../ros2_edge_residual_adaptive/tools/eval_moving_obstacles.py`, 5 world seeds x 4,000 steps per weather, same brain, numpy policy;
  saved in `../ros2_edge_residual_adaptive/results/moving_obstacles_eval.json`): goals per 1k steps 10.4 / 9.3 / 6.8 (calm / windy / gale) against 10.3 / 9.3 / 7.3 in the static world, and 0.48
  collisions per 1k steps overall (a collision in 3 of the 15 runs, 0.0 in the static world). The flyers are small, so the old brain mostly flies around them; retraining in this world is meant to
  make that reliable.
