# web3d_drones_edge_sac

Live 3D viewer for the drone flown by the **discrete Soft Actor-Critic (SAC)** brain trained in
[`../web3d_drones_edge/kaggle/v3/edge_v3_4_sac.py`](../web3d_drones_edge/kaggle/v3/edge_v3_4_sac.py) (9 actions,
wind-aware, 37 inputs) -- the same file used unchanged, no charging stations, no battery-aware state or reward.
This is the base all five sibling variants (`_conservative`, `_cbf`, `_residual`, `_riskadapt`, `_recurrent`) build
on top of.

## Why no charging stations this time

An earlier line of variants (`web3d_drones_edge_sac` + 6 siblings) added charging pads and, in one variant,
battery-aware state/reward. Testing showed the mechanic worked (a drone parked on a pad genuinely recharged), but
none of the trained brains reliably *sought out* a station -- even the one designed to. That whole line has been
moved to [`../web3d_drones_edge_battery_archive/`](../web3d_drones_edge_battery_archive/) for reference. This
folder and its siblings go back to the plain environment and focus purely on the algorithmic ideas themselves.

## Run it

    python server_edge_sac.py                 then open http://localhost:8011

`results/models/sac_wind_aware/brain_final.pth` is a copy of the **real 60-minute Kaggle-trained** checkpoint
(`../web3d_drones_edge/results_2/brain_final.pth`) -- this one is not a smoke test, it's the actual result.

## edge_v3_4_sac.py

The original standalone (one file, no project modules, needs only numpy and torch) Kaggle trainer, copied here
unchanged. Run with `!python edge_v3_4_sac.py` or paste into a Kaggle cell.
