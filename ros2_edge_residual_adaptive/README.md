# ros2_edge_residual_adaptive: the adaptive-weight residual SAC drone as a ROS 2 system

`web3d_drones_edge_residual_adaptive/server_edge_residual_adaptive.py` (one Python server: the world, the brain, the camera, the telemetry and the viewer in one process) rebuilt as ROS 2 nodes on Jazzy, with the optimisations of
`ros2_edge_drone`. The brain is the same: a trained SAC actor plus a PID baseline whose weight fades near obstacles. The 3D viewer is the same city viewer (roads, parks,
fountains, tennis club, the goal "stargate", colour-shifting obstacle gates, roaming birds and drones, buildings in real-world facade colours, cars that drive the roads with traffic lights and turns, and a modern quadcopter), fed by ROS instead of by the server. The traffic follows the time of day: rush hour at dusk, moderate by day, almost empty at night. The cars are part of the simulation (`city_world.py`, shared with the server and copied by `tools/build_core.py`): the camera node and the detector see the same driving and parked cars the viewer draws (the camera counts them as vehicles). In replay mode no traffic is shown. No other project of the repository is changed.

**Read this first: what was and was not tested.** ROS 2 is not installed where this was written, so the system was never started with `ros2 launch`, and `colcon build` was not run. What was run is listed
under [What was checked](#what-was-checked): the real node classes against a small in-process stand-in for rclpy (`tools/offline_selftest.py`), and the real `web_bridge` serving the real viewer to a headless browser.
The nodes are `ros2_edge_drone`'s nodes (which ran under ROS 2 Jazzy) with the brain, the viewer messages and the names changed. The first run on your ROS 2 machine is the real test; `ISSUES_AND_FIXES.md` is where to write down what it finds.

## How the brain flies

```
logits(s) = actor(s) + pid_scale(s) * pid_bias(s)          action = argmax(logits)
pid_bias(s)   one-hot: climb / descend / turn left / turn right / straight, from the bearing and elevation to the goal (a proportional controller with a dead band)
pid_scale(s)  1.5 when nothing is near, fading to 0.0 right next to an obstacle (from the closest of the 28 rangefinder signals)
```

Far from obstacles the simple controller steers and the network corrects it. Next to an obstacle the network decides alone. After a respawn the brain flies 90 random actions. This is `residual_policy.py`, in numpy, no torch.

| Function in the server | ROS 2 node here |
|---|---|
| World, wind, gusts, battery, noisy rangefinders, 9 actions (`EdgeDroneEnvV2`) | `sim_node` (runs `core.py`, generated from the same sources) |
| The adaptive residual SAC brain | `brain_node` (`residual_policy.py`, `weights/residual_adaptive_actor.npz`) |
| Synthetic RGB / thermal / height camera (`aerial.py`) | `camera_node` |
| Object detector, 189k parameters | `detector_node` (torch, CPU) |
| MAVLink v2 telemetry and ground station | `telemetry_node` and `gcs_node` |
| KPIs per weather | `kpi_node` |
| Replay files (`replay.jsonl`) | `recorder_node` (record) and `web_bridge` (play in the viewer) |
| The viewer and its WebSocket | `web_bridge` (serves `web/`, streams the topics, sends the browser's commands back as `/edge_residual/control`) |

## How the nodes talk

```
              /edge_residual/action [step, action]
        +-----------------------------------------+
        v                                         |
   sim_node ---- /obs [step, respawns, 37 values] ---> brain_node (adaptive residual SAC)
   (world)   \--- /pose /goal /wind /battery /rays /status /status_slow
      |            |                          \----> telemetry_node --/mavlink--> gcs_node --/gcs-->
      |            |--- /status_slow ---> camera_node --/camera/tensor, /rgb, /thermal, /truth--> detector_node --/detections, /camera/annotated-->
      |            \--- /status_slow ---> kpi_node --/kpi-->        \--> recorder_node (replay.jsonl)
      +<------------------------------------------------ /control (JSON) <--- web_bridge <--- browser
```

The topics are the ones of `ros2_edge_drone` under the namespace `/edge_residual/` (so both projects can exist side by side, but never run both at once: they share the ROS network and would want the same sim resources).
The simulator is in lockstep with the brain: it publishes the observation of step k, the brain answers `[k, action]`, and the action is applied to the state it was chosen for. A late answer holds the last action and
counts as `stale`. There is no `/shield_mask` here: this brain was trained without the safety filter.

| Topic | Type | Meaning |
|---|---|---|
| `/edge_residual/obs` | Float32MultiArray | `[step, respawns, 37 observation values]` |
| `/edge_residual/action` | Int32MultiArray | `[step, action]` (0 straight, 1 left, 2 right, 3 climb, 4 descend, 5 hover-left, 6 hover-right, 7 slow, 8 reverse) |
| `/edge_residual/pose`, `/goal`, `/wind`, `/battery` | PoseStamped, PointStamped, Vector3Stamped, BatteryState | only built while somebody subscribes |
| `/edge_residual/rays` | Float32MultiArray | 28 rangefinder rays, `[end x, end y, end z, signal]` each |
| `/edge_residual/status`, `/status_slow` | String (JSON) | everything else, every step / every `aux_every` steps |
| `/edge_residual/camera/*`, `/detections` | Image, String | the onboard views and what the detector found |
| `/edge_residual/mavlink`, `/gcs` | UInt8MultiArray, String | the radio bytes, and what the ground station made of them |
| `/edge_residual/kpi` | String (JSON) | goals and collisions per 1k steps, energy per goal, in the current weather |
| `/edge_residual/control` | String (JSON) | `{"cmd": "weather" \| "light" \| "sensors" \| "pause" \| "resume" \| "speed" \| "new_goal" \| "auto_weather" \| "auto_light", "value": ...}` |

## Set up (once, Ubuntu / WSL with ROS 2 Jazzy)

```
python3 -m venv ~/ros2_ws/venv_edge
~/ros2_ws/venv_edge/bin/python -m pip install numpy opencv-python-headless pymavlink websockets orjson     # orjson is optional (faster JSON)
# torch is only needed by detector_node: ~/ros2_ws/venv_pytorch (the launch file puts both venvs on the python path; EDGE_VENV_SITE overrides the paths)
cp -r ros2_edge_residual_adaptive/edge_residual_adaptive ~/ros2_ws/src/
cd ~/ros2_ws && colcon build --symlink-install --packages-select edge_residual_adaptive && source install/setup.bash
```

Optional, 2.3 times faster simulation: `sudo apt install g++ python3-dev pybind11-dev`, then `bash tools/build_fastray.sh` and build again. Without it everything runs on numpy with identical results.

## Run

```
ros2 launch edge_residual_adaptive edge_residual_adaptive.launch.py                    # then open http://localhost:8019 (also from Windows)
ros2 launch edge_residual_adaptive edge_residual_adaptive.launch.py weather:=gale light:=night
ros2 launch edge_residual_adaptive edge_residual_adaptive.launch.py fast:=true         # the simulator steps the moment the brain answers
ros2 launch edge_residual_adaptive edge_residual_adaptive.launch.py record:=true drop_rate:=0.02 corrupt_rate:=0.01
ros2 launch edge_residual_adaptive edge_residual_adaptive.launch.py auto_weather:=0 auto_light:=0
python3 tools/evaluate_ros.py --steps 6000            # second terminal, while a launch with fast:=true auto_weather:=0 auto_light:=0 runs
```

The weather changes every 30 s and the light every 20 s by default (as in the server). Arguments: `weather`, `world_seed`, `sim_hz` (60, keep it at 100 or below), `fast`, `aux_every` (2), `wind_comp`
(0: the brain was trained without it), `light`, `weights`, `greedy`, `pid_max` / `pid_min` (-1 = the values stored in the weights file), `auto_weather`, `auto_weather_mode`, `auto_light`, `camera_hz`, `port` (8019), `record`,
`out_dir` (default `~/edge_residual_logs`: replays and `kpi_log.csv`), `drop_rate`, `corrupt_rate`, `moving_obstacles` (true: the birds and patrol drones are real, roaming obstacles; false: only the static boxes, the world the brain was trained in).

**A different brain:** `python tools/export_residual_npz.py path/to/brain_final.pth --out my.npz`, then `weights:=/full/path/my.npz`. The tool reads the checkpoint without torch. It writes the PID weights (`--pid-max`, `--pid-min`) into the file.
**Changed the training sources?** `python tools/build_core.py` regenerates `core.py`. It also patches the obstacle density to the viewer server's (about 20% fewer obstacles per chunk than the training default) and keeps every tower 18 units inside its chunk, so that no building stands on the viewer's roads or sidewalks (the server applies the same shift).

## What was optimised, and what was measured

| What | Where | Status |
|---|---|---|
| Rangefinder ray casting in C++ (pybind11), same arithmetic, `-ffp-contract=off` | `tools/build_fastray.sh`, used by `core.py` when built | **Measured here**: world + policy 7,300 to 7,800 steps/s with numpy, 16,800 to 18,400 with C++ (about 2.3 times). A SHA-256 over every state, reward and position of 50,000 steps in all 5 weathers is identical both ways (`tools/check_fastray.py`, `results/fastray_check.txt`) |
| World regenerated from the current optimised sources (box cache per chunk window, `min`/`max` instead of `np.clip`, boxes beyond sensor range skipped) | `core.py` | **Checked here**: identical to the server's world, state by state, 6,000 steps in 3 weathers |
| Policy: weights stored transposed and contiguous, preallocated buffers, the PID bias added to one logit | `residual_policy.py` | **Measured here**: 17.6 us to 14.1 us per decision (about 20%, other runs 13 to 20%). Same action as the reference on all 4,000 real states tried, logits equal to within 1.5e-5 (the self-test's largest difference over 600 states) |
| No torch in the brain node: weights are a numpy file, and a torch-free reader exports them | `export_residual_npz.py` | Checked: the real checkpoint exported, 37 inputs, 256 / 256 hidden, 9 actions |
| Events executor (waits for events in C++), the simulator keeps the normal executor on its timer | `spin_util.py` | Inherited from `ros2_edge_drone`, where it was measured (see its README). **Not re-measured here** |
| No parameter services in any node | all nodes | Inherited, not re-measured |
| `aux_every`: pose, goal, wind, battery, rays and `status_slow` go out every 2nd step; camera, telemetry, KPI and the viewer read `status_slow` | `sim_node`, others | Inherited, not re-measured |
| Messages are only built while somebody subscribes | `sim_node` | Inherited, not re-measured (the self-test shows `/rays` is only built because the bridge subscribes) |
| `fast:=true`: the simulator steps when the brain's answer arrives | `sim_node` | Inherited, not re-measured under ROS. Offline: identical flight to the plain loop, 0 late answers |
| MAVLink parsed per frame instead of per byte | `telemetry_lib.py` | Inherited, not re-measured. Offline: 7,800 frames in 6,000 steps, 0 CRC errors, 0 lost |
| One reused message in the brain node; the bridge parses the status once per send, and encodes with orjson when it is installed | `brain_node`, `web_bridge` | Not measured separately |

The in-process bus of the self-test runs about 3,000 steps/s. That number is the Python logic of the nodes without any transport: it says nothing about ROS 2 speeds, which are limited by the messages between processes
(about 100 steps/s is safe; `ros2_edge_drone` measured 190 to 246 steps/s with `fast:=true` on its machine).

## What was checked

`python tools/offline_selftest.py` (output saved in `results/offline_selftest.txt`, all 37 checks pass):
1. `core.py` against `EdgeDroneEnvV2` of the server (with its obstacle density): identical state, position and respawns for 6,000 steps in calm, windy and gale-with-dust air.
1b. No tower stands within 17 units of a chunk border (where the viewer's roads and sidewalks are): 364 towers in 25 x 25 chunks, the closest is 18.0 away.
2. The optimised policy against the straightforward one (above).
3. `sim_node` + `brain_node` + `kpi_node` + `telemetry_node` + `gcs_node` + `web_bridge` through messages, 6,000 steps in windy air, fast mode: **the same flight as the plain loop of the server** (position to the last digit,
   56 goals, 0 collisions, 0 respawns, 9.34 goals per 1k steps; the training evaluation of this brain says 8.4 goals and 4.0 collisions per 1k in windy air, over other skies). And in timer mode, 1,200 steps.
4. Every field the HUD reads (`web/js/ui.js`, `hud()`) is in `web_bridge.state_message()`, and the world messages (obstacles, trees, vehicles) are built.
5. MAVLink: 7,800 frames, no CRC errors, none lost.
6. The cars of the city: `sim_node` publishes the traffic, `web_bridge` turns it into the "cars" message, the targets message carries the parked cars, the package copy of `city_world.py` is identical to the server's, and the camera counts the driving and parked cars really there.
7. Roaming birds and patrol drones as obstacles (`flyers.py`): 146 floating blocks send out 209 birds and 181 patrol drones, peak horizontal speed 2.20 (birds) and 1.60 (drones) units/step against the limits, they stay in the flight band, the velocity sent
   to the viewer matches the real motion; `sim_node`'s obstacle clock equals its step number, `camera_node` and `web_bridge` learn it from `/edge_residual/flyers`, the camera sees exactly the boxes the simulator collides with, every roaming collision box near
   the drone is exactly where the viewer's "flyers" message puts a bird or drone, a 1,500-step flight among them reaches goals, and the copies of `flyers.py` (package and Kaggle trainer) are identical to the server's.

Also run once, by hand (not saved as a test): the real `web_bridge` serving the viewer to headless Chromium while the simulated drone flew. The HUD filled in (steps, goals, wind, altitude, the brain's name), the city rendered, and
choosing "Calm" in the weather box on the page changed the weather of the simulator and came back to the page.

**Not checked:** `colcon build`, `ros2 launch`, DDS transport and the executors, `camera_node` and `detector_node` (they need ROS images and torch), the radio fault options, `record:=true` and replay playback under ROS, and the
speeds above under real ROS.

## Limits, stated plainly

- One drone. The simulator is a ROS node running the training project's world, not Gazebo.
- The brain node runs this one kind of policy (a SAC actor with the PID bias). Another brain type needs its own small numpy policy.
- The lockstep loop between the simulator and the brain is limited by Python and the messages: about 100 steps per second is safe.
- The camera images are synthetic and the detector runs on the CPU with torch. Nothing was run on a Jetson or a real flight controller.
- The viewer loads three.js from a CDN (`web/index.html`), so the browser needs internet.
- The trees, cars and people of the viewer are drawn from the simulated world and kept off roads and scenery by the viewer; the detector's camera image does not know about the roads.
- `core.py` is generated: change the sources and run `tools/build_core.py`, do not edit it (it also copies `city_world.py` and `flyers.py` into the package).
- The shipped brain was trained in a static world. `python3 tools/eval_moving_obstacles.py` (no ROS, no torch) flies it among the roaming birds and drones: goals per 1k steps are about the same (10.4 / 9.3 / 6.8 in calm / windy / gale air
  against 10.3 / 9.3 / 7.3 static) and there are 0.48 collisions per 1k steps overall instead of 0.0 (`results/moving_obstacles_eval.json`). Train a new brain in the roaming world with the Kaggle trainer (it does that by default) and use it with `weights:=`.

## Folder

```
ros2_edge_residual_adaptive/
  edge_residual_adaptive/          the ROS 2 package (copy it into ~/ros2_ws/src)
    edge_residual_adaptive/        nodes: sim, brain, camera, detector, telemetry, gcs, kpi, recorder, web_bridge; core.py (generated), residual_policy.py, spin_util.py ...
    launch/  weights/  web/        the launch file, residual_adaptive_actor.npz + detector.pt, the viewer (index.html, app.js, js/*.js)
  tools/                           build_core.py, build_fastray.sh, check_fastray.py, export_residual_npz.py, evaluate_ros.py, offline_selftest.py
  results/                         offline_selftest.txt, fastray_check.txt
  ISSUES_AND_FIXES.md              what went wrong while building it
```
