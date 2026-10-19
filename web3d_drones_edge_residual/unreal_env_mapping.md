# Unreal environment (MyProject2 / AutonomousDrone level)

Built from this project's own world generator (`DroneEnv(seed=1234)`, chunks -8..7 = a 4 km x 4 km area), so the
obstacle layout is the same one the brain was trained on. Every obstacle is placed at its exact box centre/size.

## Coordinate mapping (sim -> Unreal)

    UE.x = sim.x * 100      (1 sim unit = 1 m = 100 uu)
    UE.y = -sim.y * 100     (y is flipped: sim is right-handed / CCW yaw, UE is left-handed)
    UE.z = sim.z * 100
    UE yaw (deg) = -sim yaw (deg)

With the y flip, "turn left" in the sim is a left turn in Unreal too (no mirrored world).

## What replaced what

| sim box                        | Unreal                                                        |
|--------------------------------|---------------------------------------------------------------|
| tower (z0 = 0)                 | skyscraper (h >= 160 m) / brick or concrete building, lit windows; some with rooftop penthouse + antenna mast |
| floating block ("pink cubes")  | blimp (with gondola) or hot-air balloon (with basket)         |

Blimps/balloons are ellipsoids inscribed in the original box, so they are slightly smaller than the training box.

## Actor tags / folders

- All obstacles: tag `obstacle` (also `building`, `rooftop`, `blimp`, `balloon`); folders `Obstacles/...`
- `GoalMarker` (tag `goal`, no collision, 30 m radius sphere) and `DroneStart` (PlayerStart at sim (0, 0, 80)), folder `Drone`.
- Assets: `/Game/DroneEnv/` (`M_Facade`, `M_Solid` and their material instances).

The ground is the level's existing `Floor` at z = 0 (sim ground plane; the sim crashes at z - 6 < 0).

## Running without the Unreal editor

- `python headless_sim.py` -- brain vs. this exact world, no Unreal, ~1500-2700 steps/s. Works now.
- `python ue_bridge.py` -- same Python sim, drawn live in Unreal (drone `Drone`, `GoalMarker`, `ChaseCam` are moved through
  the Remote Control HTTP API, port 30010; object paths are in `ue_actors.json`). **Untested.**
- Packaging the level as a standalone `.exe` failed: UnrealBuildTool rejects the only MSVC toolchain installed (14.39).
  Fix: Visual Studio Installer -> Modify VS 2022 -> Individual components -> "MSVC v143 - VS 2022 C++ x64/x86 build tools
  (latest)" (14.44 or newer), then run
  `RunUAT.bat BuildCookRun -project=<MyProject2.uproject> -platform=Win64 -clientconfig=Development -build -cook -stage -pak -archive -archivedirectory=<out>`
  from `E:\InstalledSoftwares\Unreal\UE_5.8\Engine\Build\BatchFiles`, and start the game with `-RCWebControlEnable`.
- Project changes made for this: `RemoteControl` plugin enabled in `MyProject2.uproject`; `GameDefaultMap` now
  `/Game/AutonomousDrone` and `WebControl.EnableServerOnStartup=1` in `Config/DefaultEngine.ini`.
