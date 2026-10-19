"""
The whole drone system in one launch: the adaptive-weight residual SAC brain flying the drone in weather, as ROS 2 nodes.

    ros2 launch edge_residual_adaptive edge_residual_adaptive.launch.py
    ros2 launch edge_residual_adaptive edge_residual_adaptive.launch.py weather:=gale light:=night
    ros2 launch edge_residual_adaptive edge_residual_adaptive.launch.py fast:=true            # the simulator steps the moment the brain answers (no fixed rate)
    ros2 launch edge_residual_adaptive edge_residual_adaptive.launch.py record:=true drop_rate:=0.02 corrupt_rate:=0.01
    ros2 launch edge_residual_adaptive edge_residual_adaptive.launch.py auto_weather:=0 auto_light:=0       # both automatic changes off (ON by default: weather every 30 s, light every 20 s)

Then open  http://localhost:8019  (also from Windows, through WSL's localhost forwarding).

Arguments: weather (calm | breeze | windy | gale | gale_dusty), world_seed, sim_hz (simulation steps per second, keep it at 100 or below), fast (step when the brain
answers), aux_every (pose / goal / wind / battery / rays / status_slow go out every N steps, default 2), wind_comp (share of the wind push the flight controller
cancels; the brain was trained with 0), light (day | dusk | night), weights (an exported residual_adaptive_actor.npz), greedy (most probable action, or false to sample),
pid_max / pid_min (override the PID bias weights stored in the weights file, -1 = use the file), auto_weather (seconds between weather changes, default 30, 0 = off),
auto_weather_mode (cycle | random), auto_light (seconds between light changes, default 20, 0 = off), camera_hz, port, record (write replay.jsonl), out_dir
(replays, kpi_log.csv), drop_rate / corrupt_rate (radio link faults).
Never start two of these at once: they share the ROS network (the same topic names) and the port.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = 'edge_residual_adaptive'

# The python packages (numpy, opencv, pymavlink, websockets, torch) live in two virtual environments, see the README. EDGE_VENV_SITE (a ':'-separated list of
# site-packages folders) replaces this default.
DEFAULT_SITE = [os.path.expanduser('~/ros2_ws/venv_edge/lib/python3.12/site-packages'), os.path.expanduser('~/ros2_ws/venv_pytorch/lib/python3.12/site-packages')]
VENV_SITE = os.environ['EDGE_VENV_SITE'].split(':') if os.environ.get('EDGE_VENV_SITE') else DEFAULT_SITE

ARGS = {'weather': 'windy', 'world_seed': '1234', 'sim_hz': '60.0', 'fast': 'false', 'aux_every': '2', 'wind_comp': '0.0', 'light': 'day', 'weights': '',
        'greedy': 'true', 'pid_max': '-1.0', 'pid_min': '-1.0', 'auto_weather': '30', 'auto_weather_mode': 'cycle', 'auto_light': '20', 'camera_hz': '2.0',
        'port': '8019', 'record': 'false', 'out_dir': os.path.expanduser('~/edge_residual_logs'), 'drop_rate': '0.0', 'corrupt_rate': '0.0', 'moving_obstacles': 'true'}


def make_nodes(context):
    v = {k: LaunchConfiguration(k).perform(context) for k in ARGS}
    b = lambda name: v[name].lower() == 'true'
    seed = int(v['world_seed'])
    brain = {'greedy': b('greedy'), 'pid_max': float(v['pid_max']), 'pid_min': float(v['pid_min'])}
    if v['weights']:
        brain['weights'] = v['weights']
    nodes = [
        Node(package=PKG, executable='sim_node', name='sim_node', output='screen',
             parameters=[{'world_seed': seed, 'weather': v['weather'], 'wind_comp': float(v['wind_comp']), 'sim_hz': float(v['sim_hz']), 'fast': b('fast'),
                          'aux_every': int(v['aux_every']), 'auto_weather_s': float(v['auto_weather']), 'auto_weather_mode': v['auto_weather_mode'],
                          'auto_light_s': float(v['auto_light']), 'light': v['light'], 'moving_obstacles': b('moving_obstacles')}]),
        Node(package=PKG, executable='brain_node', name='brain_node', output='screen', parameters=[brain]),
        Node(package=PKG, executable='camera_node', name='camera_node', output='screen',
             parameters=[{'world_seed': seed, 'light': v['light'], 'hz': float(v['camera_hz'])}]),
        Node(package=PKG, executable='detector_node', name='detector_node', output='screen'),
        Node(package=PKG, executable='telemetry_node', name='telemetry_node', output='screen',
             parameters=[{'drop_rate': float(v['drop_rate']), 'corrupt_rate': float(v['corrupt_rate'])}]),
        Node(package=PKG, executable='gcs_node', name='gcs_node', output='screen'),
        Node(package=PKG, executable='kpi_node', name='kpi_node', output='screen', parameters=[{'log_dir': v['out_dir']}]),
        Node(package=PKG, executable='web_bridge', name='web_bridge', output='screen', parameters=[{'port': int(v['port']), 'world_seed': seed, 'log_dir': v['out_dir']}]),
    ]
    if b('record'):
        nodes.append(Node(package=PKG, executable='recorder_node', name='recorder_node', output='screen',
                          parameters=[{'out_dir': v['out_dir'], 'world_seed': seed}]))
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=d) for k, d in ARGS.items()]
        + [SetEnvironmentVariable('PYTHONPATH', ':'.join(VENV_SITE) + ':' + os.environ.get('PYTHONPATH', '')),
           OpaqueFunction(function=make_nodes)])
