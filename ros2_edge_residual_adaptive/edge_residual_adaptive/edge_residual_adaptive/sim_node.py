"""
The drone simulator as a ROS 2 node: the same world, weather, wind, battery and noisy rangefinders as the training (core.py).

Every tick it applies the latest action of the brain, steps the world, and publishes what a drone would have:

    /edge_residual/obs         Float32MultiArray  [step, respawns, the 37 observation values]  (what the brain reads)
    /edge_residual/pose        PoseStamped        position (x, y, z) and heading (frame 'map', z up)
    /edge_residual/goal        PointStamped       the current waypoint
    /edge_residual/wind        Vector3Stamped     the wind push the drone feels right now
    /edge_residual/battery     BatteryState       percentage from the energy model
    /edge_residual/rays        Float32MultiArray  28 rangefinder rays, [end x, end y, end z, signal] each
    /edge_residual/traffic     Float32MultiArray  [clock, n, then per car x, y, heading, type, colour index, braking, speed] (only while somebody subscribes; 20 times a second at 60 steps/s)
    /edge_residual/flyers      Float32MultiArray  [step number of the moving floating obstacles, 1 if they move] (only while somebody subscribes; 20 times a second at 60 steps/s)
    /edge_residual/status      String (JSON)      everything else: step, goals, collisions, respawns, weather, energy, battery, ...
Subscribes:
    /edge_residual/action      Int32MultiArray    [step, action]  the brain's answer to the observation with that step number
    /edge_residual/control     String (JSON)      {"cmd": "pause" | "resume" | "speed" | "weather" | "new_goal" | "auto_weather" | "auto_light", "value": ...}
                                               auto_weather: {"seconds": 30, "mode": "cycle" | "random"}, auto_light: {"seconds": 20}, seconds 0 = off
Automatic changes (parameters auto_weather_s, auto_weather_mode, auto_light_s, or the commands above): the weather changes every N simulated seconds
(60 steps = 1 s) in the order calm, breeze, windy, gale, gale_dusty or at random, and the light goes day, dusk, night, dusk, day, ... The simulator counts,
and announces a light change to the camera and the viewer by publishing {"cmd": "light"} on /edge_residual/control.

Timing (fast:=true steps the moment the answer arrives, with no fixed rate): one tick = one simulation step, at sim_hz * speed. The brain answers the observation of step k before tick k+1, so the action is applied to
the state it was chosen for. If the answer is late, the last action is held and the counter 'stale' goes up (a real drone would do the same).
"""

import json
import math
import random
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped, Vector3Stamped
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Float32MultiArray, Int32MultiArray, String

from .spin_util import spin
from . import city_world as CW
from .core import CONDITIONS, GOAL_RADIUS, N_SENSORS, EdgeDroneEnvV3

WEATHERS = list(CONDITIONS)
LIGHT_CYCLE = ['day', 'dusk', 'night', 'dusk']


class SimNode(Node):

    def __init__(self):
        super().__init__('sim_node', start_parameter_services=False)
        for name, default in [('world_seed', 1234), ('rng_seed', 5), ('weather', 'windy'), ('wind_comp', 0.0), ('wind_aware', True),
                              ('sim_hz', 60.0), ('auto_weather_s', 0.0), ('auto_weather_mode', 'cycle'), ('auto_light_s', 0.0), ('light', 'day'), ('fast', False), ('aux_every', 2), ('moving_obstacles', True)]:
            self.declare_parameter(name, default)
        p = lambda n: self.get_parameter(n).value
        self.hz = float(p('sim_hz'))
        self.aux_every = max(1, int(p('aux_every')))      # pose, goal, wind, battery and rays go out every N steps (the observation and the status every step)
        self.fast = bool(p('fast'))              # step as soon as the brain's answer arrives (no fixed rate), the timer stays as a time-out
        self.env = EdgeDroneEnvV3(seed=int(p('world_seed')), rng_seed=int(p('rng_seed')), weather=str(p('weather')),
                                  wind_aware=bool(p('wind_aware')), wind_comp=float(p('wind_comp')))
        self.env.flyers_on = bool(p('moving_obstacles'))   # the floating blocks (the viewer's birds and patrol drones) are real, MOVING obstacles; false = the static world the brain was trained in
        if self.env.flyers_on:
            self.env._update_sensors(); self.env._update_nav()      # the first observation already sees them
        self.seed = int(p('world_seed'))
        self.traffic = CW.Traffic(seed=self.seed)          # the cars of the city: the camera, the detector and the viewer all see these
        self.step_no = 0
        self.collisions = 0
        self.respawns = 0
        self.stale = 0
        self.last_action = 0
        self.pending = None                      # (step number the action answers, action)
        self.auto_weather = dict(seconds=float(p('auto_weather_s')), mode=str(p('auto_weather_mode')))
        self.auto_light = dict(seconds=float(p('auto_light_s')))
        self.light = str(p('light'))
        self.light_phase = LIGHT_CYCLE.index(self.light) if self.light in LIGHT_CYCLE else 0
        self.weather_steps = self.light_steps = 0
        self.pick = random.Random(2)
        self.paused = False
        self.started = False                     # becomes True when the first observation is out
        self.last_reward = 0.0
        self.speed = 1.0

        self.pub_obs = self.create_publisher(Float32MultiArray, '/edge_residual/obs', 10)
        self.pub_pose = self.create_publisher(PoseStamped, '/edge_residual/pose', 10)
        self.pub_goal = self.create_publisher(PointStamped, '/edge_residual/goal', 10)
        self.pub_wind = self.create_publisher(Vector3Stamped, '/edge_residual/wind', 10)
        self.pub_batt = self.create_publisher(BatteryState, '/edge_residual/battery', 10)
        self.pub_rays = self.create_publisher(Float32MultiArray, '/edge_residual/rays', 10)
        self.pub_traffic = self.create_publisher(Float32MultiArray, '/edge_residual/traffic', 5)
        self.pub_flyers = self.create_publisher(Float32MultiArray, '/edge_residual/flyers', 5)      # [step number of the floating obstacles, 1 if they move]: all that the camera node and the viewer need to know where they are
        self.pub_status = self.create_publisher(String, '/edge_residual/status', 10)
        self.pub_slow = self.create_publisher(String, '/edge_residual/status_slow', 10)     # the same text every aux_every steps, for the viewer and the KPIs
        self.pub_control = self.create_publisher(String, '/edge_residual/control', 10)
        self.create_subscription(Int32MultiArray, '/edge_residual/action', self.on_action, 10)
        self.create_subscription(String, '/edge_residual/control', self.on_control, 10)
        self.last_step_time = time.monotonic()
        self.timer = self.create_timer(0.05 if self.fast else 1.0 / self.hz, self.tick)
        self.first = self.create_timer(1.0, self.first_state)      # the first observation, once the other nodes are up, so the brain can answer
        self.get_logger().info(f'sim: weather={p("weather")} wind_comp={p("wind_comp")} world_seed={self.seed} at {self.hz:.0f} steps/s{" (fast: steps when the brain answers)" if self.fast else ""}')

    def first_state(self):
        self.first.cancel()
        self.publish_state(action=0, reward=0.0)
        self.last_step_time = time.monotonic()
        self.started = True

    # ---- inputs ---------------------------------------------------------------------
    def on_action(self, msg):
        if len(msg.data) >= 2:
            self.pending = (int(msg.data[0]), int(msg.data[1]))
            if self.fast and self.started and not self.paused and self.pending[0] == self.step_no:
                self.tick()

    def on_control(self, msg):
        try:
            c = json.loads(msg.data)
        except ValueError:
            return
        cmd, value = c.get('cmd'), c.get('value')
        if cmd == 'pause':
            self.paused = True
            self.publish_status(self.last_action, self.last_reward, force=True)      # nothing else is published while paused: tell the others now
        elif cmd == 'resume':
            self.paused = False
            self.publish_status(self.last_action, self.last_reward, force=True)
        elif cmd == 'speed':
            self.speed = max(0.25, min(8.0, float(value)))
            if not self.fast:
                self.timer.cancel()
                self.timer = self.create_timer(1.0 / (self.hz * self.speed), self.tick)
        elif cmd == 'weather' and value in CONDITIONS:
            self.env.set_weather(value)
            self.weather_steps = 0
        elif cmd == 'auto_weather' and isinstance(value, dict):
            self.auto_weather = dict(seconds=max(0.0, float(value.get('seconds', 0))), mode='random' if value.get('mode') == 'random' else 'cycle')
            self.weather_steps = 0
        elif cmd == 'auto_light' and isinstance(value, dict):
            self.auto_light = dict(seconds=max(0.0, float(value.get('seconds', 0))))
            self.light_steps = 0
        elif cmd == 'light' and value in LIGHT_CYCLE and value != self.light:      # set by hand (this node's own announcements already match)
            self.light, self.light_steps = value, 0
            self.light_phase = LIGHT_CYCLE.index(value)
        elif cmd == 'new_goal':
            self.env.new_goal()
            self.env._update_nav()

    # ---- one simulation step ----------------------------------------------------------
    def tick(self):
        if self.paused or not self.started:
            return
        now = time.monotonic()
        if self.step_no == 0 and self.pending is None:       # wait for the brain's first answer: a brain that is still starting would otherwise leave the drone flying straight ahead
            if now - self.last_step_time > 1.0:              # into the first building. The first observation is sent again once a second in case the brain missed it
                self.last_step_time = now
                self.publish_state(action=0, reward=0.0)
            return
        if self.fast and now - self.last_step_time < 0.05 and (self.pending is None or self.pending[0] != self.step_no):
            return                               # fast mode: the timer only acts as a time-out when no answer came for 50 ms
        self.last_step_time = now
        if self.pending is not None and self.pending[0] == self.step_no:
            action = self.pending[1]
        else:
            action = self.last_action
            self.stale += 1
        self.last_action = action
        respawned = self.env.step(action)
        self.step_no += 1
        if self.step_no % 3 == 0 and self.pub_flyers.get_subscription_count():
            fm = Float32MultiArray()
            fm.data = [float(self.env.flyer_t), 1.0 if self.env.flyers_on else 0.0]
            self.pub_flyers.publish(fm)
        if self.step_no % 3 == 0:                            # the traffic runs at 20 steps per second of simulated time (0.4 ms each)
            self.traffic.step(3.0 / 60.0, self.env.x, self.env.y, self.light)
            if self.pub_traffic.get_subscription_count():
                rows = self.traffic.snapshot()
                tm = Float32MultiArray()
                tm.data = [self.traffic.clock, float(len(rows))] + [float(v) for r in rows for v in r]
                self.pub_traffic.publish(tm)
        self.weather_steps += 1
        self.light_steps += 1
        self.automatic_changes()
        self.collisions += int(self.env.is_colliding)
        self.respawns += int(respawned)
        self.publish_state(action, self.env.reward)

    def automatic_changes(self):
        aw, al = self.auto_weather, self.auto_light
        if aw['seconds'] > 0 and self.weather_steps >= aw['seconds'] * 60:
            names = list(CONDITIONS)
            current = self.env.weather_name
            nxt = self.pick.choice([n for n in names if n != current]) if aw['mode'] == 'random' else names[(names.index(current) + 1) % len(names)]
            self.env.set_weather(nxt)
            self.weather_steps = 0
        if al['seconds'] > 0 and self.light_steps >= al['seconds'] * 60:
            self.light_phase = (self.light_phase + 1) % len(LIGHT_CYCLE)
            self.light, self.light_steps = LIGHT_CYCLE[self.light_phase], 0
            m = String()
            m.data = json.dumps({'cmd': 'light', 'value': self.light})
            self.pub_control.publish(m)

    def publish_state(self, action, reward):
        e, now = self.env, self.get_clock().now().to_msg()
        obs = e.get_state()
        m = Float32MultiArray()
        m.data = [float(self.step_no), float(self.respawns)] + [float(v) for v in obs]
        self.pub_obs.publish(m)

        if self.step_no % self.aux_every:
            self.last_reward = reward
            self.publish_status(action, reward)
            return
        if self.pub_pose.get_subscription_count():                 # nobody listening, nothing to build
            pose = PoseStamped()
            pose.header.stamp, pose.header.frame_id = now, 'map'
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = float(e.x), float(e.y), float(e.z)
            yaw = math.radians(e.yaw)
            pose.pose.orientation.z, pose.pose.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
            self.pub_pose.publish(pose)
        if self.pub_goal.get_subscription_count():                 # nobody listening, nothing to build
            goal = PointStamped()
            goal.header.stamp, goal.header.frame_id = now, 'map'
            goal.point.x, goal.point.y, goal.point.z = float(e.goal[0]), float(e.goal[1]), float(e.goal[2])
            self.pub_goal.publish(goal)
        if self.pub_wind.get_subscription_count():                 # nobody listening, nothing to build
            wind = Vector3Stamped()
            wind.header.stamp, wind.header.frame_id = now, 'map'
            wind.vector.x, wind.vector.y, wind.vector.z = float(e.wind[0]), float(e.wind[1]), float(e.wind[2])
            self.pub_wind.publish(wind)
        if self.pub_batt.get_subscription_count():                 # nobody listening, nothing to build
            batt = BatteryState()
            batt.header.stamp = now
            batt.percentage = float(e.battery_percent) / 100.0
            batt.present = True
            self.pub_batt.publish(batt)
        if self.pub_rays.get_subscription_count():                 # nobody listening, nothing to build
            rays = Float32MultiArray()
            ends = e.sensor_end
            rays.data = np.column_stack((np.asarray(ends, dtype=np.float32)[:N_SENSORS, :3], np.asarray(e.signals, dtype=np.float32)[:N_SENSORS])).ravel().tolist()
            self.pub_rays.publish(rays)
        self.last_reward = reward
        self.publish_status(action, reward)

    def publish_status(self, action, reward, force=False):
        e = self.env
        s = String()
        s.data = json.dumps({
            'step': self.step_no, 'x': round(float(e.x), 2), 'y': round(float(e.y), 2), 'z': round(float(e.z), 2), 'yaw': round(float(e.yaw), 1),
            'vz': round(float(e.vz), 2), 'goal': [round(float(v), 1) for v in e.goal], 'goal_radius': GOAL_RADIUS,
            'wind': [round(float(v), 3) for v in e.wind], 'battery': round(e.battery_percent, 1), 'weather': e.weather_name,
            'weather_index': WEATHERS.index(e.weather_name), 'goals': e.goals_reached, 'collisions': self.collisions, 'respawns': self.respawns,
            'colliding': bool(e.is_colliding), 'reward': round(float(reward), 3), 'action': int(action), 'energy': round(float(e.energy), 2),
            'ground_dist': round(float(e.ground_dist), 1), 'stale': self.stale, 'paused': self.paused, 'speed': self.speed, 'hz': self.hz,
            'world_seed': self.seed, 'light': self.light, 'auto': {'weather': self.auto_weather, 'light': self.auto_light}, 'wind_comp': e.wind_comp, 'time': round(time.time(), 3)})
        self.pub_status.publish(s)
        if force or self.step_no % self.aux_every == 0:
            self.pub_slow.publish(s)


def main(args=None):
    rclpy.init(args=args)
    node = SimNode()
    try:
        spin(node, paced=not node.fast)         # fast mode has no timer to keep, so it can use the events executor
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
