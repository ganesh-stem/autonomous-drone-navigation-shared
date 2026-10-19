"""
KPIs of the flight as a ROS 2 node, per weather: how well is the drone doing in the weather it is in now?

Subscribes /edge_residual/status_slow (the status every aux_every steps). The counters restart whenever the weather changes. Publishes /edge_residual/kpi (String, JSON, twice a second):
    weather, steps, goals, collisions, goals_per_1k, collisions_per_1k, energy_per_goal, ground_speed, battery, stale (late brain answers)
The same measures as the evaluations of the training project (goals and collisions per 1000 steps, energy per goal, ground speed).
Every finished weather segment is appended to <log_dir>/kpi_log.csv (parameter log_dir, default /tmp/edge_residual), and the running one on shutdown.
"""

import csv
import json
import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

FIELDS = ['weather', 'steps', 'goals', 'collisions', 'goals_per_1k', 'collisions_per_1k', 'energy_per_goal', 'ground_speed', 'battery', 'stale']

from .spin_util import spin


class KpiNode(Node):

    def __init__(self):
        super().__init__('kpi_node', start_parameter_services=False)
        self.declare_parameter('log_dir', '/tmp/edge_residual')
        self.log_dir = str(self.get_parameter('log_dir').value)
        os.makedirs(self.log_dir, exist_ok=True)
        self.path = os.path.join(self.log_dir, 'kpi_log.csv')
        with open(self.path, 'w', newline='') as fh:
            csv.writer(fh).writerow(FIELDS)
        self.base = None
        self.last = None
        self.raw = None
        self.kpi = {}
        self.pub = self.create_publisher(String, '/edge_residual/kpi', 5)
        self.create_subscription(String, '/edge_residual/status_slow', self.on_status, 10)
        self.create_timer(0.5, self.publish)

    def on_status(self, msg):
        s = json.loads(msg.data)
        if self.base is None or s['weather'] != self.base['weather'] or s['step'] < self.base['step']:
            self.flush()
            self.base = dict(weather=s['weather'], step=s['step'], goals=s['goals'], collisions=s['collisions'], energy=s['energy'], ground=s['ground_dist'],
                             stale=s['stale'])
        self.last = s
        b = self.base
        steps = max(s['step'] - b['step'], 0)
        goals, coll = s['goals'] - b['goals'], s['collisions'] - b['collisions']
        self.kpi = dict(weather=s['weather'], steps=steps, goals=goals, collisions=coll,
                        goals_per_1k=round(1000.0 * goals / max(steps, 1), 2), collisions_per_1k=round(1000.0 * coll / max(steps, 1), 1),
                        energy_per_goal=round((s['energy'] - b['energy']) / goals, 1) if goals else None,
                        ground_speed=round((s['ground_dist'] - b['ground']) / max(steps, 1), 2), battery=s['battery'], stale=s['stale'] - b['stale'])

    def flush(self):
        if self.kpi and self.kpi.get('steps', 0) > 0:
            with open(self.path, 'a', newline='') as fh:
                csv.writer(fh).writerow([self.kpi.get(k, '') if self.kpi.get(k) is not None else '' for k in FIELDS])

    def publish(self):
        if self.kpi:
            m = String()
            m.data = json.dumps(self.kpi)
            self.pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = KpiNode()
    try:
        spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.flush()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
