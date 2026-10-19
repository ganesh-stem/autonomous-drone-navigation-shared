"""
Replay recorder as a ROS 2 node: writes replay.jsonl in the format of the training project (a meta line, then one line of 20 numbers per recorded step),
so a flight made through ROS can be played in the 3D viewer's Replay drawer of web3d_drones_edge, or sliced with slice_replay.py.

Subscribes /edge_residual/status. Parameters: out_dir (default /tmp/edge_residual), every (record every Nth step, default 1).
Step line: [step, x, y, z, yaw, action, reward, hit, goal x, y, z, goals, score, 0, vz, wind x, y, z, battery %, weather index]  (score = mean reward of the last 1000 steps)
"""

import json
import os
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

WEATHERS = ['calm', 'breeze', 'windy', 'gale', 'gale_dusty']

from .spin_util import spin


class RecorderNode(Node):

    def __init__(self):
        super().__init__('recorder_node', start_parameter_services=False)
        self.declare_parameter('out_dir', '/tmp/edge_residual')
        self.declare_parameter('every', 1)
        self.declare_parameter('world_seed', 1234)
        out = str(self.get_parameter('out_dir').value)
        os.makedirs(out, exist_ok=True)
        self.every = max(1, int(self.get_parameter('every').value))
        self.fh = open(os.path.join(out, 'replay.jsonl'), 'w', encoding='utf-8')
        self.fh.write(json.dumps({'t': 'meta', 'kind': 'drone_edge', 'version': 2, 'world_seed': int(self.get_parameter('world_seed').value), 'wind_aware': True,
                                  'weathers': WEATHERS, 'actions': 9}) + '\n')
        self.rewards = deque(maxlen=1000)
        self.buf = []
        self.count = 0
        self.create_subscription(String, '/edge_residual/status', self.on_status, 50)
        self.create_timer(5.0, self.flush)

    def on_status(self, msg):
        s = json.loads(msg.data)
        if s['step'] == 0:
            return
        self.rewards.append(s['reward'])
        if s['step'] % self.every:
            return
        score = sum(self.rewards) / len(self.rewards)
        g, w = s['goal'], s['wind']
        self.buf.append(f"[{s['step']},{s['x']:.1f},{s['y']:.1f},{s['z']:.1f},{s['yaw']:.1f},{s['action']},{s['reward']:.3f},{int(s['colliding'])},"
                        f"{g[0]:.0f},{g[1]:.0f},{g[2]:.0f},{s['goals']},{score:.3f},0,{s['vz']:.1f},{w[0]:.2f},{w[1]:.2f},{w[2]:.2f},{s['battery']:.1f},{s['weather_index']}]")
        self.count += 1

    def flush(self):
        if self.buf:
            self.fh.write('\n'.join(self.buf) + '\n')
            self.fh.flush()
            self.buf.clear()

    def close(self):
        self.flush()
        self.fh.close()


def main(args=None):
    rclpy.init(args=args)
    node = RecorderNode()
    try:
        spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
