"""
The adaptive-weight residual SAC brain as a ROS 2 node.

Subscribes /edge_residual/obs and runs the policy of residual_policy.py (the actor of the trained network plus the PID bias whose weight fades near obstacles,
weights/residual_adaptive_actor.npz, see tools/export_residual_npz.py), then publishes /edge_residual/action [step, action].
It also does what the training and the viewer server did around the network:
  - after a respawn (the drone was stuck and teleported) it flies 90 random actions to get moving again
  - it takes the most probable action (greedy:=true, the way a finished SAC brain is evaluated) or samples from the policy (greedy:=false)
It needs no torch: the weights are a numpy file.
"""

import random

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32MultiArray

from .spin_util import spin
from .residual_policy import ResidualAdaptiveActor

RANDOM_STEPS_AFTER_RESPAWN = 90


class BrainNode(Node):

    def __init__(self):
        super().__init__('brain_node', start_parameter_services=False)
        share = get_package_share_directory('edge_residual_adaptive')
        for name, default in [('weights', share + '/weights/residual_adaptive_actor.npz'), ('greedy', True), ('seed', 7), ('pid_max', -1.0), ('pid_min', -1.0)]:
            self.declare_parameter(name, default)
        p = lambda n: self.get_parameter(n).value
        pid_max, pid_min = float(p('pid_max')), float(p('pid_min'))
        self.actor = ResidualAdaptiveActor(p('weights'), pid_max if pid_max >= 0 else None, pid_min if pid_min >= 0 else None)     # negative = use the values stored in the file
        self.greedy = bool(p('greedy'))
        self.rng = random.Random(int(p('seed')))
        self.np_rng = np.random.default_rng(int(p('seed')))
        self.respawns = 0
        self.random_left = 0
        self.decisions = 0
        self.out = Int32MultiArray()                                    # one message object, reused for every answer
        self.pub = self.create_publisher(Int32MultiArray, '/edge_residual/action', 10)
        self.create_subscription(Float32MultiArray, '/edge_residual/obs', self.on_obs, 10)
        self.get_logger().info(f'brain: adaptive residual SAC, reads {self.actor.n_obs} inputs, {self.actor.n_actions} actions, PID weight {self.actor.pid_max} (far) to '
                               f'{self.actor.pid_min} (next to an obstacle), {"most probable action" if self.greedy else "sampling"}')

    def on_obs(self, msg):
        data = msg.data
        step, respawns = int(data[0]), int(data[1])
        if respawns != self.respawns:                            # the drone was teleported: fly randomly for a moment
            self.respawns = respawns
            self.random_left = RANDOM_STEPS_AFTER_RESPAWN
        if self.random_left > 0:
            self.random_left -= 1
            action = self.rng.randrange(self.actor.n_actions)
        else:
            action = self.actor.act(np.asarray(data[2:], dtype=np.float32), self.np_rng, self.greedy)
        self.out.data = [step, action]
        self.pub.publish(self.out)
        self.decisions += 1


def main(args=None):
    rclpy.init(args=args)
    node = BrainNode()
    try:
        spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
