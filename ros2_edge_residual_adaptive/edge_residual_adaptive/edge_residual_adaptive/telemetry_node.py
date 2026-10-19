"""
MAVLink v2 telemetry of the drone as a ROS 2 node (the vehicle side of the radio link).

Subscribes /edge_residual/status and turns every step into the MAVLink frames a flight controller would send: HEARTBEAT, LOCAL_POSITION_NED, ATTITUDE,
SYS_STATUS (battery), WIND_COV and NAMED_VALUE_FLOAT (distance to the goal). Publishes the raw bytes of each frame on
    /edge_residual/mavlink   UInt8MultiArray
Parameters: drop_rate (chance that a frame is lost on the radio, default 0), corrupt_rate (chance that one byte of a frame is damaged, default 0).
The ground station (gcs_node) decodes the stream, checks every CRC and counts what was lost.
"""

import json
import random

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, UInt8MultiArray

from .spin_util import spin
from .telemetry_lib import Telemetry


class Vehicle:
    """What Telemetry.step_frames() reads from a drone, filled from the status message."""
    x = y = z = yaw = distance = 0.0
    battery_percent = 100.0
    wind = (0.0, 0.0, 0.0)


class TelemetryNode(Node):

    def __init__(self):
        super().__init__('telemetry_node', start_parameter_services=False)
        self.declare_parameter('drop_rate', 0.0)
        self.declare_parameter('corrupt_rate', 0.0)
        self.drop = float(self.get_parameter('drop_rate').value)
        self.corrupt = float(self.get_parameter('corrupt_rate').value)
        self.rng = random.Random(3)
        self.tm = Telemetry()
        self.veh = Vehicle()
        self.pub = self.create_publisher(UInt8MultiArray, '/edge_residual/mavlink', 50)
        # status_slow carries the same text as status every aux_every-th step (2 by default), so every step that sends a frame (all even ones) is still seen
        self.create_subscription(String, '/edge_residual/status_slow', self.on_status, 10)

    def on_status(self, msg):
        s = json.loads(msg.data)
        v = self.veh
        v.x, v.y, v.z, v.yaw = s['x'], s['y'], s['z'], s['yaw']
        v.battery_percent = s['battery']
        v.wind = tuple(s['wind'])
        step = int(s['step'])
        if step % 20 == 10:                                           # the only step that sends the distance to the goal
            g = s['goal']
            v.distance = float(np.linalg.norm([g[0] - v.x, g[1] - v.y, g[2] - v.z]))
        for frame in self.tm.step_frames(v, step):
            if self.rng.random() < self.drop:
                continue                                          # lost on the radio
            data = bytearray(frame)
            if self.rng.random() < self.corrupt:
                data[self.rng.randrange(len(data))] ^= 0x55       # one damaged byte: the CRC will catch it
            out = UInt8MultiArray()
            out.data = bytes(data)
            self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = TelemetryNode()
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
