"""
A simulated ground station as a ROS 2 node: it reads the radio bytes of /edge_residual/mavlink, parses the MAVLink v2 frames, checks the CRC of every
one and counts what arrived, what was damaged and what was lost (gaps in the sequence numbers).

Publishes /edge_residual/gcs (String, JSON, once a second): frames, bytes, crc_errors, lost, frames per message type, the last battery and wind report.
"""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, UInt8MultiArray

from .spin_util import spin
from .telemetry_lib import GroundStation


class GcsNode(Node):

    def __init__(self):
        super().__init__('gcs_node', start_parameter_services=False)
        self.gs = GroundStation()
        self.bytes = 0
        self.pub = self.create_publisher(String, '/edge_residual/gcs', 5)
        self.create_subscription(UInt8MultiArray, '/edge_residual/mavlink', self.on_bytes, 100)
        self.create_timer(1.0, self.report)

    def on_bytes(self, msg):
        data = bytes(msg.data)
        self.bytes += len(data)
        self.gs.feed(data)

    def report(self):
        st = self.gs.stats()
        m = String()
        m.data = json.dumps({'frames': st['frames'], 'bytes': self.bytes, 'crc_errors': st['crc_errors'], 'lost': st['lost_packets'],
                             'by_type': st['by_type'], 'battery': st['battery'], 'wind': st['wind']})
        self.pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = GcsNode()
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
