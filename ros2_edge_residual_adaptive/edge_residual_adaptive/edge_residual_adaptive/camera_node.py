"""
The drone's onboard camera as a ROS 2 node: synthetic RGB, thermal and height (LiDAR-like) views of the ground under the drone, with the vehicles,
people, trees and hot rocks of the world (aerial.py). The world geometry comes from the same seed the simulator uses.

Subscribes:  /edge_residual/pose, /edge_residual/wind (vibration blur grows with the wind), /edge_residual/control ({"cmd": "light", "value": "day"|"dusk"|"night"})
Publishes (params: hz, default 2):
    /edge_residual/camera/tensor   Float32MultiArray  the 5 x 96 x 96 image (RGB, thermal, height), flat, for the detector
    /edge_residual/camera/rgb      Image (rgb8)       the colour view, for rqt_image_view or any ROS image tool
    /edge_residual/camera/thermal  Image (bgr8)       the thermal view in false colour
    /edge_residual/camera/truth    String (JSON)      what is really in the picture (labels per class), altitude, light. Published AFTER the tensor.
"""

import json
import types

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String

from .spin_util import spin
from . import aerial
from . import city_world as CW
from .core import DroneEnv


def image_msg(arr, encoding, stamp):
    m = Image()
    m.header.stamp, m.header.frame_id = stamp, 'drone_camera'
    m.height, m.width = int(arr.shape[0]), int(arr.shape[1])
    m.encoding = encoding
    m.step = int(arr.shape[1] * arr.shape[2])
    m.data = arr.tobytes()
    return m


class CameraNode(Node):

    def __init__(self):
        super().__init__('camera_node', start_parameter_services=False)
        self.declare_parameter('world_seed', 1234)
        self.declare_parameter('hz', 2.0)
        self.declare_parameter('light', 'day')
        self.world = DroneEnv(seed=int(self.get_parameter('world_seed').value))
        self.light = str(self.get_parameter('light').value)
        self.raw = None
        self.traffic_rows = []                     # the driving cars of the last /traffic message
        self.wind = np.zeros(3)
        self.count = 0
        self.pub_tensor = self.create_publisher(Float32MultiArray, '/edge_residual/camera/tensor', 2)
        self.pub_rgb = self.create_publisher(Image, '/edge_residual/camera/rgb', 2)
        self.pub_thermal = self.create_publisher(Image, '/edge_residual/camera/thermal', 2)
        self.pub_truth = self.create_publisher(String, '/edge_residual/camera/truth', 2)
        self.create_subscription(String, '/edge_residual/status_slow', self.on_status, 10)      # position and wind in one small message (was two: pose and wind)
        self.create_subscription(String, '/edge_residual/control', self.on_control, 10)
        self.create_subscription(Float32MultiArray, '/edge_residual/traffic', self.on_traffic, 2)
        self.create_subscription(Float32MultiArray, '/edge_residual/flyers', self.on_flyers, 2)       # the floating obstacles move: the step number tells this world where they are
        self.create_timer(1.0 / float(self.get_parameter('hz').value), self.tick)

    def on_status(self, m):
        self.raw = m.data                          # parsed at the camera rate (2 a second), not on every message

    def on_flyers(self, m):
        self.world.flyers_on, self.world.flyer_t = bool(m.data[1]), int(m.data[0])      # the camera's copy of the sky now has the floating blocks where the simulator has them

    def on_traffic(self, m):
        d = m.data
        n = int(d[1])
        self.traffic_rows = [d[2 + 7 * i: 9 + 7 * i] for i in range(n)]       # kept as received; only the cars inside the picture are used, at the camera rate

    def on_control(self, m):
        try:
            c = json.loads(m.data)
        except ValueError:
            return
        if c.get('cmd') == 'light' and c.get('value') in aerial.LIGHT:
            self.light = c['value']

    def tick(self):
        if self.raw is None:
            return
        st = json.loads(self.raw)
        p = types.SimpleNamespace(x=st['x'], y=st['y'], z=st['z'])
        self.wind = np.array(st['wind'])
        alt = float(np.clip(p.z, 35.0, 100.0))
        radius = aerial.FOOTPRINT_PER_ALT * alt
        rng = np.random.default_rng(self.count)
        vibration = min(2.0, 0.5 * float(np.linalg.norm(self.wind)))
        scene = aerial.targets_near(self.world, p.x, p.y, radius)
        scene['vehicles'] = CW.parked_aerial(CW.parked_near(self.world, p.x, p.y, 1), p.x, p.y, radius) + CW.aerial_from_rows(self.traffic_rows, p.x, p.y, radius)   # the same cars the viewer draws
        img, labels = aerial.render(rng, self.world.boxes_near(p.x, p.y, radius), p.x, p.y, alt, self.light, vibration, scene=scene)
        self.count += 1
        stamp = self.get_clock().now().to_msg()
        t = Float32MultiArray()
        t.data = img.reshape(-1).tolist()
        self.pub_tensor.publish(t)
        rgb = np.ascontiguousarray((img[:3].transpose(1, 2, 0) * 255).astype(np.uint8))
        self.pub_rgb.publish(image_msg(rgb, 'rgb8', stamp))
        thermal = cv2.applyColorMap((img[3] * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        self.pub_thermal.publish(image_msg(np.ascontiguousarray(thermal), 'bgr8', stamp))
        truth = [0, 0, 0]
        for c, *_ in labels:
            truth[c] += 1
        s = String()
        s.data = json.dumps({'truth': truth, 'names': aerial.CLASSES, 'alt': round(alt), 'light': self.light, 'labels': [list(map(int, l)) for l in labels]})
        self.pub_truth.publish(s)


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
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
