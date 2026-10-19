"""
The object detector as a ROS 2 node: the small YOLO-style CNN (189k parameters) of the drone project, run on the camera's 5-channel picture.
It finds vehicles, people and buildings.

Subscribes:  /edge_residual/camera/tensor, /edge_residual/camera/truth (the detector runs when the truth of a picture arrives), /edge_residual/control
             ({"cmd": "sensors", "value": "fused" | "rgb" | "thermal" | "lidar" | "thermal+lidar"}: which of the camera's sensors the detector may use)
Publishes:
    /edge_residual/detections        String (JSON)  found and really-there counts per class, the boxes, altitude, light, sensor set
    /edge_residual/camera/annotated  Image (bgr8)   RGB, thermal and height views side by side with the detector's boxes drawn
Parameters:  weights (default: weights/detector.pt of this package), score_threshold (0.4)
"""

import json

import cv2
import numpy as np
import rclpy
import torch
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String

from .spin_util import spin
from .aerial import CLASSES, SIZE
from .detector_model import MODALITIES, Detector, decode

COLORS = [(0, 0, 255), (0, 255, 255), (255, 128, 0)]          # vehicle, person, building (BGR)


class DetectorNode(Node):

    def __init__(self):
        super().__init__('detector_node', start_parameter_services=False)
        self.declare_parameter('weights', get_package_share_directory('edge_residual_adaptive') + '/weights/detector.pt')
        self.declare_parameter('score_threshold', 0.4)
        torch.set_num_threads(1)
        self.model = Detector()
        self.model.load_state_dict(torch.load(self.get_parameter('weights').value, map_location='cpu'))
        self.model.eval()
        self.threshold = float(self.get_parameter('score_threshold').value)
        self.sensors = 'fused'
        self.tensor = None
        self.pub_det = self.create_publisher(String, '/edge_residual/detections', 2)
        self.pub_img = self.create_publisher(Image, '/edge_residual/camera/annotated', 2)
        self.create_subscription(Float32MultiArray, '/edge_residual/camera/tensor', self.on_tensor, 2)
        self.create_subscription(String, '/edge_residual/camera/truth', self.on_truth, 2)
        self.create_subscription(String, '/edge_residual/control', self.on_control, 10)
        self.get_logger().info(f'detector ready ({sum(p.numel() for p in self.model.parameters()):,} parameters)')

    def on_tensor(self, m):
        self.tensor = np.asarray(m.data, dtype=np.float32).reshape(5, SIZE, SIZE)

    def on_control(self, m):
        try:
            c = json.loads(m.data)
        except ValueError:
            return
        if c.get('cmd') == 'sensors' and c.get('value') in MODALITIES:
            self.sensors = c['value']

    def on_truth(self, m):
        if self.tensor is None:
            return
        truth = json.loads(m.data)
        x = np.zeros_like(self.tensor)
        chans = MODALITIES[self.sensors]
        x[chans] = self.tensor[chans]
        with torch.no_grad():
            dets = decode(self.model(torch.from_numpy(x[None])), self.threshold)[0]
        found = [0, 0, 0]
        for c, *_ in dets:
            found[c] += 1
        rgb = (x[:3].transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8).copy()
        thermal = cv2.applyColorMap((x[3] * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        height = cv2.cvtColor((x[4] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        panels = []
        for im in (rgb, thermal, height):
            im = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            for c, s, x0, y0, x1, y1 in dets:
                cv2.rectangle(im, (int(x0 * 2), int(y0 * 2)), (int(x1 * 2), int(y1 * 2)), COLORS[c], 1)
            panels.append(im)
        panel = np.ascontiguousarray(np.concatenate(panels, axis=1))
        img = Image()
        img.header.frame_id = 'drone_camera'
        img.header.stamp = self.get_clock().now().to_msg()
        img.height, img.width, img.encoding, img.step, img.data = panel.shape[0], panel.shape[1], 'bgr8', panel.shape[1] * 3, panel.tobytes()
        self.pub_img.publish(img)
        out = String()
        out.data = json.dumps({'found': found, 'truth': truth['truth'], 'names': CLASSES, 'alt': truth['alt'], 'light': truth['light'], 'sensors': self.sensors,
                               'boxes': [[int(c), round(float(s), 2), round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)] for c, s, x0, y0, x1, y1 in dets]})
        self.pub_det.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
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
