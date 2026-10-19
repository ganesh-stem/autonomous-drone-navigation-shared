import os
from glob import glob

from setuptools import setup

package_name = 'edge_residual_adaptive'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'weights'), glob('weights/*')),
        (os.path.join('share', package_name, 'web'), [f for f in glob('web/*') if os.path.isfile(f)]),     # index.html, app.js
        (os.path.join('share', package_name, 'web', 'js'), glob('web/js/*.js')),                           # the viewer's modules
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='edge_residual_adaptive',
    maintainer_email='you@example.com',
    description='Adaptive-weight residual SAC drone (PID baseline + learned correction) in weather, as ROS 2 nodes',
    license='MIT',
    entry_points={
        'console_scripts': [
            'sim_node = edge_residual_adaptive.sim_node:main',
            'brain_node = edge_residual_adaptive.brain_node:main',
            'camera_node = edge_residual_adaptive.camera_node:main',
            'detector_node = edge_residual_adaptive.detector_node:main',
            'telemetry_node = edge_residual_adaptive.telemetry_node:main',
            'gcs_node = edge_residual_adaptive.gcs_node:main',
            'kpi_node = edge_residual_adaptive.kpi_node:main',
            'recorder_node = edge_residual_adaptive.recorder_node:main',
            'web_bridge = edge_residual_adaptive.web_bridge:main',
        ],
    },
)
