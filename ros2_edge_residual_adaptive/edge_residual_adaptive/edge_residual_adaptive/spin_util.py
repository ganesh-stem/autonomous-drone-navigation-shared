"""Spinning a node. The events executor of rclpy (experimental in Jazzy) waits for events in C++ instead of rebuilding a Python wait set on every message,
which cut the CPU time of the nodes a lot. Set EDGE_EXECUTOR=default to use the normal executor, and it is also the fallback if the events executor is missing."""

import os

import rclpy
from rclpy.executors import ExternalShutdownException


def spin(node, paced=False):
    """paced=True: the node runs on a timer that must keep time (the simulator at sim_hz). The events executor ran such a timer at about 88% of its rate
    in the tests here (52.9 steps/s at 60), the normal executor keeps it exact (60.0), so those nodes use the normal executor."""
    if not paced and os.environ.get('EDGE_EXECUTOR', 'events') != 'default':
        try:
            from rclpy.experimental import EventsExecutor
            executor = EventsExecutor()
            executor.add_node(node)
            executor.spin()
            return
        except ImportError:
            pass
        except ExternalShutdownException:              # Ctrl+C or the launch stopping: leave quietly, without a traceback
            return
    rclpy.spin(node)
