"""
Evaluate the drone through ROS: fly the running system through every weather for a fixed number of steps and print goals and collisions per 1000
steps, energy per goal and ground speed, the same measures as the evaluations of the training project.

    # terminal 1 (a launch that is not shared with anything else)
    ros2 launch edge_residual_adaptive edge_residual_adaptive.launch.py fast:=true auto_weather:=0 auto_light:=0        # fast: the simulator waits for the brain, so no late answers (sim_hz:=100 also works)
    # terminal 2
    python3 tools/evaluate_ros.py --steps 6000

Every weather gets --steps steps, one after the other (the drone keeps flying, only the weather changes). Late answers of the brain ('stale') are counted too:
a high number means the simulation was run faster than the brain could answer, so use a lower sim_hz. Writes ros_eval.json (--out).
One run in one sky (the world seed of the launch), so it is a check that ROS reproduces the behaviour, not a replacement for the 6-seed evaluation.
"""

import argparse
import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

WEATHERS = ['calm', 'breeze', 'windy', 'gale', 'gale_dusty']


class Evaluator(Node):

    def __init__(self):
        super().__init__('evaluate_ros')
        self.status = None
        self.create_subscription(String, '/edge_residual/status', self.on_status, 10)
        self.control = self.create_publisher(String, '/edge_residual/control', 10)

    def on_status(self, m):
        self.status = json.loads(m.data)

    def set_weather(self, name):
        m = String()
        m.data = json.dumps({'cmd': 'weather', 'value': name})
        self.control.publish(m)

    def spin_until(self, cond, timeout):
        end = time.time() + timeout
        while time.time() < end and not cond():
            rclpy.spin_once(self, timeout_sec=0.05)
        return cond()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--out', default='ros_eval.json')
    ap.add_argument('--timeout', type=float, default=900.0, help='seconds allowed per weather')
    args = ap.parse_args()
    rclpy.init()
    node = Evaluator()
    if not node.spin_until(lambda: node.status is not None, 30):
        raise SystemExit('no /edge_residual/status: is the launch running?')
    for cmd in ('auto_weather', 'auto_light'):                  # the sweep changes the weather itself: the automatic changes must be off
        m = String()
        m.data = json.dumps({'cmd': cmd, 'value': {'seconds': 0}})
        node.control.publish(m)
    node.spin_until(lambda: False, 1.0)
    results = {}
    print(f'{"weather":11s} {"goals/1k":>9s} {"coll/1k":>8s} {"energy/goal":>12s} {"ground speed":>13s} {"late answers":>13s}')
    for w in WEATHERS:
        node.set_weather(w)
        node.spin_until(lambda: node.status['weather'] == w, 10)
        node.spin_until(lambda: False, 0.5)
        b = dict(node.status)
        if not node.spin_until(lambda: node.status['step'] - b['step'] >= args.steps, args.timeout):
            print(f'{w}: timed out after {args.timeout:.0f} s'); break
        e = node.status
        steps, goals, coll = e['step'] - b['step'], e['goals'] - b['goals'], e['collisions'] - b['collisions']
        r = dict(steps=steps, goals=goals, collisions=coll, goals_per_1k=round(1000 * goals / steps, 2), collisions_per_1k=round(1000 * coll / steps, 1),
                 energy_per_goal=round((e['energy'] - b['energy']) / goals, 1) if goals else None, ground_speed=round((e['ground_dist'] - b['ground_dist']) / steps, 2),
                 late_answers=e['stale'] - b['stale'], respawns=e['respawns'] - b['respawns'])
        results[w] = r
        print(f'{w:11s} {r["goals_per_1k"]:9.2f} {r["collisions_per_1k"]:8.1f} {str(r["energy_per_goal"]):>12s} {r["ground_speed"]:13.2f} {r["late_answers"]:13d}', flush=True)
    json.dump(dict(world_seed=node.status['world_seed'], sim_hz=node.status['hz'], steps_per_weather=args.steps, results=results), open(args.out, 'w'), indent=2)
    print('wrote', args.out)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
