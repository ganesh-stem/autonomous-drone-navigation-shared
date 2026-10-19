"""
Does the C++ ray casting (pybind11) change anything, and how much faster is it? Flies the brain through 5 weathers x 10,000 steps (world + policy, no ROS) and prints
the speed and a SHA-256 over every state, reward and position of the run. Run it twice: once with the built fastray*.so next to core.py, once without (numpy).
The two hashes must be identical.

    python tools/check_fastray.py
"""
import sys, time, hashlib, numpy as np
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'edge_residual_adaptive'))
from edge_residual_adaptive import core
from edge_residual_adaptive.residual_policy import ResidualAdaptiveActor
actor = ResidualAdaptiveActor(str(pathlib.Path(__file__).resolve().parents[1] / 'edge_residual_adaptive' / 'weights' / 'residual_adaptive_actor.npz'))
h = hashlib.sha256(); total = 0.0; steps = 0
for weather in ['calm', 'breeze', 'windy', 'gale', 'gale_dusty']:
    env = core.EdgeDroneEnvV3(seed=1234, rng_seed=5, weather=weather, wind_aware=True)
    rng = np.random.default_rng(3)
    t = time.perf_counter()
    for k in range(10000):
        s = np.asarray(env.get_state(), dtype=np.float32)
        a = actor.act(s) if k % 5 else int(rng.integers(0, 9))
        env.step(a)
        h.update(s.tobytes()); h.update(np.float64(env.reward).tobytes()); h.update(np.asarray([env.x, env.y, env.z, env.yaw], dtype=np.float64).tobytes())
    total += time.perf_counter() - t; steps += 10000
print('C++' if core._fastray is not None else 'numpy', f'{steps/total:7.0f} steps/s (world + policy)', h.hexdigest()[:20])
