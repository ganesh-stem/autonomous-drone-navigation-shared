"""
MAVLink v2 telemetry for the simulated drone.

    tm = Telemetry()                              # the "vehicle" side: encodes the drone state into MAVLink frames
    frames = tm.step_frames(env, step)            # list of bytes: what goes out on the radio this step
    gs = GroundStation()                          # the "ground" side: parses a byte stream, checks the CRC of every frame
    for b in frames: gs.feed(b)
    gs.stats()                                    # frames per message type, CRC errors, lost packets (sequence gaps), positions

Messages (all standard MAVLink common dialect, so QGroundControl or Mission Planner would understand them):
    HEARTBEAT 1 Hz, LOCAL_POSITION_NED 10 Hz, ATTITUDE 10 Hz, SYS_STATUS 2 Hz (battery), WIND_COV 2 Hz, NAMED_VALUE_FLOAT 1 Hz (distance to goal)
Coordinates: world x = east, world y = north, z = up, so NED is (north = y, east = x, down = -z). 1 world unit = 1 m. One sim step = 1/20 s.
"""

import math
import time

from pymavlink.dialects.v20 import common as mav2

STEP_S = 0.05


class _Sink:
    """pymavlink wants a file-like object to write into. We collect the bytes ourselves."""
    def write(self, data):
        pass


class Telemetry:
    def __init__(self, sysid=1, compid=1):
        self.mav = mav2.MAVLink(_Sink(), srcSystem=sysid, srcComponent=compid)
        self.mav.robust_parsing = True
        self.bytes_sent = 0
        self.frames_sent = 0
        self.by_type = {}
        self._prev = None

    def _pack(self, msg):
        data = msg.pack(self.mav)
        self.mav.seq = (self.mav.seq + 1) % 256                  # pack() does not advance the sequence number by itself
        self.bytes_sent += len(data)
        self.frames_sent += 1
        self.by_type[msg.get_type()] = self.by_type.get(msg.get_type(), 0) + 1
        return data

    def step_frames(self, env, step):
        """The frames to send after sim step number `step` (0, 1, 2, ...)."""
        t_ms = int(step * STEP_S * 1000)
        out = []
        m = self.mav
        vx = vy = vz = 0.0
        if self._prev is not None:
            dt = STEP_S * max(1, step - self._prev[3])               # more than one step apart when the node only sees every 2nd status
            vx, vy, vz = ((env.x - self._prev[0]) / dt, (env.y - self._prev[1]) / dt, (env.z - self._prev[2]) / dt)
        self._prev = (env.x, env.y, env.z, step)
        if step % 20 == 0:
            out.append(self._pack(m.heartbeat_encode(mav2.MAV_TYPE_QUADROTOR, mav2.MAV_AUTOPILOT_GENERIC,
                                                     mav2.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 0, mav2.MAV_STATE_ACTIVE)))
        if step % 2 == 0:
            out.append(self._pack(m.local_position_ned_encode(t_ms, env.y, env.x, -env.z, vy, vx, -vz)))
            yaw = math.radians(90.0 - env.yaw)                       # heading from north, clockwise
            out.append(self._pack(m.attitude_encode(t_ms, 0.0, 0.0, (yaw + math.pi) % (2 * math.pi) - math.pi, 0.0, 0.0, 0.0)))
        if step % 10 == 0:
            batt = int(round(getattr(env, 'battery_percent', 100.0)))
            out.append(self._pack(m.sys_status_encode(0, 0, 0, 500, int(12600 * batt / 100.0 + 9000 * (1 - batt / 100.0)), -1, batt, 0, 0, 0, 0, 0, 0)))
            w = getattr(env, 'wind', (0.0, 0.0, 0.0))
            out.append(self._pack(m.wind_cov_encode(t_ms * 1000, float(w[1]), float(w[0]), float(-w[2]), 0.0, 0.0, 0.0, 0.0, 0.0)))
        if step % 20 == 10:
            out.append(self._pack(m.named_value_float_encode(t_ms, b'goal_dist', float(env.distance))))
        return out


class GroundStation:
    def __init__(self):
        self.mav = mav2.MAVLink(_Sink())
        self.mav.robust_parsing = False
        self.frames = 0
        self.by_type = {}
        self.crc_errors = 0
        self.lost = 0
        self._last_seq = None
        self.positions = []          # (t_ms, north, east, down)
        self.battery = None
        self.wind = None

    def feed(self, data):
        """A whole, well-formed frame (the usual case) goes to the parser in one piece, which is about 3 times faster. Anything odd (a wrong length, a
        cut or a prefixed frame, or half a frame still waiting in the parser) goes byte by byte as before, so the counters are the same either way."""
        if len(data) >= 12 and data[0] == 0xFD and len(data) == 12 + data[1] + (13 if data[2] & 1 else 0) and self.mav.buf_len() == 0:
            self._take(data)
        else:
            for byte in data:
                self._take(bytes([byte]))

    def _take(self, chunk):
        try:
            msg = self.mav.parse_char(chunk)          # one call, as with one byte: it gives at most one message, or raises for a damaged one
        except mav2.MAVError:
            self.crc_errors += 1
            return
        if msg is None:
            return
        t = msg.get_type()
        if t == 'BAD_DATA' or t.startswith('UNKNOWN'):      # a damaged message id: no valid frame, its sequence number is garbage
            self.crc_errors += 1
            return
        self.frames += 1
        self.by_type[t] = self.by_type.get(t, 0) + 1
        seq = msg.get_seq()
        if self._last_seq is not None:
            self.lost += (seq - self._last_seq - 1) % 256
        self._last_seq = seq
        if t == 'LOCAL_POSITION_NED':
            self.positions.append((msg.time_boot_ms, msg.x, msg.y, msg.z))
        elif t == 'SYS_STATUS':
            self.battery = msg.battery_remaining
        elif t == 'WIND_COV':
            self.wind = (msg.wind_x, msg.wind_y, msg.wind_z)

    def stats(self):
        return dict(frames=self.frames, by_type=self.by_type, crc_errors=self.crc_errors, lost_packets=self.lost,
                    battery=self.battery, wind=self.wind)


if __name__ == '__main__':
    # tiny self-test: 400 steps of a calm flight, round trip through the ground station
    from env_edge import EdgeDroneEnv
    env = EdgeDroneEnv(weather='gale', wind_aware=False)
    tm, gs = Telemetry(), GroundStation()
    t0 = time.time()
    for i in range(400):
        env.step(0)
        for f in tm.step_frames(env, i):
            gs.feed(f)
    print('sent', tm.frames_sent, 'frames', tm.bytes_sent, 'bytes; ground station:', gs.stats())
    n, e, d = gs.positions[-1][1:]
    print(f'last position north {n:.2f} east {e:.2f} down {d:.2f}   truth north {env.y:.2f} east {env.x:.2f} down {-env.z:.2f}')
