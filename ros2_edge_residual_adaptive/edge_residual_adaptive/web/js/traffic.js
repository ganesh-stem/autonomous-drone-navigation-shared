// Traffic: draws the cars of the city. The cars are part of the SIMULATION (city_world.py: traffic on the road lattice with lanes, queues, traffic lights, turns, rush hour at dusk,
// few cars at night, and the parked cars of the car parks), so the onboard camera and the detector see exactly the same cars as this view. The server sends
//   "cars"     the driving cars, about 15 times a second: [x, y, heading, type, colour index, braking, speed]
//   "targets"  (with .parked) the parked cars of the blocks around the drone
// and this module only draws them: between two messages each car keeps moving along its heading at its speed, so the motion is smooth.
// The traffic lights are drawn here too, painted from the server's clock (the same 24 s cycle as city_world.py, so the cars and the lights agree).

import * as THREE from 'three';
import { scene, env } from './core.js';
import { drone } from './drone.js';
import { BLOCK, ROAD_HALF } from './city.js';
import { Fleet, CAR_COLORS } from './cars.js';

const STOP = ROAD_HALF + 2.5;
export const fleet = new Fleet(700);
export const trafficBody = fleet.body;

let cars = [], parked = [], serverClock = 0, receivedAt = 0;
const stats = { active: 0, moving: 0, waiting: 0, parked: 0 };
export const trafficStats = () => ({ ...stats, clock: +serverClock.toFixed(1) });
export const trafficData = () => ({ cars, parked });                              // for tests

export function setCars(msg) { cars = msg.cars; serverClock = msg.t; receivedAt = performance.now() / 1000; }
export function setParked(list) { parked = list; }

// ── traffic lights: a 24 s cycle, 0 = green, 1 = yellow, 2 = red (axis 0 = roads that run along x, axis 1 = along y). Identical to signal() in city_world.py.
function signal(axis, t, kx, ky) {
  const p = (t + ((kx * 7 + ky * 13) % 24 + 24) % 24) % 24;                      // every junction has its own phase
  return axis === 0 ? (p < 10 ? 0 : p < 12 ? 1 : 2) : (p < 13 ? 2 : p < 22 ? 0 : 1);
}
const SIG_COLORS = [0x3dff7a, 0xffc933, 0xff3030];
const poles = new THREE.InstancedMesh(new THREE.BoxGeometry(0.5, 1, 0.5).translate(0, 0.5, 0), new THREE.MeshLambertMaterial({ color: 0x3a3f47 }), 700);
const heads = new THREE.InstancedMesh(new THREE.BoxGeometry(1.4, 3, 1.4).translate(0, 0.5, 0), new THREE.MeshLambertMaterial({ color: 0x1b1e24 }), 700);
const bulbs = new THREE.InstancedMesh(new THREE.SphereGeometry(0.62, 8, 6), new THREE.MeshBasicMaterial({ color: 0xffffff }), 700);
for (const m of [poles, heads, bulbs]) { m.count = 0; m.frustumCulled = false; scene.add(m); }
let sigList = [], sigBlock = [NaN, NaN];                                         // [axis, kx, ky] per signal, and the block the list was built for
const _m = new THREE.Matrix4(), _s = new THREE.Matrix4(), _c = new THREE.Color();

function buildSignals(bx, by) {
  let n = 0; sigList = [];
  const R = ROAD_HALF + 2;
  for (let kx = bx - 3; kx <= bx + 4; kx++) for (let ky = by - 3; ky <= by + 4; ky++) {
    const cx = kx * BLOCK, cy = ky * BLOCK;
    // [x, y, axis]: the stop line of each of the four approaches, with the signal on the driver's right-hand kerb
    for (const [x, y, axis] of [[cx - STOP, cy - R, 0], [cx + STOP, cy + R, 0], [cx + R, cy - STOP, 1], [cx - R, cy + STOP, 1]]) {
      _m.makeTranslation(x, 0, -y); poles.setMatrixAt(n, _m.multiply(_s.makeScale(1, 6.2, 1)));
      _m.makeTranslation(x, 5.4, -y); heads.setMatrixAt(n, _m);
      _m.makeTranslation(x, 6.1, -y); bulbs.setMatrixAt(n, _m);
      sigList.push([axis, kx, ky]); n++;
    }
  }
  poles.count = heads.count = bulbs.count = n;
  poles.instanceMatrix.needsUpdate = heads.instanceMatrix.needsUpdate = bulbs.instanceMatrix.needsUpdate = true;
  sigBlock = [bx, by];
}
function paintSignals(t) {
  for (let i = 0; i < sigList.length; i++) { const [axis, kx, ky] = sigList[i]; bulbs.setColorAt(i, _c.setHex(SIG_COLORS[signal(axis, t, kx, ky)])); }
  if (bulbs.instanceColor) bulbs.instanceColor.needsUpdate = true;
}

export function updateTraffic() {
  const now = performance.now() / 1000;
  if (now - receivedAt > 2) cars = [];                                            // no cars message for a while (a replay is on, or the server is gone): show none
  const e = Math.min(0.25, Math.max(0, now - receivedAt));                        // seconds since the last cars message
  const bx = Math.floor(drone.position.x / BLOCK), by = Math.floor(-drone.position.z / BLOCK);
  if (bx !== sigBlock[0] || by !== sigBlock[1]) buildSignals(bx, by);
  paintSignals(serverClock + e);

  const lights = env.light !== 'day';
  stats.active = cars.length; stats.moving = 0; stats.waiting = 0; stats.parked = parked.length;
  fleet.begin();
  for (const [x, y, h, type, col, brake, v] of cars) {
    fleet.add(x + Math.cos(h) * v * e, -(y + Math.sin(h) * v * e), h, type, CAR_COLORS[col], brake === 1, lights);
    if (v > 1) stats.moving++; else stats.waiting++;
  }
  for (const [x, y, h, type, col] of parked) fleet.add(x, -y, h, type, CAR_COLORS[col], false, false);
  fleet.end();
}
