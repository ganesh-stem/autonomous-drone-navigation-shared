// Cars: one detailed model (three body styles) drawn with instanced meshes, so a whole fleet is a handful of draw calls.
// A car is local x = forward, y = up, z = side. Each style is a list of boxes: lower body, glass cabin, a roof in a lighter shade of the body colour, bumpers,
// four wheels, two headlights and two tail lights (the tail lights glow brighter while braking, the headlights are on when it is dusk or night).

import * as THREE from 'three';
import { scene } from './core.js';

// body = [centre x, y, z, size x, y, z], glass / roof likewise. L = length, W = width.
export const CAR_TYPES = [
  { name: 'sedan', L: 9.4, W: 4.1, body: [0, 1.3, 0, 9.4, 1.5, 4.1], glass: [-0.5, 2.55, 0, 4.9, 1.05, 3.5], roof: [-0.5, 3.14, 0, 4.0, 0.16, 3.3], wheelR: 0.95, wheelX: 3.0, track: 2.0, lightY: 1.5, lightZ: 1.45 },
  { name: 'suv', L: 9.8, W: 4.4, body: [0, 1.55, 0, 9.8, 1.9, 4.4], glass: [-0.3, 3.05, 0, 6.6, 1.2, 3.8], roof: [-0.3, 3.68, 0, 6.0, 0.16, 3.6], wheelR: 1.1, wheelX: 3.1, track: 2.15, lightY: 1.8, lightZ: 1.55 },
  { name: 'van', L: 10.2, W: 4.4, body: [0, 2.0, 0, 10.2, 2.7, 4.4], glass: [3.55, 2.95, 0, 2.6, 1.15, 3.9], roof: [-0.9, 3.42, 0, 8.4, 0.16, 4.3], wheelR: 1.05, wheelX: 3.2, track: 2.15, lightY: 1.7, lightZ: 1.6 },
];
export const CAR_COLORS = [0xd23c3c, 0x2f6fd6, 0xe8c23a, 0xf0f0f0, 0x38a86a, 0x2b2f38, 0xd9772b, 0x8a93a6, 0x7a3fb0, 0x1f9aa8];

const _M = new THREE.Matrix4(), _Q = new THREE.Quaternion(), _P = new THREE.Vector3(), _S = new THREE.Vector3(), _E = new THREE.Euler(), _C = new THREE.Color();
const UP = new THREE.Vector3(0, 1, 0);

export class Fleet {
  constructor(max) {
    this.max = max; this.n = 0;
    const box = new THREE.BoxGeometry(1, 1, 1);
    const wheel = new THREE.CylinderGeometry(1, 1, 1, 14).rotateX(Math.PI / 2);                // axis along the car's side (z)
    const ball = new THREE.SphereGeometry(1, 8, 6);
    const mk = (geo, mat, count) => { const m = new THREE.InstancedMesh(geo, mat, count); m.count = 0; m.frustumCulled = false; scene.add(m); return m; };
    const paint = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.35, metalness: 0.45 });
    this.body = mk(box, paint, max);
    this.roof = mk(box, paint, max);
    this.glass = mk(box, new THREE.MeshStandardMaterial({ color: 0x1c2733, roughness: 0.15, metalness: 0.7 }), max);
    this.bumper = mk(box, new THREE.MeshStandardMaterial({ color: 0x14161a, roughness: 0.8 }), max * 2);
    this.wheels = mk(wheel, new THREE.MeshStandardMaterial({ color: 0x101114, roughness: 0.9 }), max * 4);
    this.head = mk(ball, new THREE.MeshBasicMaterial({ color: 0xffffff }), max * 2);
    this.tail = mk(ball, new THREE.MeshBasicMaterial({ color: 0xffffff }), max * 2);
    this.meshes = [this.body, this.roof, this.glass, this.bumper, this.wheels, this.head, this.tail];
  }

  begin() { this.n = 0; }

  // x, z: position in scene coordinates (z = -world y); yaw: rotation about the vertical axis (= the world heading); braking / lightsOn: lamp states
  add(x, z, yaw, typeIdx, color, braking, lightsOn) {
    if (this.n >= this.max) return;
    const t = CAR_TYPES[typeIdx], i = this.n++, c = Math.cos(yaw), s = Math.sin(yaw);
    _Q.setFromAxisAngle(UP, yaw);
    const put = (mesh, k, lx, ly, lz, sx, sy, sz, col) => {
      _P.set(x + lx * c + lz * s, ly, z - lx * s + lz * c); _S.set(sx, sy, sz);
      mesh.setMatrixAt(k, _M.compose(_P, _Q, _S));
      if (col !== undefined) mesh.setColorAt(k, col);
    };
    put(this.body, i, ...t.body, _C.set(color));
    put(this.roof, i, ...t.roof, _C.set(color).lerp(_WHITE, 0.25));
    put(this.glass, i, ...t.glass);
    put(this.bumper, i * 2, t.L / 2 - 0.1, 0.75, 0, 0.45, 0.7, t.W + 0.1); put(this.bumper, i * 2 + 1, -t.L / 2 + 0.1, 0.75, 0, 0.45, 0.7, t.W + 0.1);
    let k = i * 4;
    for (const wx of [-t.wheelX, t.wheelX]) for (const wz of [-t.track, t.track]) put(this.wheels, k++, wx, t.wheelR, wz, t.wheelR, t.wheelR, 0.8);
    const hc = lightsOn ? 0xfff2c0 : 0xb9b59c, tc = braking ? 0xff2a2a : lightsOn ? 0xc41c1c : 0x8a1414;
    for (const sz of [-1, 1]) {
      put(this.head, i * 2 + (sz > 0 ? 1 : 0), t.L / 2, t.lightY, sz * t.lightZ, 0.42, 0.34, 0.42, _C.set(hc));
      put(this.tail, i * 2 + (sz > 0 ? 1 : 0), -t.L / 2, t.lightY, sz * t.lightZ, 0.4, 0.3, 0.4, _C.set(tc));
    }
  }

  end() {
    const n = this.n, per = [n, n, n, n * 2, n * 4, n * 2, n * 2];
    this.meshes.forEach((m, j) => { m.count = per[j]; m.instanceMatrix.needsUpdate = true; if (m.instanceColor) m.instanceColor.needsUpdate = true; });
  }
}
const _WHITE = new THREE.Color(0xffffff);
void _E;
