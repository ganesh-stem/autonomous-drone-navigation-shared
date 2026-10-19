// The player's drone model, its sensor rays and the flight trail.

import * as THREE from 'three';
import { T, scene, lerp } from './core.js';

// ── the drone: a sleek quadcopter. x = forward (the camera nose), y = up, z = side ──────────────────────────────────
const drone = new THREE.Group();          // world position + yaw
const tilt = new THREE.Group();           // pitch / roll
drone.add(tilt); scene.add(drone);
export const DRONE_COLOR = 0xe9eef5;      // pearl white; the shell turns red while the drone is colliding (main.js)
const bodyMat = new THREE.MeshPhysicalMaterial({ color: DRONE_COLOR, metalness: 0.3, roughness: 0.22, clearcoat: 1, clearcoatRoughness: 0.1 });
const graphite = new THREE.MeshStandardMaterial({ color: 0x1b2028, metalness: 0.6, roughness: 0.35 });
const carbon = new THREE.MeshStandardMaterial({ color: 0x0e1014, metalness: 0.3, roughness: 0.55 });
const glassMat = new THREE.MeshStandardMaterial({ color: 0x07131f, metalness: 0.85, roughness: 0.08 });
const silver = new THREE.MeshStandardMaterial({ color: 0xa3adbb, metalness: 0.9, roughness: 0.3 });
const glow = (c) => new THREE.MeshBasicMaterial({ color: c });
const part = (geo, mat, x, y, z, sx = 1, sy = 1, sz = 1) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); m.scale.set(sx, sy, sz); tilt.add(m); return m; };
const ball = new THREE.SphereGeometry(1, 28, 18), cube = new THREE.BoxGeometry(1, 1, 1), cyl = new THREE.CylinderGeometry(1, 1, 1, 20);

part(ball, bodyMat, 0, 0, 0, 5.6, 1.5, 2.7);                                      // streamlined hull
part(ball, graphite, -0.9, 1.0, 0, 2.7, 0.85, 1.7);                                // battery hatch on top
part(ball, glassMat, 4.0, -0.05, 0, 2.3, 1.0, 1.9);                                // nose sensor dome
part(cube, glow(0x32e6ff), 5.55, 0.3, 0, 0.22, 0.3, 3.0);                          // cyan light visor across the nose
for (const z of [-1, 1]) part(cube, glow(0x32e6ff), 0.4, 0.62, z * 2.15, 3.6, 0.1, 0.16);        // accent lines along the shoulders
part(cube, glow(0x0a1f8f), -5.5, 0.1, 0, 0.2, 0.28, 2.4);                          // rear light bar (navy blue)
part(cyl, graphite, 3.0, -0.65, 0, 0.18, 1.3, 0.18);                               // gimbal arm
part(ball, graphite, 3.0, -1.45, 0, 1.3, 1.3, 1.3);                                // gimbal housing
part(ball, glassMat, 3.85, -1.5, 0, 0.75, 0.75, 0.75);                             // camera lens
const lensRing = part(new THREE.TorusGeometry(1, 0.14, 8, 28), glow(0x32e6ff), 4.1, -1.5, 0, 0.62, 0.62, 0.62); lensRing.rotation.y = Math.PI / 2;
const belly = part(new THREE.TorusGeometry(1, 0.06, 8, 40), glow(0x32e6ff), 0, -1.0, 0, 2.4, 2.4, 1); belly.rotation.x = Math.PI / 2;       // glowing underside ring
for (const [lx, lz] of [[3.3, 2.2], [3.3, -2.2], [-3.3, 2.2], [-3.3, -2.2]]) { part(cyl, carbon, lx, -0.9, lz, 0.17, 1.8, 0.17); part(cube, carbon, lx, -1.85, lz, 0.9, 0.16, 0.45); }   // landing legs

const rotors = [];                                                                // the spinning propellers (main.js turns them)
const guardGeo = new THREE.TorusGeometry(4.35, 0.17, 8, 44).rotateX(Math.PI / 2);
const bladeGeo = new THREE.BoxGeometry(7.6, 0.07, 0.62);                          // a two-blade propeller: one bar through the hub, like the props of real quadcopters
const discMat = new THREE.MeshBasicMaterial({ color: 0xbfe6ff, transparent: true, opacity: 0.28, depthWrite: false });
for (const [ax, az] of [[7, 7], [7, -7], [-7, 7], [-7, -7]]) {
  const len = Math.hypot(ax, az), arm = part(cube, carbon, ax * 0.55, 0.15, az * 0.55, len * 0.9, 0.5, 1.0);          // carbon arm from the hull to the motor
  arm.rotation.y = -Math.atan2(az, ax);
  part(cyl, graphite, ax, 0.25, az, 1.15, 0.95, 1.15);                           // motor mount
  part(cyl, silver, ax, 0.95, az, 0.85, 0.5, 0.85);                              // motor bell
  part(guardGeo, carbon, ax, 1.2, az);                                            // propeller guard ring
  part(ball, glow(ax > 0 ? (az > 0 ? 0x47ff8c : 0xff4d4d) : 0xffffff), ax, 0.25, az + Math.sign(az) * 1.3, 0.28, 0.28, 0.28);             // navigation light: green / red at the front, white at the back
  const prop = new THREE.Group(); prop.position.set(ax, 1.45, az); tilt.add(prop);
  const blade = new THREE.Mesh(bladeGeo, carbon); blade.rotation.z = 0.12;                                       // a little pitch
  prop.add(blade, new THREE.Mesh(new THREE.SphereGeometry(0.5, 12, 8), silver));
  const disc = new THREE.Mesh(new THREE.CylinderGeometry(3.9, 3.9, 0.05, 28), discMat); disc.position.y = -0.02; prop.add(disc);          // the motion blur of the blades
  rotors.push(prop);
}
const shadow = new THREE.Mesh(new THREE.CircleGeometry(9, 20), new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35 }));
shadow.rotation.x = -Math.PI / 2; scene.add(shadow);

// ── sensors + trail ────────────────────────────────────────────────────────────
const NS = 28;
const sensorLines = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ vertexColors: true }));
sensorLines.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NS * 6), 3));
sensorLines.geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(NS * 6), 3));
sensorLines.frustumCulled = false; scene.add(sensorLines);
const TRAIL_N = 400, trailPos = new Float32Array(TRAIL_N * 3);
const trail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffd76a, transparent: true, opacity: 0.8 }));
trail.geometry.setAttribute('position', new THREE.BufferAttribute(trailPos, 3)); trail.frustumCulled = false; scene.add(trail);
let trailCount = 0;

export function resetTrail() { trailCount = 0; }
export function updateSensors(visible, sensors) {                              // segments from the drone to each ray's end, coloured by proximity
  sensorLines.visible = visible;
  if (!visible) return;
  const pa = sensorLines.geometry.attributes.position, ca = sensorLines.geometry.attributes.color;
  const o = drone.position;
  sensors.forEach((s, i) => {
    const e = T(s[0], s[1], s[2]), heat = Math.min(1, s[3] * 2.5);
    pa.setXYZ(i * 2, o.x, o.y, o.z); pa.setXYZ(i * 2 + 1, e.x, e.y, e.z);
    const r = lerp(0.2, 1, heat), gch = lerp(1, 0.2, heat), b = 0.3;
    ca.setXYZ(i * 2, r, gch, b); ca.setXYZ(i * 2 + 1, r, gch, b);
  });
  pa.needsUpdate = true; ca.needsUpdate = true;
}
export function updateTrail(visible) {
  trail.visible = visible;
  if (!visible) return;
  if (trailCount < TRAIL_N) trailCount++; else trailPos.copyWithin(0, 3);
  const k = trailCount - 1; trailPos[k * 3] = drone.position.x; trailPos[k * 3 + 1] = drone.position.y; trailPos[k * 3 + 2] = drone.position.z;
  trail.geometry.setDrawRange(0, trailCount); trail.geometry.attributes.position.needsUpdate = true;
}

export { drone, tilt, bodyMat, rotors, shadow };
