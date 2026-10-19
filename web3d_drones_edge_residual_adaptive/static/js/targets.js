// Targets on the ground: the sim's cars, people, trees and rocks, kept off roads and scenery (cars drive in the road lanes).

import * as THREE from 'three';
import { T, scene } from './core.js';
import { BLOCK, ROAD_HALF, WALK, city } from './city.js';
import { setParked } from './traffic.js';

// ── targets on the ground: vehicles, people, trees, hot rocks ─────────────────────
const _m = new THREE.Matrix4(), _c = new THREE.Color();
const MAXT = 400;
const hash01 = (x, y) => { const v = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453; return v - Math.floor(v); };   // same object, same look, every time
const SHIRTS = [0xe04848, 0x3f7dd6, 0xf0c23a, 0x49b36a, 0xffffff, 0xa45ad6];
const personMesh = new THREE.InstancedMesh(new THREE.CapsuleGeometry(0.8, 2.2, 4, 8), new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.8 }), MAXT);
const headMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(0.75, 10, 8), new THREE.MeshStandardMaterial({ color: 0xe0b08a, roughness: 0.8 }), MAXT);
// trees: pines (a brown trunk with 3 or 4 stacked green cones),
// each with a soft shadow on the ground
const trunkMesh = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.16, 0.3, 1, 7), new THREE.MeshStandardMaterial({ color: 0x5a3f2a, roughness: 0.95 }), MAXT);
const coneMesh = new THREE.InstancedMesh(new THREE.ConeGeometry(1, 1, 9), new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.9 }), MAXT * 4);
const shadowGeo = new THREE.CircleGeometry(1, 16); shadowGeo.rotateX(-Math.PI / 2);
const shadowMesh = new THREE.InstancedMesh(shadowGeo, new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.22, depthWrite: false }), MAXT);
const rockMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 8, 6), new THREE.MeshStandardMaterial({ color: 0x8a8a8a, roughness: 1 }), MAXT);
for (const m of [personMesh, headMesh, trunkMesh, coneMesh, shadowMesh, rockMesh]) { m.count = 0; m.frustumCulled = false; scene.add(m); }
const _q = new THREE.Quaternion(), _sc = new THREE.Vector3(), _v = new THREE.Vector3();
const distToRoad = (v) => { const m = ((v % BLOCK) + BLOCK) % BLOCK; return Math.min(m, BLOCK - m); };   // distance to the nearest road centre line along one axis
function setTargets(t) {
  if (t.parked) setParked(t.parked);                                              // the parked cars of the car parks around the drone (drawn by traffic.js)
  let i = 0;
  const put = (mesh, k, base, q, dx, dy, dz, sx, sy, sz) => { _v.set(dx, dy, dz).applyQuaternion(q); mesh.setMatrixAt(k, _m.compose(base.clone().add(_v), q, _sc.set(sx, sy, sz))); };
  for (const [x, y] of t.people.slice(0, MAXT)) {
    if (distToRoad(x) < ROAD_HALF + 1 || distToRoad(y) < ROAD_HALF + 1 || city.blocked(x, y)) continue;   // not on the asphalt or in a pond / court / bed
    const base = T(x, y, 0); _q.identity();
    put(personMesh, i, base, _q, 0, 2.3, 0, 1, 1, 1);
    personMesh.setColorAt(i, _c.set(SHIRTS[Math.floor(hash01(y, x) * SHIRTS.length)]));
    put(headMesh, i, base, _q, 0, 4.95, 0, 1, 1, 1);
    i++;
  }
  personMesh.count = headMesh.count = i; personMesh.instanceMatrix.needsUpdate = headMesh.instanceMatrix.needsUpdate = true;
  if (personMesh.instanceColor) personMesh.instanceColor.needsUpdate = true; i = 0;

  // trees: pines. r is the crown radius that the onboard camera image uses. Each tree gets 3 or 4 stacked cones, its own height and shade of green.
  let nt = 0, nc = 0;
  for (const [x, y, r] of t.trees.slice(0, MAXT)) {
    if (distToRoad(x) < ROAD_HALF + WALK + 2 || distToRoad(y) < ROAD_HALF + WALK + 2 || city.blocked(x, y)) continue;   // trees only on open lawn: off the roads/sidewalks and out of courts, plazas, beds, water and paths
    const h = hash01(x, y), h2 = hash01(y, x), h3 = hash01(x + 3, y + 5);
    _q.identity();
    shadowMesh.setMatrixAt(nt, _m.compose(T(x, y, 0.15), _q, _sc.set(r * 1.5, 1, r * 1.5)));
    const trunkH = r * (1.0 + h2 * 0.5), tiers = h < 0.5 ? 3 : 4, tall = 1.0 + h3 * 0.4;
    trunkMesh.setMatrixAt(nt, _m.compose(T(x, y, trunkH / 2), _q, _sc.set(r * 0.8, trunkH, r * 0.8)));
    for (let k = 0; k < tiers; k++) {
      const rad = r * (1.3 - (0.9 / tiers) * k) , hh = r * 1.5 * tall;
      coneMesh.setMatrixAt(nc, _m.compose(T(x, y, trunkH * 0.5 + k * r * 0.85 * tall + hh / 2), _q, _sc.set(rad, hh, rad)));
      coneMesh.setColorAt(nc, _c.setHSL(0.35 + h3 * 0.04, 0.5, 0.15 + 0.05 * k + h2 * 0.03)); nc++;
    }
    nt++;
  }
  trunkMesh.count = shadowMesh.count = nt; coneMesh.count = nc;
  for (const m of [trunkMesh, shadowMesh, coneMesh]) m.instanceMatrix.needsUpdate = true;
  if (coneMesh.instanceColor) coneMesh.instanceColor.needsUpdate = true; i = 0;
  for (const [x, y, r] of t.rocks.slice(0, MAXT)) if (distToRoad(x) >= ROAD_HALF + WALK + 2 && distToRoad(y) >= ROAD_HALF + WALK + 2 && !city.blocked(x, y)) rockMesh.setMatrixAt(i++, _m.compose(T(x, y, r * 0.5), _q.identity(), _sc.set(r, r * 0.6, r)));
  rockMesh.count = i; rockMesh.instanceMatrix.needsUpdate = true;
}

export { setTargets, trunkMesh, coneMesh, personMesh, rockMesh };
