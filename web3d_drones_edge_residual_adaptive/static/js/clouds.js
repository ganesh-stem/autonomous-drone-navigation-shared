// Clouds: a fixed set of puffs that wrap around the drone, so the sky never runs out.

import * as THREE from 'three';
import { scene } from './core.js';

// clouds: a fixed set of puffs that wrap around the drone, so the sky never runs out
const CLOUD_N = 70, CLOUD_BOX = 5000;
const clouds = new THREE.Group();
const cloudMat = new THREE.MeshLambertMaterial({ color: 0xffffff, transparent: true, opacity: 0.8, fog: false });
const cloudBase = [];
{ let s = 12345; const rnd = () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
  for (let i = 0; i < CLOUD_N; i++) {
    const puff = new THREE.Group();
    for (let k = 0; k < 4; k++) {
      const m = new THREE.Mesh(new THREE.SphereGeometry(80 + rnd() * 120, 10, 8), cloudMat);
      m.position.set((rnd() - 0.5) * 320, (rnd() - 0.5) * 60, (rnd() - 0.5) * 240); m.scale.y = 0.45; puff.add(m);
    }
    cloudBase.push([rnd() * CLOUD_BOX, 1100 + rnd() * 1300, rnd() * CLOUD_BOX]); clouds.add(puff);
  } }
scene.add(clouds);

export function updateClouds(p) {                                              // keep the puffs in a box around the drone
  cloudBase.forEach((b, i) => {
    const wx = ((b[0] - p.x) % CLOUD_BOX + CLOUD_BOX) % CLOUD_BOX - CLOUD_BOX / 2;
    const wz = ((b[2] - p.z) % CLOUD_BOX + CLOUD_BOX) % CLOUD_BOX - CLOUD_BOX / 2;
    clouds.children[i].position.set(p.x + wx, b[1], p.z + wz);
  });
}
