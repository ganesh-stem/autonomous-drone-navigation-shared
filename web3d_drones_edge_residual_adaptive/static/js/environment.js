// Environment: wind streaks + arrow around the drone, and the sky / fog / light that follow weather and time of day.

import * as THREE from 'three';
import { T, scene, sun, sky, hemi, env, lerp } from './core.js';
import { drone } from './drone.js';
import { updateBuildingLight } from './buildings.js';

// ── wind: streaks that blow through the air around the drone, and an arrow ────────
const STREAKS = 260, RANGE = 260;
const streakPos = new Float32Array(STREAKS * 6), streakBase = new Float32Array(STREAKS * 3);
for (let i = 0; i < STREAKS; i++) { streakBase[i * 3] = (Math.random() - 0.5) * RANGE * 2; streakBase[i * 3 + 1] = (Math.random() - 0.3) * RANGE; streakBase[i * 3 + 2] = (Math.random() - 0.5) * RANGE * 2; }
const streaks = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55 }));
streaks.geometry.setAttribute('position', new THREE.BufferAttribute(streakPos, 3)); streaks.frustumCulled = false; scene.add(streaks);
const windArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 30, 0x5ad1ff, 8, 5); scene.add(windArrow);
let windPhase = 0;
function updateWind(dt) {
  const wx = env.wind[0], wy = env.wind[1], wz = env.wind[2], sp = Math.hypot(wx, wy);
  windPhase += dt;
  const flow = T(wx, wy, wz).multiplyScalar(60);                       // three-space drift per second
  streaks.visible = sp > 0.05;
  if (streaks.visible) {
    const o = drone.position, len = 0.5 + sp * 0.7;
    for (let i = 0; i < STREAKS; i++) {
      const bx = streakBase[i * 3], by = streakBase[i * 3 + 1], bz = streakBase[i * 3 + 2];
      let x = ((bx + flow.x * windPhase + RANGE * 3) % (RANGE * 2)) - RANGE, z = ((bz + flow.z * windPhase + RANGE * 3) % (RANGE * 2)) - RANGE;
      const y = by + flow.y * windPhase * 0 ;
      streakPos.set([o.x + x, o.y + y, o.z + z, o.x + x + flow.x * 0.05 * len, o.y + y + flow.y * 0.05 * len, o.z + z + flow.z * 0.05 * len], i * 6);
    }
    streaks.geometry.attributes.position.needsUpdate = true;
  }
  windArrow.visible = sp > 0.05;
  if (windArrow.visible) {
    windArrow.position.copy(drone.position).add(_up26);
    windArrow.setDirection(T(wx, wy, wz).normalize()); windArrow.setLength(14 + sp * 22, 8, 5);
  }
}

// ── look: light level and weather change the sky ──────────────────────────────────
const LOOK = { day: [0xc4e4ff, 0x0a3d91, 0.95, 1.6], dusk: [0xf0a070, 0x2b2a6e, 0.55, 0.7], night: [0x0a1226, 0x02040c, 0.22, 0.12] };
const skyTop = sky.material.uniforms.top.value, skyBottom = sky.material.uniforms.bottom.value;
const _tmp = new THREE.Color(), _greyH = new THREE.Color(0x7d8794), _greyT = new THREE.Color(0x4a5566), _up26 = new THREE.Vector3(0, 26, 0);
function updateLook() {
  const [hor, top, hemiI, sunI] = LOOK[env.light] || LOOK.day;
  const grey = { calm: 0, breeze: 0.05, windy: 0.25, gale: 0.5, gale_dusty: 0.6 }[env.weather] || 0;
  _tmp.set(hor).lerp(_greyH, grey); skyBottom.lerp(_tmp, 0.05); scene.fog.color.copy(skyBottom);
  _tmp.set(top).lerp(_greyT, grey); skyTop.lerp(_tmp, 0.05);
  hemi.intensity = lerp(hemi.intensity, hemiI * (1 - 0.3 * grey), 0.05); sun.intensity = lerp(sun.intensity, sunI * (1 - 0.4 * grey), 0.05);
  scene.fog.far = 2600 - grey * 1500;
  updateBuildingLight(env.light);                                              // the windows light up at dusk and at night
}

export { updateWind, updateLook };
