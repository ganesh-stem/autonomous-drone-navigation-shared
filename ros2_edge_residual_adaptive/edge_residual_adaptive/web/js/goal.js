// The goal marker (a "stargate"): rim, vortex, arcs, jewels, motes, light pillar, landing pad and the shockwave on arrival.
// goalRing drives the goal's size / tilt / wobble; the obstacle gates copy its tilt and use goalLook.gateRadius.

import * as THREE from 'three';
import { T, scene } from './core.js';

// ── goal marker ────────────────────────────────────────────────────────────────
const goalGroup = new THREE.Group(); scene.add(goalGroup);
// The goal is a "stargate": an emerald rim with a swirling energy vortex, counter-rotating gold arcs, a ring of rim jewels, orbiting
// light motes, a pillar of light down to a pulsing landing pad (and up into the sky), and a shockwave when the drone reaches it.
// goalRing stays the driver of the goal's size / tilt / wobble (the obstacle gates copy its tilt), everything else follows it.
const goalRing = new THREE.Mesh(new THREE.TorusGeometry(1, 0.06, 12, 64), new THREE.MeshBasicMaterial({ color: 0x47ff8c }));
const GOAL_GREEN = 0x47ff8c, GOAL_MINT = 0x9dffc4, GOAL_GOLD = 0xffd76a;
const goalGlow = (color, op, side = THREE.FrontSide) => new THREE.MeshBasicMaterial({ color, transparent: true, opacity: op, blending: THREE.AdditiveBlending, depthWrite: false, side });

function swirlTexture() {                                                        // spiral arms fading outward, on a transparent background
  const c = document.createElement('canvas'); c.width = c.height = 512; const g = c.getContext('2d');
  g.translate(256, 256);
  const outer = 256 / 0.96;                                                      // RingGeometry maps its OUTER radius to the texture edge
  for (let arm = 0; arm < 5; arm++) for (let pass = 0; pass < 2; pass++) {
    g.beginPath();
    for (let i = 0; i <= 80; i++) { const t = i / 80, a = arm * 1.2566 + t * 3.4, r = (0.2 + 0.74 * t) * outer; i ? g.lineTo(Math.cos(a) * r, Math.sin(a) * r) : g.moveTo(Math.cos(a) * r, Math.sin(a) * r); }
    g.strokeStyle = pass ? 'rgba(235,255,245,0.85)' : 'rgba(80,255,170,0.35)'; g.lineWidth = pass ? 3 : 16; g.lineCap = 'round'; g.stroke();
  }
  const grad = g.createRadialGradient(0, 0, outer * 0.18, 0, 0, outer * 0.96);   // soft energy haze between the arms
  grad.addColorStop(0, 'rgba(70,255,170,0)'); grad.addColorStop(0.5, 'rgba(70,255,170,0.20)'); grad.addColorStop(1, 'rgba(70,255,170,0.05)');
  g.fillStyle = grad; g.beginPath(); g.arc(0, 0, outer * 0.96, 0, 6.283); g.fill();
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}
function fadeTexture(brightAtTop) {                                              // vertical alpha gradient for the light pillar
  const c = document.createElement('canvas'); c.width = 4; c.height = 256; const g = c.getContext('2d');
  const gr = g.createLinearGradient(0, 0, 0, 256);
  gr.addColorStop(0, brightAtTop ? 'rgba(255,255,255,0.7)' : 'rgba(255,255,255,0)'); gr.addColorStop(1, brightAtTop ? 'rgba(255,255,255,0.04)' : 'rgba(255,255,255,0.7)');
  g.fillStyle = gr; g.fillRect(0, 0, 4, 256);
  return new THREE.CanvasTexture(c);
}

const goalFace = new THREE.Group();                                              // unit-radius group that takes the ring's size, tilt and wobble
const goalSpin = new THREE.Group(); goalFace.add(goalSpin);
const goalHalo = new THREE.Mesh(new THREE.TorusGeometry(1, 0.22, 10, 64), goalGlow(GOAL_GREEN, 0.14));
const goalVortex = new THREE.Mesh(new THREE.RingGeometry(0.2, 0.96, 80, 1), new THREE.MeshBasicMaterial({ map: swirlTexture(), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
const goalArcA = new THREE.Mesh(new THREE.TorusGeometry(0.84, 0.035, 8, 56, Math.PI * 1.55), goalGlow(GOAL_GOLD, 0.95));
const goalArcB = new THREE.Mesh(new THREE.TorusGeometry(0.7, 0.03, 8, 56, Math.PI * 1.1), goalGlow(GOAL_MINT, 0.9));
const goalArcC = new THREE.Mesh(new THREE.TorusGeometry(1.1, 0.025, 8, 56, Math.PI * 0.55), goalGlow(GOAL_GOLD, 0.9));
const goalShock = new THREE.Mesh(new THREE.TorusGeometry(1, 0.05, 8, 64), goalGlow(0xffffff, 0));    // expanding ring when a goal is reached
goalFace.add(goalHalo, goalArcA, goalArcB, goalArcC, goalShock); goalSpin.add(goalVortex);
const goalJewels = [];                                                           // 12 faceted gems around the rim, drifting slowly
const jewelGeo = new THREE.OctahedronGeometry(0.075, 0);
for (let i = 0; i < 12; i++) { const jm = new THREE.Mesh(jewelGeo, new THREE.MeshBasicMaterial({ color: i % 3 === 0 ? GOAL_GOLD : GOAL_MINT })); goalSpin.add(jm); goalJewels.push(jm); }
const goalMotes = [];                                                            // light motes orbiting through the ring on tilted orbits
const moteGeo = new THREE.SphereGeometry(0.045, 8, 6);
for (let i = 0; i < 9; i++) { const mm = new THREE.Mesh(moteGeo, goalGlow(i % 2 ? GOAL_MINT : GOAL_GOLD, 0.95)); goalFace.add(mm); goalMotes.push(mm); }

const PILLAR_R = 2.6;
const goalPillarDown = new THREE.Mesh(new THREE.CylinderGeometry(PILLAR_R, PILLAR_R * 1.5, 1, 20, 1, true), new THREE.MeshBasicMaterial({ color: GOAL_GREEN, map: fadeTexture(true), transparent: true, depthWrite: false, side: THREE.DoubleSide }));
const goalPillarUp = new THREE.Mesh(new THREE.CylinderGeometry(PILLAR_R * 0.7, PILLAR_R, 1, 20, 1, true), new THREE.MeshBasicMaterial({ color: GOAL_MINT, map: fadeTexture(false), transparent: true, depthWrite: false, side: THREE.DoubleSide }));
const goalPad = new THREE.Group();                                               // landing pad where the pillar meets the ground: two rings + a ripple that keeps expanding
const padRing = (r, tube, color, op) => new THREE.Mesh(new THREE.TorusGeometry(r, tube, 8, 64).rotateX(Math.PI / 2), goalGlow(color, op));
const padA = padRing(26, 0.7, GOAL_GREEN, 0.8), padB = padRing(15, 0.5, GOAL_GOLD, 0.7), padRipple = new THREE.Mesh(new THREE.RingGeometry(0.92, 1, 64).rotateX(-Math.PI / 2), goalGlow(GOAL_MINT, 0.5, THREE.DoubleSide));
const padDisc = new THREE.Mesh(new THREE.CircleGeometry(25, 48).rotateX(-Math.PI / 2), goalGlow(GOAL_GREEN, 0.10));
goalPad.add(padA, padB, padRipple, padDisc);
const goalRisers = [];                                                           // rings that climb the pillar toward the goal
for (let i = 0; i < 4; i++) { const rm = new THREE.Mesh(new THREE.TorusGeometry(PILLAR_R * 2.1, 0.22, 6, 32).rotateX(Math.PI / 2), goalGlow(GOAL_MINT, 0.6)); goalRisers.push(rm); goalGroup.add(rm); }
goalGroup.add(goalRing, goalFace, goalPillarDown, goalPillarUp, goalPad);

function updateGoal(g, tSec, dt) {                                               // called every frame after goalRing's size / tilt were set
  goalFace.scale.copy(goalRing.scale); goalFace.rotation.copy(goalRing.rotation);
  const sp = tSec;
  goalSpin.rotation.z = sp * 0.35;
  goalVortex.rotation.z = -sp * 0.9;                                              // the vortex turns against the rim
  goalArcA.rotation.z = sp * 1.1; goalArcB.rotation.z = -sp * 1.5 + 1; goalArcC.rotation.z = sp * 0.7 + 3;
  goalHalo.material.opacity = 0.12 + 0.05 * Math.sin(sp * 2.4) + burst * 0.3;
  goalRing.material.color.setHex(GOAL_GREEN).lerp(_wh, burst * 0.7);
  for (let i = 0; i < goalJewels.length; i++) { const a = (i / goalJewels.length) * 6.283, jm = goalJewels[i]; jm.position.set(Math.cos(a) * 1.13, Math.sin(a) * 1.13, 0); jm.rotation.set(sp * 2 + i, sp * 1.4, 0); jm.scale.setScalar(0.85 + 0.35 * Math.sin(sp * 3 + i * 0.9)); }
  for (let i = 0; i < goalMotes.length; i++) {                                    // each mote: its own orbit plane, speed and phase
    const a = sp * (0.8 + (i % 4) * 0.25) + i * 0.7, tilt = (i * 0.61) % 1.4 - 0.2, r = 0.82 + 0.12 * Math.sin(sp * 1.7 + i);
    goalMotes[i].position.set(Math.cos(a) * r, Math.sin(a) * r * Math.cos(tilt), Math.sin(a) * r * Math.sin(tilt) * 0.9);
    goalMotes[i].scale.setScalar(0.8 + 0.5 * Math.sin(sp * 5 + i * 1.3));
  }
  const sh = 1 - burst;                                                           // shockwave: expands from the rim and fades as the burst decays
  goalShock.scale.setScalar(1 + sh * 2.2); goalShock.material.opacity = burst * 0.6;
  const h = Math.max(g.z, 1);                                                     // pillar down to the ground, and a taller fading one up into the sky
  goalPillarDown.scale.set(1, h, 1); goalPillarDown.position.y = -h / 2;
  goalPillarUp.scale.set(1, 700, 1); goalPillarUp.position.y = 350;
  goalPillarDown.material.opacity = 0.5 + 0.12 * Math.sin(sp * 3); goalPillarUp.material.opacity = 0.42 + 0.1 * Math.sin(sp * 3 + 1);   // normal (not additive) blending: additive washes out to white against the bright sky
  goalPad.position.y = -h + 0.5;                                                  // pad sits on the ground under the goal
  padA.rotation.y = sp * 0.5; padB.rotation.y = -sp * 0.8;
  padA.scale.setScalar(1 + 0.05 * Math.sin(sp * 3)); padDisc.material.opacity = 0.08 + 0.05 * Math.sin(sp * 3);
  const rp = (sp * 0.45) % 1; padRipple.scale.setScalar(4 + rp * 26); padRipple.material.opacity = 0.55 * (1 - rp);
  for (let i = 0; i < goalRisers.length; i++) {
    const u = (sp * 0.28 + i / goalRisers.length) % 1;                            // 0 at the ground, 1 at the goal
    goalRisers[i].position.y = -h + u * h; goalRisers[i].scale.setScalar(1 - 0.35 * u); goalRisers[i].material.opacity = 0.65 * Math.sin(u * Math.PI);
  }
}
const _wh = new THREE.Color(0xffffff);

let burst = 0;                                                                   // 1 right after a goal is reached, decays to 0
const goalLook = { gateRadius: 30 };                                             // goal radius with only its gentle breathing -- the "reached" burst is NOT applied to the obstacle gates
export function goalBurst() { burst = 1; }
export function updateGoalMarker(g, gr, dt) {                                    // g = {x, y, z} goal position, gr = goal radius
  goalGroup.position.copy(T(g.x, g.y, g.z));
  const burstPart = burst * 0.5, pulse = 1 + 0.06 * Math.sin(performance.now() / 250) + burstPart; burst = Math.max(0, burst - dt * 2);
  goalLook.gateRadius = gr * (pulse - burstPart); goalRing.scale.setScalar(gr * pulse); goalRing.rotation.x = Math.PI / 2 + Math.sin(performance.now() / 900) * 0.25; goalRing.rotation.y += 0.01;
  updateGoal(g, performance.now() / 1000, dt);
}

export { goalRing, goalLook };
